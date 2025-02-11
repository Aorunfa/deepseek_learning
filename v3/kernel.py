from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config
import pdb


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)                         # 一维网格点的序号
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算绝对位置
    x = tl.load(x_ptr + offs).to(tl.float32)            # 取数，一维
    s = tl.max(tl.abs(x)) / 448.                        # 计算放缩系数， fp8最大正范围448
    y = x / s                                           # 量化
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)                           
    tl.store(s_ptr + pid, s)                            

    # pdb.set_trace()

"""执行fp8量化"""
def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """

    # 编译后，按照grid网格函数定位对应网格序号进行取数和计算
    pid_m = tl.program_id(axis=0)                          # 当前网格行序号
    pid_n = tl.program_id(axis=1)                          # 当前网格列序号
    n = tl.cdiv(N, BLOCK_SIZE)                             # 每个网格列跨度
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # 网格元素内存指针的的行偏移
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # 网格元素内存指针的的列偏移
    offs = offs_m[:, None] * N + offs_n[None, :]           # 先按行定位行绝对位置，再加上偏移 
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)     # 根据权重起始指针和二维偏移获得权重
    s = tl.load(s_ptr + pid_m * n + pid_n)                  # 一维指针，加载一个放缩系数。想象一个二维矩阵，第一个元素从s_ptr开始，按行偏移pid_m * n个缩放系数位置，再向右取当前第pid_n个缩放系数
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)                    # 存储在缓冲区


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))                        # 定义网格分块函数，多线程并行计算结果  以(24, 16)分块， 网格每一个点分配给一个线程去执行操作
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)  # num_stages和num_warps为并行参数
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K']) # 定义自动调优函数，根据核函数传入的N，K进行自动选择
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)                                              # token grid index
    pid_n = tl.program_id(axis=1)                                              # 输出维度 grid index
    k = tl.cdiv(K, BLOCK_SIZE_K)                                               # 输入维度按
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # pdb.set_trace()

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]                      # 沿第二个轴定位token绝对位置，加上blocksize偏移, 得到二维指针。shape: _ ; 16,1 ; 1,128
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]                      # 沿第一个轴定位参数矩阵行绝对位置，加上blocksize偏移, 得到二维指针。shape:_ ; 1,32 ; 128, 1
    a_s_ptrs = a_s_ptr + offs_m * k                                             # 取对应block的放缩系数, k表示每行划分blocksize的个数，得到一维指针
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k                           # 同理取对应block的放缩系数，得到一维指针                     

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):   # 遍历token行，以block_size作为步长                                                        
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)        
        b_s = tl.load(b_s_ptrs)
        
        pdb.set_trace()

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]               # 求两者内积，再乘上两者放缩系数
        a_ptrs += BLOCK_SIZE_K                                                  # 指针偏移blocksize
        b_ptrs += BLOCK_SIZE_K                                                  # 指针偏移blocksize
        a_s_ptrs += 1                                                           # 放缩参数指针偏移一位, 放缩参数矩阵的列数为k
        b_s_ptrs += 1                                                           # 放缩参数指针偏移一位, 放缩参数矩阵的列数为k
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]                       # 取所有指针
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)    
    

def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)                                                                  # 输入维度
    M = a.numel() // K                                                              # token总数
    N = b.size(0)                                                                   # 输出维度
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())             # 存储结果
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))    # 网格按照token数和输出维度进行划分(最终输出形状)
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)                                                   # 按照网格启动并发任务，执行分块矩阵乘法
    return c


if __name__ == '__main__':
    pass

    """
    trion 基础操作与调试学习资料
    https://github.com/HarleysZhang/llm_note/blob/main/4-hpc_basic/%E7%90%86%E8%A7%A3triton%E5%86%85%E6%A0%B8%E6%95%99%E7%A8%8B1.md
    """
