# deepseek_learning - ongoing
一个快速学习deepseek v3模型以及r1强化学习的仓库，侧重快速理解技术报告模型设计细节

# 一. v3
技术报告与代码来自[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)

学习讲解资料来自[EZ.Encoder](https://www.youtube.com/@ez.encoder.academy)

## 代码说明
deepseekV3开源的推理代码兼容分布式多卡推理，将embedding, MLA, MOE模块中的线性层平分到所有的卡中，使得小显存卡也能组合起来运行。在阅读代码时需要注意`// word_size`, `dist.all_reduce()`等分布式推理的map-reduce的操作。

可以按照v3/model.py阅读代码，建议先跳过linear函数中量化与反量化操作

## 模型结构
### MLA 多头低秩注意力
背景：传统模型在推理时使用kv缓存进行加速，当文本越长需要的显存消耗越大（线性），MLA通过优化kv缓存机制，降低推理需要的显存消耗

传统kv缓存生成方式：当预测得到下一个token，将其添加到历史输入进行前向传播，仅对该token生成k-v键值，如`k = h * Wk`，加入到历史token的k-v矩阵后进行前向计算，存储用于下一次计算

MLA优化原理：存储kv变成存储一个更低维的中间态`kv = h * Wkv_a`, `Wkv_a`对h进行降维。 后续再由kv乘以一个矩阵得到v矩阵。

MLA其他创新：ROPE旋转位置编码按照维度进行解耦，划分无位置编码与有位置编码部分，最后计算softmax前对没有位置编码和有位置编码的查询结果进行相加。有点类似residual短连接效果，同时位置标记只由部分维度执行，减少全局影响的同时提高计算效率


### MOE + loss-free的专家均衡策略
MOE原理：
MOE模块设置了共享专家和专精专家（routed_experts），每一个专家都是一个小的MLP，专精专家由一个路由gate进行分配(线性层，输出维度为model dim, 输出维度为专精专家数)，激活topk的专精专家进行前向传播。

MOE模块预设了分组过滤，先过滤组，再从组选择topk专家。最后加总共享专家和专精专家的结果。

MOE训练弊端：gate的分配难以控制，可能导致分配偏向少数专精专家

MOE训练优化：
        在损失函数中增加专精专家的分配不均衡损失，这种方法会影响对其他损失函数的关注度。
        根据专精专家的历史分配次数，调整gate分类头输出的softmax分类logit，提高少分配专家的logit。--- loss-free，deepseek使用这种方法？
           
## 训练
### MTP 多token预测监督 
背景：基于decoder的问答模型只监督下一token的预测，但人在进行语言组织输出时通预见常不止一个文字，可能是一个词、一个句柄或一小段话，这更有利与语言组织的准确性。另一方面，下一token的输出出现偏离很容易会导致回答跑飞，考虑下下token，下下下token的可能性有利于当前token预测的纠偏。因此这种带有一定跨度的预见直观上可以进一个步提高监督的强度。

deepseekV3技术报告中的MTP模块与主模型共享emedding层、输出头。主模型与上一个MTP模块在输出头前的隐态输出到下一个MTP模块，与输入的embedding直接进行相加。

补充：MTP经典方式：并行与串行

并行的MTP是指隐藏态同时喂入多个输出头进行，每个输出头对应的gt是依次往后偏移的，如input为`t0 t1 t2 t3`，head1的gt为`t1 t2 t3 t4`，head2的gt为`t2 t3 t4 t5`，依次类推....；串行的MTP则是上一个head的输出作为下一个head的输入(之一)，计算head输出与对应gt损失

并行的MTP并行地将隐态编码成依次偏移的gt预测，但推理时仍然是预测下一个token，直觉上并行的MTP是不符合推理模式的。串行的MTP结构最开始用于推理的解码过程，用来提高推理的速度，这里在训练时采用，更符合推理模式。

deepseek采用的方式 [图]

### fp8量化操作
开源的推理代码定义了bf16和fp8两种量化推理模式，在linear函数中进行量化或反量化操作，以下只对量化相关的操作进行通俗解读。

注意这一部分适用于理解如何进行自定义量化推理，而量化训练的代码应该还包含更多反向传播的操作

总体上，使用了triton编写矩阵计算内核，用于并发执行分块矩阵乘法。基本思路是每个进程根据id取数，进行运算，结果写入预定义的内存地址，只覆盖进程id对应的块。所有进程执行完毕，则得到最终的结果。

首先定义了一个网格，用于启动并行执行线程，每个线程获得对应网格点对应的id（一维网格只用序数，二维有行列序数）

以执行fp8反量化矩阵乘法为例进行说明
```python
# 定义二维网格，(M, N)表示原始网格行列，（BLOCK_SIZE_M, BLOCK_SIZE_N)表示分块小网格的行列, triton.cdiv表示向上取整
# 最终进程获得的id为（i，j），对应分块后的位置，映射到原始网格的位置为(i * BLOCK_SIZE_M, j * BLOCK_SIZE_N)
grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))    
# 按照网格启动进程，执行分块矩阵乘法。每个进程的id是二维的
fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
"""
a为token向量, b为线性层参数矩阵, c为存储矩阵, a_s为a的缩放参数矩阵, b_s为线性层缩放参数矩阵, 是按照block_size * block_size二维分块进行量化的
M为a的token行数, N为b的列数, K为a的token维度
"""
```

然后，线程执行核函数，根据id从输入矩阵中取数，这个取数逻辑需要考虑输入矩阵本身是如何进行分块量化的
```python
# 进程分配的行id
pid_m = tl.program_id(axis=0)                                              
# 进程分配的列id
pid_n = tl.program_id(axis=1)
# 计算每个token的维度上进行fp8量化的块数量                                             
k = tl.cdiv(K, BLOCK_SIZE_K)                                    
# 计算当前进程在行维度的连续块的绝对位置；一维；百分号用于超边界处理
offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
# 计算当前进程在列维度的连续块的位置；
offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
# 量化块的连续位置 array([0, 1, 2, ..., BLOCK_SIZE_K])
offs_k = tl.arange(0, BLOCK_SIZE_K)

# 取出a对应的分块的内存指针，二维，形状为(BLOCK_SIZE_M，BLOCK_SIZE_K)
a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
# 取出b对应的分块的内存指针，二维，形状为(BLOCK_SIZE_K, BLOCK_SIZE_N)
b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]                      
# 取出a量化缩放矩阵，一维，形状（BLOCK_SIZE_M, 1)，对应每个token第一个量化块的缩放系数
a_s_ptrs = a_s_ptr + offs_m * k 
# 取出线性层参数矩阵b的量化参数矩阵，这里b是按照(block_size * block_size)二维块量化                                    
b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k 

for i in range(k):   # 遍历token行，以block_size作为步长                                                        
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0) # mask为掩码处理超边界填充other
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)        
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :] # 求两者内积，再乘上两者放缩系数
        a_ptrs += BLOCK_SIZE_K                                    # 指针偏移blocksize，即行块沿行移动blocksize步长
        b_ptrs += BLOCK_SIZE_K                                    # 同理
        a_s_ptrs += 1                                             # 放缩参数指针偏移一位, 放缩参数矩阵的列数为k
        b_s_ptrs += 1                                             # 放缩参数指针偏移一位, 放缩参数矩阵的列数为k
```

最后，对应数值进行目标矩阵乘法，结果写入预定义的存储地址，每个进程只写入自己对应的的块
```python
c = accumulator.to(c_ptr.dtype.element_ty)                                   # 同步数据格式
offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]                       # 定位存储区域对应的块
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, c, mask=mask)                                                
```

注意，要进入`@triton.jit`编译的核函数可以使用`import pdb`，在对应的断点位置写一行`pdb.set_trace()`，运行文件后则会进入核函数，可以通过命令行进行交互[pdb](https://github.com/HarleysZhang/llm_note/blob/main/4-hpc_basic/%E7%90%86%E8%A7%A3triton%E5%86%85%E6%A0%B8%E6%95%99%E7%A8%8B1.md)


### 加速通信方法
comming soon

# 二. r1
r1的使用v3作为基础模型，训练增加了强化学习的策略，使得在标签数据较少的情况下也能学习到知识并泛化
