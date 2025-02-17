# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import sys
import os
p = os.path.dirname(__file__)
sys.path.insert(0, p.split('/example')[0])
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


"""
python -i examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""

args = [{
    'dataset_name': 'trl-internal-testing/descriptiveness-sentiment-trl-style',
    'dataset_train_split': 'descriptiveness',
    'output_dir': 'models/minimal/ppo' ,
    'num_ppo_epochs': 1 ,
    'num_mini_batches': 1 ,
    'learning_rate': 3e-6 ,
    'per_device_train_batch_size': 1 ,
    'gradient_accumulation_steps': 16 ,
    'total_episodes': 10000 ,
    'model_name_or_path': 'EleutherAI/pythia-1b-deduped',
    'sft_model_path': 'EleutherAI/pythia-1b-deduped' ,
    'reward_model_path': 'EleutherAI/pythia-1b-deduped',
    'local_rollout_forward_batch_size': 1 ,
    'missing_eos_penalty': 1.0,
}]

args =  """--dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
    """
# print(args)
args = """--dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0
    """
a = 1
args = [s.strip() for s in args.split('  ')]
args = [s for s in args if s != '']
args_list = []
for s in args:
    args_list.extend(s.split(' '))
if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses(args=args_list)
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()

    """
    PPO要素介绍
    策略模型 : 训练目标
    参考模型 : 基于规则或基于模型，输出参考策略，用于对照输出策略，避免每次输出策略偏离过大
    奖励模型 : 对生成的query-response进行打分, response 来自于策略模型自回归生成。奖励模型应该是可提供训练偏好指导的
    价值模型 : 基于模型，评估在已生成query-response(t token)，生成的第t+1个token的价值 (st 状态下)
    T : 轨迹抽样长度，对应最大的生成序列长度
    策略分布 : 自回归每次下一个token，logits的分布对应策略分布
    状态st : 当前输入的prompt data + 历史生成了的第t-1个token

    trl库实现流程:
    01 生成响应: 策略模型根据prompt自回归生成完整，长度不超过T。和query拼接得到query-response，同时得到对应的logitprobs。获得采样轨迹
    02 生成参考: 参考模型根据query-response进行前向传播得到ref_logitprobs
    03 奖励打分: 奖励模型对query-response进行打分，每个pair对应一个分数
    04 计算价值: 价值模型对每一个response进采样轨迹评估每一个阶段的价值, score[t]表示做出t时刻决策前的价值评估
    05 损失计算
        - 决策收益
            -- t时刻决策优势：zt = t时刻决策奖励 - t时刻决策前价值 + 衰减系数 * (t + 1)时刻决策前价值
            -- t时刻决策优势期望：At = zt + d(t + 1) * z(t + 1) + ... + d(T) * z(t), d为和采样时间步相关的衰减函数
            - 决策期望收益：累加(new_policy(t | st) / old_policy(t | st) * At)
        - kl损失
            - 计算采样轨迹的new logprobs和old logprobs偏离程度
        - 价值偏离 ==== 如何理解这个价值偏离
            -- 目标价值 = 模型t时刻决策 + t时刻前的价值
            - 价值偏离损失：forward计算的V[t]应该与generate得到的价值Vtarget[t]是相近的
            -- 减小模型training过程与inference过程的偏离
    """
