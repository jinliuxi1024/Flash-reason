from transformers import AutoConfig, AutoModel, AutoModelForCausalLM,AutoTokenizer
from deepseek_v3.configuration_deepseek import DeepseekV3Config
from deepseek_v3.modeling_deepseek import DeepseekV3Model, DeepseekV3ForCausalLM
# 进行动态注册
AutoConfig.register("deepseek_v3", DeepseekV3Config)
AutoModel.register(DeepseekV3Config, DeepseekV3Model)
AutoModelForCausalLM.register(DeepseekV3Config, DeepseekV3ForCausalLM)
import torch
# ================== 配置模块 ==================
config =DeepseekV3Config(
        vocab_size=129280,
        hidden_size=1024,
        intermediate_size=2560,
        moe_intermediate_size = 512,
        num_hidden_layers=8,
        num_nextn_predict_layers=1,
        num_attention_heads=12,
        num_key_value_heads=12,
        n_shared_experts = 1,
        n_routed_experts = 16,
        ep_size = 1,
        routed_scaling_factor = 2.5,
        kv_lora_rank = 64,
        q_lora_rank = 128,
        qk_rope_head_dim = 16,
        v_head_dim = 48,
        qk_nope_head_dim = 48,
        topk_method = 'noaux_tc',
        n_group = 2,
        topk_group = 1,
        num_experts_per_tok = 4,
        moe_layer_freq = 1,
        first_k_dense_replace = 1,
        norm_topk_prob = True,
        scoring_func = 'sigmoid',
        aux_loss_alpha = 0.001,
        seq_aux = True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 2048,
        "type": "yarn"
        },
        attention_bias=False,
        attention_dropout=0.0,
        attn_implementation = "flash_attention_2",
        torch_dtype ="bfloat16",
)

# ================== 模型初始化 ==================
model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained('deepseek_tokenizer')
tokenizer.padding_side = "left"  # Flash Attention必须项
tokenizer.pad_token = tokenizer.eos_token

# ================== 数据预处理 ==================

from datasets import Dataset, concatenate_datasets

#加载数据集
from datasets import load_dataset, concatenate_datasets

# 自定义缓存目录
cache_dir = "pretrain/cache"  # 指定你希望存储缓存文件的路径

# 加载数据集并指定缓存目录
#dataset_1 = load_dataset('pretrain_zh/r1_zh', cache_dir=cache_dir)['train']
#dataset_2 = load_dataset('pretrain_zh/piv_zh', cache_dir=cache_dir)['train']
dataset_3 = load_dataset('pretrain_zh/web_zh', cache_dir=cache_dir)['train']
dataset_4 = load_dataset('pretrain_zh/web_en', cache_dir=cache_dir)['train']
#dataset_5 = load_dataset('pretrain_zh/piv_zh', cache_dir=cache_dir)['train']
dataset_6 = load_dataset('pretrain_zh/r1_zh', cache_dir=cache_dir)['train']
dataset_7 = load_dataset('pretrain_zh/r1_en/data', cache_dir=cache_dir)['train']


# 数据转换函数
def convert_to_text(example, dataset_type):
    # 判断数据集类型，动态处理字段
    if 'problem' in example and 'generations' in example and dataset_type in ['openr1_zh']:
        # 对 chat_zh, code_en, math_en: text = instruction + response
        text = example.get('problem', '') + example.get('generations', '')[0] 
    elif 'content' in example and dataset_type in ['piv_zh','novel_zh']:
        # 对 wiki_zh, piv_zh: text = content
        text = example.get('content', '')
    elif 'reasoning_content' in example and dataset_type in ['r1_zh']:
        # 对 wiki_zh, piv_zh: text = content
        text = example.get('input', '')+example.get('reasoning_content', '')+example.get('content', '')
    elif  'generations' in example and  dataset_type in ['r1_en']:
        # 对 wiki_zh, piv_zh: text = content
        text = example.get('problem', '')+example.get('generations', '')[0]
    elif 'text' in example and dataset_type in ['sk_zh','web_zh']:
        # 对 novel_zh: 保持 text 不变
        text = example.get('text', '')
    else:
        # 默认处理：返回空字符串，避免出错
        text = ''
    
    # 返回统一格式
    return {"text": text}

# 转换每个数据集并只保留 'text' 列
#formatted_dataset_1 = dataset_1.map(lambda x: convert_to_text(x, 'r1_zh')).remove_columns([col for col in dataset_1.column_names if col != 'text'])
#formatted_dataset_2 = dataset_2.map(lambda x: convert_to_text(x, 'novel_zh')).remove_columns([col for col in dataset_2.column_names if col != 'text'])
formatted_dataset_3 = dataset_3.remove_columns([col for col in dataset_3.column_names if col != 'text'])
formatted_dataset_4 = dataset_4.remove_columns([col for col in dataset_4.column_names if col != 'text'])
#formatted_dataset_5 = dataset_5.map(lambda x: convert_to_text(x, 'piv_zh')).remove_columns([col for col in dataset_5.column_names if col != 'text'])
formatted_dataset_6 = dataset_6.map(lambda x: convert_to_text(x, 'r1_zh')).remove_columns([col for col in dataset_6.column_names if col != 'text'])
formatted_dataset_7 = dataset_7.map(lambda x: convert_to_text(x, 'r1_en')).remove_columns([col for col in dataset_7.column_names if col != 'text'])

# 依次检查其他数据集
# 合并数据集
train_dataset = concatenate_datasets([
    #formatted_dataset_1,
    #formatted_dataset_2,
    formatted_dataset_3,
    formatted_dataset_4,
    #formatted_dataset_5,
    formatted_dataset_6,
    formatted_dataset_7,
])



def process_func(examples):
    max_token = 512  # 设置最大长度

    # 用于保存结果的列表
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    # 遍历样本逐个处理
    for content in examples['text']:
        # 对文本进行分词
        encoded_text = tokenizer(
            content,
            truncation=True,  # 启用截断
            max_length=131072,
            return_tensors=None
        )

        # 获取分词后的 input_ids
        input_ids = encoded_text["input_ids"]

        # 按 max_token 长度分割 input_ids
        for i in range(0, len(input_ids), max_token):
            # 取出分段
            chunk_input_ids = input_ids[i:i + max_token]
            chunk_attention_mask = [1] * len(chunk_input_ids)  # 根据分段长度生成 attention_mask

            # 填充到 max_token 长度
            padding_length = max_token - len(chunk_input_ids)
            chunk_input_ids += [tokenizer.pad_token_id] * padding_length
            chunk_attention_mask += [0] * padding_length

            # 复制 input_ids 为 labels
            chunk_labels = chunk_input_ids.copy()

            # 将分段结果添加到批量列表中
            input_ids_batch.append(chunk_input_ids)
            attention_mask_batch.append(chunk_attention_mask)
            labels_batch.append(chunk_labels)

    # 返回批量处理结果
    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch
    }

train_dataset = train_dataset.map(
    process_func,
    batched=True,
    remove_columns=train_dataset.column_names,
)
train_dataset = train_dataset.shuffle(seed=3407)  # 随机打乱
# ================== 训练配置 ==================

from transformers import Trainer, TrainingArguments, get_scheduler
import torch
import math  # 导入 math 模块
import wandb
wandb.login(key="4b54ec64a5118d478b6ac62b059cb5a08247e8b8")
# 定义DeepSpeed配置（动态启用bf16/fp16）
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "contiguous_gradients": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "bf16": {"enabled": torch.cuda.is_bf16_supported()},
    "fp16": {"enabled": not torch.cuda.is_bf16_supported()}
}
training_args = TrainingArguments(
    output_dir='deepseekmini_pretrain',  # 输出路径，包括模型检查点、中间文件等
    overwrite_output_dir=True,  # 是否覆写 output_dir
    do_train=True,  # 是否做训练
    remove_unused_columns=False,
    per_device_train_batch_size=1,  # 每设备批次
    gradient_accumulation_steps=32,  # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快
    learning_rate=2e-4,  # 学习率大小
    lr_scheduler_type='cosine',  # 学习率调度策略，LLM 训练一般都用余弦
    weight_decay=0.01,  # 权重衰减
    bf16=torch.cuda.is_bf16_supported(),  # 尝试配置 bf16
    fp16=not torch.cuda.is_bf16_supported(),  # bf16 不行就上 fp16
    warmup_ratio=0.05,  # 预热比例
    logging_steps=50,  # 打印步骤间隔
    report_to='wandb',  # 日志输出目标，不想用 wandb 可以设置为 None
    num_train_epochs=1,  # 训练轮数，2 ~ 3 即可
    save_steps=1000,                            # 检查点保存步骤间隔
    seed=3407,  # 随机种子
    deepspeed=deepspeed_config,  # 应用DeepSpeed配置
    gradient_checkpointing=True, # 启动梯度检查点
)
# 自定义学习率调度器
def create_custom_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-4):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # 预热阶段：学习率从 0 增长到初始学习率
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # 余弦衰减阶段：结合最低学习率限制
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(cosine_decay, min_lr / training_args.learning_rate)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

# 创建学习率调度器
num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)
scheduler = create_custom_scheduler(optimizer, num_warmup_steps, num_training_steps)

trainer = Trainer(
    model=model,  # 模型实例
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练集
    tokenizer=tokenizer,  # 分词器
    optimizers=(optimizer, scheduler),  # 同时传递优化器和调度器
)
#trainer.train()
trainer.train(resume_from_checkpoint=True)