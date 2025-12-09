import pandas as pd
import numpy as np
import os
import argparse
import datetime
import torch

os.environ["WANDB_DISABLED"] = "true"

from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import TrainerCallback  # [新增] 导入 TrainerCallback

from transformers import VisionEncoderDecoderModel, CLIPModel, CLIPVisionModel, EncoderDecoderModel
from src.vision_encoder_decoder import SmallCap, SmallCapConfig
from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM
from src.opt import ThisOPTConfig, ThisOPTForCausalLM

from src.utils import *

# for attention with 28M params, we devide the attention dimensions by 1
# for attention with 14M params, we devide the attention dimensions by 2, etc.
PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25


# [新增] 辅助损失调度器回调函数
class AuxLossScheduler(TrainerCallback):
    """
    在指定的 Epoch 后，降低或关闭 Aux Loss 的权重，让模型专注于生成指标 (CIDEr)。
    """

    def __init__(self, switch_epoch: int, new_lambda: float = 0.01):
        self.switch_epoch = switch_epoch
        self.new_lambda = new_lambda
        self.has_switched = False

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        current_epoch = int(state.epoch)

        # 检查是否达到切换 Epoch
        if current_epoch >= self.switch_epoch:
            # 处理多卡训练 (DataParallel / DistributedDataParallel)
            actual_model = model.module if hasattr(model, 'module') else model

            # 修改模型属性 (如果存在)
            if hasattr(actual_model, 'lambda_aux'):
                if actual_model.lambda_aux != self.new_lambda:
                    print(f"\n[AuxLossScheduler] Epoch {current_epoch}: "
                          f"Changing lambda_aux from {actual_model.lambda_aux} to {self.new_lambda} "
                          "to focus on CIDEr optimization.")
                    actual_model.lambda_aux = self.new_lambda

            # 同步修改 config 中的值 (为了严谨)
            if hasattr(actual_model, 'config') and hasattr(actual_model.config, 'lambda_aux'):
                actual_model.config.lambda_aux = self.new_lambda


def get_model_and_auxiliaries(args):
    # register model types
    if "xglm" in args.decoder_name:
        AutoConfig.register("this_xglm", ThisXGLMConfig)
        AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
        AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    elif "opt" in args.decoder_name:
        AutoConfig.register("this_opt", ThisOPTConfig)
        AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
        AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)

    else:
        AutoConfig.register("this_gpt2", ThisGPT2Config)
        AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)

    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

    # create and configure model
    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]

    encoder_path = "./clip-vit-base-patch32"
    decoder_path = "./gpt2"
    feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(decoder_path)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    # ---- load CLIP text encoder & tokenizer (prompt encoder) ----
    clip_text_tokenizer = CLIPTokenizer.from_pretrained(encoder_path)
    clip_text_encoder = CLIPTextModel.from_pretrained(encoder_path)

    # freeze text encoder by default (可选)
    if getattr(args, "freeze_text_encoder", True):
        for p in clip_text_encoder.parameters():
            p.requires_grad = False

    use_ar_adapter = not getattr(args, "disable_ar_adapter", False)
    use_dual_path_refiner = getattr(args, "use_dual_path_refiner", False)
    use_cbsa = getattr(args, "use_cbsa", False)

    # 如果启用CBSA，自动禁用AR模块（避免重复增强）
    if use_cbsa:
        use_ar_adapter = False
        print("CBSA enabled: AR adapter will be disabled to avoid redundant enhancement.")

    # [修改] 计算 VG-TAR 的 FFN 维度，强制设为 1x Hidden Size 以减少参数
    # 默认 Transformer FFN 是 4x，这里改为 1x (例如 768 -> 768)
    text_hidden_size = clip_text_encoder.config.hidden_size
    tar_ffn_dim = int(text_hidden_size * 4)
    print(f"Setting VG-TAR FFN dim to {tar_ffn_dim} (1x hidden size) for parameter efficiency.")

    # pass text encoder & tokenizer into SmallCap factory
    model = SmallCap.from_encoder_decoder_pretrained(
        encoder_path,
        decoder_path,
        use_dpg=args.use_dpg,
        use_visual_selector=args.use_visual_selector,
        # === AR 图像增强参数 ===
        use_ar_adapter=use_ar_adapter,
        ar_down_ratio=args.ar_down_ratio,
        ar_dropout=args.ar_dropout,
        ar_use_gate=not args.disable_ar_gate,
        # ===
        # === 双路径增强模块参数 ===
        use_dual_path_refiner=use_dual_path_refiner,
        dual_path_init_alpha=args.dual_path_init_alpha,
        apr_n_queries=args.apr_n_queries,
        apr_n_heads=args.apr_n_heads,
        apr_proj_back=args.apr_proj_back,  # 保持 True (默认)
        apr_dropout=args.apr_dropout,
        # ===
        # === CBSA视觉增强模块参数 ===
        use_cbsa=args.use_cbsa,
        # ===
        # === TAR模块参数 ===
        use_tar_module=not args.disable_tar_module,
        tar_down_ratio=args.tar_down_ratio,
        tar_dropout=args.tar_dropout,
        tar_use_gate=not args.disable_tar_gate,
        tar_init_gate=args.tar_init_gate,
        tar_ffn_dim=tar_ffn_dim,  # [新增] 传入压缩后的 FFN 维度
        # ===
        cross_attention_reduce_factor=cross_attention_reduce_factor,
        text_encoder=clip_text_encoder,
        text_tokenizer=clip_text_tokenizer
    )

    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if not args.disable_rag:
        model.config.k = args.k
        model.config.retrieval_encoder = args.retrieval_encoder
    model.config.max_length = CAPTION_LENGTH
    model.config.rag = not args.disable_rag

    # === FiLM 融合模块配置 ===
    model.config.film_per_token = args.film_per_token
    # film_use_residual: 如果指定了--disable_film_residual则为False，否则为True（默认）
    model.config.film_use_residual = not args.disable_film_residual
    model.config.film_init_alpha = args.film_init_alpha
    model.config.film_text_pool = args.film_text_pool
    model.config.film_dropout = args.film_dropout
    if args.film_hidden is not None:
        model.config.film_hidden = args.film_hidden
    # === FiLM 融合模块配置结束 ===

    # === TAR模块配置 ===
    model.config.use_tar_module = not args.disable_tar_module
    model.config.tar_down_ratio = args.tar_down_ratio
    model.config.tar_dropout = args.tar_dropout
    model.config.tar_use_gate = not args.disable_tar_gate
    model.config.tar_init_gate = args.tar_init_gate
    model.config.tar_ffn_dim = tar_ffn_dim  # [新增] 保存到 config
    # === TAR模块配置结束 ===

    # === 辅助对比损失配置 ===
    model.config.use_aux_loss = not getattr(args, "disable_aux_loss", False)
    model.config.lambda_aux = float(getattr(args, "lambda_aux", 0.1))
    # === 辅助对比损失配置结束 ===

    # print("model",model)
    # print(stop)
    # freeze parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    if "xglm" in args.decoder_name or "opt" in args.decoder_name:
        if not args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'encoder_attn' not in name:
                    param.requires_grad = False

    else:
        if not args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'crossattention' not in name:
                    param.requires_grad = False

    # count trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Training a model with {} trainable parameters.'.format(num_trainable_params))
    ar_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and 'image_ar_adapter' in n
    )
    print(f'AR adapter trainable params: {ar_params}')

    # 统计双路径增强模块的可训练参数
    dual_path_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and 'dual_path_refiner' in n
    )
    if dual_path_params > 0:
        print(f'DualPathRefiner trainable params: {dual_path_params}')

    # 统计 FiLM 融合模块的可训练参数
    film_fusion_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and 'film_fusion' in n
    )
    if film_fusion_params > 0:
        print(f'FiLM fusion trainable params: {film_fusion_params}')

    # 统计 TAR 模块的可训练参数
    tar_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and 'tar_module' in n
    )
    if tar_params > 0:
        print(f'TAR module trainable params: {tar_params}')

    # 统计辅助对比损失模块的可训练参数
    aux_loss_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and 'aux_projector' in n
    )
    if aux_loss_params > 0:
        print(f'Auxiliary Contrastive Loss trainable params: {aux_loss_params}')
        print(f'Auxiliary Contrastive Loss weight (lambda_aux): {model.config.lambda_aux}')

    return model, tokenizer, feature_extractor, clip_text_tokenizer, clip_text_encoder


def get_data(tokenizer, max_length, args):
    data = load_data_for_training(args.annotations_path, args.captions_path)
    train_df = pd.DataFrame(data['train'])

    if args.ablation_visual:
        train_dataset = AblationFeaturesDataset(
            df=train_df,
            features_path=os.path.join(args.features_dir, 'train.hdf5'),
            tokenizer=tokenizer,
            rag=not args.disable_rag,
            template_path=args.template_path,
            k=args.k,
            max_caption_length=max_length)
    else:
        train_dataset = TrainDataset(
            df=train_df,
            features_path=os.path.join(args.features_dir, 'train.hdf5'),
            tokenizer=tokenizer,
            rag=not args.disable_rag,
            template_path=args.template_path,
            k=args.k,
            max_caption_length=max_length)

    return train_dataset


def main(args):
    model, tokenizer, feature_extractor, clip_text_tokenizer, clip_text_encoder = get_model_and_auxiliaries(args)
    train_dataset = get_data(tokenizer, model.config.max_length, args)

    # 如果用户指定了自定义输出目录名称，直接使用
    if args.output_dir_name:
        output_dir = os.path.join(args.experiments_dir, args.output_dir_name)
    else:
        # 自动生成输出目录名称
        model_type = 'norag' if args.disable_rag else 'rag'
        if args.ablation_visual:
            output_dir = '{}_{}M_{}_ablation'.format(model_type, args.attention_size, args.decoder_name)
        else:
            output_dir = '{}_{}M_{}'.format(model_type, args.attention_size, args.decoder_name)

        output_dir = os.path.join(args.experiments_dir, output_dir)
        use_ar_adapter = not args.disable_ar_adapter
        use_dual_path_refiner = args.use_dual_path_refiner
        use_tar_module = not args.disable_tar_module
        use_cbsa = getattr(args, "use_cbsa", False)

        # 如果启用CBSA，自动禁用AR模块（避免重复增强）
        if use_cbsa:
            use_ar_adapter = False

        # 根据使用的增强模块添加路径标识
        if use_dual_path_refiner:
            output_dir = output_dir + "_dualpath"
        elif use_cbsa:
            output_dir = output_dir + "_cbsa"
        elif use_ar_adapter:
            output_dir = output_dir + "_ar"

        if args.exp_suffix:
            suffix_clean = args.exp_suffix.strip()
            if suffix_clean:
                output_dir = f"{output_dir}_{suffix_clean}"

    def collate_fn(batch):
        # batch: list of samples (dict)
        # extract retrieved_captions (may be list[str] per sample)
        retrieved = [example.get("retrieved_captions", None) for example in batch]

        # 提取原始文本（用于计算 GT text embeds）
        gt_texts = [example.get("text", None) for example in batch]

        collated = default_data_collator(batch)

        # 计算 GT text embeds（用于辅助对比损失）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if gt_texts is not None and all(t is not None for t in gt_texts):
            # 使用 CLIP text encoder 编码 GT 文本
            with torch.no_grad():
                gt_toks = clip_text_tokenizer(gt_texts, padding=True, truncation=True, return_tensors="pt")
                gt_toks = {k: v.to(device) for k, v in gt_toks.items()}
                gt_text_outputs = clip_text_encoder(**gt_toks)
                # 使用 CLS token 的 hidden state 作为全局特征 [Batch, Clip_Dim]
                gt_text_embeds = gt_text_outputs.last_hidden_state[:, 0, :].detach()  # [B, D_text]
            collated["gt_text_embeds"] = gt_text_embeds
        else:
            collated["gt_text_embeds"] = None

        # If retrived is None or rag disabled, keep None
        if retrieved is None or all(x is None for x in retrieved):
            collated["retrieved_captions"] = None
            return collated

        # Encode retrieved captions into CLIP text pooled embeddings (CLS token)
        # retrieved is list where each item is e.g. ["cap1", "cap2", ...] or None
        # convert None to empty list
        retrieved = [r if r is not None else [] for r in retrieved]

        # flatten for tokenization efficiency
        flat_caps = []
        idx_map = []  # for reconstructing per-sample groups
        for caps in retrieved:
            idx_map.append(len(flat_caps))
            flat_caps.extend(caps)  # order preserved

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if len(flat_caps) == 0:
            # nothing to encode
            collated["retrieved_captions"] = None
            return collated

        # tokenize and encode in batches
        toks = clip_text_tokenizer(flat_caps, padding=True, truncation=True, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = clip_text_encoder(**toks)
            # pooled: use CLS token hidden state
            pooled = out.last_hidden_state[:, 0, :].detach()  # [sum_k, D_text]

        # rebuild per-sample lists and pad to max_k
        D = pooled.size(-1)
        batch_caps_emb = []
        masks = []
        cur = 0
        maxk = 0
        for caps in retrieved:
            k = len(caps)
            maxk = max(maxk, k)
        # if maxk == 0, no captions at all (shouldn't happen due to earlier check)
        cur = 0
        for caps in retrieved:
            k = len(caps)
            if k > 0:
                seg = pooled[cur:cur + k]  # [k, D]
                cur += k
            else:
                seg = torch.zeros(0, D, device=device, dtype=pooled.dtype)
            # pad to maxk
            if seg.size(0) < maxk:
                pad = torch.zeros(maxk - seg.size(0), D, device=device, dtype=seg.dtype)
                seg = torch.cat([seg, pad], dim=0)
                mask = torch.cat([torch.ones(k, device=device), torch.zeros(maxk - k, device=device)])
            else:
                mask = torch.ones(k, device=device)
            batch_caps_emb.append(seg)  # [maxk, D]
            masks.append(mask)  # [maxk]
        # stack into tensors
        retrieved_tensor = torch.stack(batch_caps_emb, dim=0)  # [B, maxk, D]
        retrieved_mask = torch.stack(masks, dim=0).long()  # [B, maxk]

        collated["retrieved_captions"] = retrieved_tensor
        collated["retrieved_captions_mask"] = retrieved_mask
        return collated

    # 计算总训练步数和 warmup 步数
    if args.warmup_steps is None:
        steps_per_epoch = len(train_dataset) // (args.batch_size * args.gradient_steps)
        total_steps_estimate = steps_per_epoch * args.n_epochs
        warmup_steps = min(1000, int(total_steps_estimate * 0.1))
        print(f"自动设置 warmup_steps = {warmup_steps} (估算总步数: {total_steps_estimate})")
    else:
        warmup_steps = args.warmup_steps
        print(f"使用指定的 warmup_steps = {warmup_steps}")

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate=args.lr,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=args.n_epochs,
        logging_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=True,
        # 添加梯度裁剪
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        # 添加 Warmup 和 Cosine Annealing 调度器
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
    )

    # 创建自定义优化器，为门控参数设置独立的学习率
    def create_optimizer_with_gate_lr(model, training_args, gate_lr_ratio):
        """创建优化器，为 VG-TAR 模块的门控参数设置更低的学习率"""
        gate_params = []
        other_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'tar_module' in name and 'gate' in name:
                    gate_params.append(param)
                else:
                    other_params.append(param)

        base_lr = training_args.learning_rate
        gate_lr = base_lr * gate_lr_ratio

        param_groups = [
            {"params": other_params, "lr": base_lr},
        ]

        if len(gate_params) > 0:
            param_groups.append({
                "params": gate_params,
                "lr": gate_lr,
            })
            print(f"✓ 为 {len(gate_params)} 个门控参数设置独立学习率: {gate_lr:.6f}")

        from torch.optim import AdamW
        optimizer = AdamW(
            param_groups,
            lr=base_lr,
            betas=(training_args.adam_beta1 if hasattr(training_args, 'adam_beta1') else 0.9,
                   training_args.adam_beta2 if hasattr(training_args, 'adam_beta2') else 0.999),
            eps=training_args.adam_epsilon if hasattr(training_args, 'adam_epsilon') else 1e-8,
            weight_decay=training_args.weight_decay if hasattr(training_args, 'weight_decay') else 0.0,
        )
        return optimizer

    print("\n=== 优化器配置 ===")
    optimizer = create_optimizer_with_gate_lr(model, training_args, args.gate_lr_ratio)

    # [新增] 初始化 AuxLossScheduler 回调
    callbacks = []
    if not args.disable_aux_loss:
        # 默认策略：在最后 20% 的 Epoch 降低 Aux Loss，或者使用用户指定的 Epoch
        decay_epoch = args.aux_loss_decay_epoch
        if decay_epoch is None:
            decay_epoch = max(1, args.n_epochs - 2)  # 默认最后 2 个 Epoch

        print(f"\n[Strategy] Aux Loss 将在第 {decay_epoch} 个 Epoch 降低权重，专注于 CIDEr 优化。")
        callbacks.append(AuxLossScheduler(switch_epoch=decay_epoch, new_lambda=0.01))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        tokenizer=feature_extractor,
        optimizers=(optimizer, None),
        callbacks=callbacks,  # [新增] 添加回调
    )

    trainer.train()


def get_args_parser():
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--features_dir", type=str, default="features/",
                        help="Directory where cached input image features are stored")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json",
                        help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--experiments_dir", type=str, default="experiments/",
                        help="Directory where trained models will be saved")

    parser.add_argument("--encoder_name", type=str, default="clip-vit-base-patch32",
                        help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2",
                        help="Decoder name as found of HuggingFace or stored locally")
    parser.add_argument("--attention_size", type=float, default=7,
                        help="Number of parameters in the cross attention {28, 14, 7, 3.5, 1.75}")
    parser.add_argument("--train_decoder", action="store_true", default=False,
                        help="Whether to train the decoder in addition to the attention")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64",
                        help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_resnet50x64.json",
                        help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping (default: 1.0, set to 0 to disable)")
    parser.add_argument("--exp_suffix", type=str, default="",
                        help="Optional suffix appended to the auto-generated experiment directory")

    parser.add_argument("--ablation_visual", action="store_true", default=False,
                        help="Whether to blank visual features")
    parser.add_argument("--disable_ar_adapter", action="store_true", default=True,
                        help="Disable AR adapter enhancement on visual tokens")
    parser.add_argument("--ar_down_ratio", type=int, default=2,
                        help="Bottleneck ratio for AR adapter hidden size")
    parser.add_argument("--ar_dropout", type=float, default=0.1,
                        help="Dropout rate inside AR adapter")
    parser.add_argument("--disable_ar_gate", action="store_true", default=False,
                        help="Disable learnable gate inside AR adapter")
    parser.add_argument("--use_cbsa", action="store_true", default=False,
                        help="Enable CBSA (Contract-and-Broadcast Self-Attention) visual refiner")
    parser.add_argument("--use_mamba", action="store_true", default=False,
                        help="Enable Mamba Adapter for visual refinement.")

    # === 双路径增强模块 (DualPathRefiner) 参数 ===
    parser.add_argument("--use_dual_path_refiner", action="store_true", default=False,
                        help="Enable dual path refiner (LayerNorm + AdapterResidual + AttentionPoolingRefiner)")
    parser.add_argument("--dual_path_init_alpha", type=float, default=0.1,
                        help="Initial value for alpha1 and alpha2 (dual path weights)")
    parser.add_argument("--apr_n_queries", type=int, default=1,
                        help="Number of queries for AttentionPoolingRefiner")
    parser.add_argument("--apr_n_heads", type=int, default=8,
                        help="Number of attention heads for AttentionPoolingRefiner")
    parser.add_argument("--apr_proj_back", action="store_true", default=False,
                        help="Whether to project back to token space in AttentionPoolingRefiner")
    parser.add_argument("--apr_dropout", type=float, default=0.1,
                        help="Dropout rate inside AttentionPoolingRefiner")
    # === 双路径增强模块参数结束 ===

    # === FiLM 融合模块参数 ===
    parser.add_argument("--film_per_token", action="store_true", default=False,
                        help="Enable per-token FiLM modulation (default: False, global channel-wise)")
    parser.add_argument("--disable_film_residual", action="store_true", default=False,
                        help="Disable residual connection in FiLM (default: residual enabled)")
    parser.add_argument("--film_init_alpha", type=float, default=0.1,
                        help="Initial value for residual scaling factor in FiLM (default: 0.1)")
    parser.add_argument("--film_text_pool", type=str, default="mean", choices=["mean", "cls"],
                        help="Text feature pooling method for FiLM: 'mean' or 'cls' (default: mean)")
    parser.add_argument("--film_dropout", type=float, default=0.0,
                        help="Dropout rate in FiLM MLP (default: 0.0)")
    parser.add_argument("--film_hidden", type=int, default=None,
                        help="Hidden dimension for FiLM MLP (default: max(256, fusion_dim))")
    # === FiLM 融合模块参数结束 ===

    # === TAR模块参数 ===
    parser.add_argument("--disable_tar_module", action="store_true", default=False,
                        help="Disable TAR module enhancement on text encoder outputs")
    parser.add_argument("--tar_down_ratio", type=int, default=8,
                        help="Bottleneck ratio for TAR module hidden size (default: 8)")
    parser.add_argument("--tar_dropout", type=float, default=0.1,
                        help="Dropout rate inside TAR module (default: 0.1)")
    parser.add_argument("--disable_tar_gate", action="store_true", default=False,
                        help="Disable learnable gate inside TAR module")
    parser.add_argument("--tar_init_gate", type=float, default=-4.0,
                        help="Initial value for gate parameter in TAR module (default: -4.0, ensures Sigmoid(gate) near 0 at initialization)")
    # === TAR模块参数结束 ===

    # === 训练策略参数 ===
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of warmup steps for learning rate scheduler (default: 1000 or 10%% of total steps, whichever is smaller)")
    parser.add_argument("--gate_lr_ratio", type=float, default=0.5,
                        help="Learning rate ratio for gate parameters relative to baseline LR (default: 0.5, i.e., gate LR = 0.5 * base LR)")
    # === 训练策略参数结束 ===

    # === 辅助对比损失参数 ===
    parser.add_argument("--disable_aux_loss", action="store_true", default=False,
                        help="Disable auxiliary contrastive loss (default: enabled)")
    parser.add_argument("--lambda_aux", type=float, default=0.1,
                        help="Weight for auxiliary contrastive loss (default: 0.1)")
    # [新增] 参数控制 Aux Loss 衰减时机
    parser.add_argument("--aux_loss_decay_epoch", type=int, default=None,
                        help="Epoch to reduce/disable auxiliary loss (default: last 2 epochs)")
    # === 辅助对比损失参数结束 ===

    # === 输出路径配置 ===
    parser.add_argument("--output_dir_name", type=str, default=None,
                        help="Custom output directory name (if not specified, will auto-generate based on model config)")
    # === 输出路径配置结束 ===

    parser.add_argument("--use_dpg", action="store_true", default=False,
                        help="Use Detail-Oriented Prompt Generation (DPG) instead of VG-TAR.")
    # MoE 参数
    parser.add_argument("--use_moe", action="store_true", default=False,
                        help="Enable Mixture-of-Experts (MoE) Adapter.")
    parser.add_argument("--moe_num_experts", type=int, default=4,
                        help="Number of experts in MoE.")
    parser.add_argument("--moe_topk", type=int, default=2,
                        help="Top-K experts active per token.")

    parser.add_argument("--use_visual_selector", action="store_true", default=False,
                        help="Enable Text-Guided Visual Selector for visual denoising.")

    return parser

if __name__ == '__main__':
    # 调用上面的函数获取 parser
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)