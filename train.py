import pandas as pd
import numpy as np
import os
import argparse
import datetime

os.environ["WANDB_DISABLED"] = "true"

from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments
from transformers import CLIPTextModel, CLIPTokenizer

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
    # pass text encoder & tokenizer into SmallCap factory
    model = SmallCap.from_encoder_decoder_pretrained(
        encoder_path,
        decoder_path,
        # === AR 图像增强参数 ===
        use_ar_adapter=use_ar_adapter,
        ar_down_ratio=args.ar_down_ratio,
        ar_dropout=args.ar_dropout,
        ar_use_gate=not args.disable_ar_gate,
        # ===
        # === 双路径增强模块参数 ===
        use_dual_path_refiner=use_dual_path_refiner,
        dual_path_init_alpha=getattr(args, "dual_path_init_alpha", 0.1),
        apr_n_queries=getattr(args, "apr_n_queries", 1),
        apr_n_heads=getattr(args, "apr_n_heads", 8),
        apr_proj_back=getattr(args, "apr_proj_back", True),
        apr_dropout=getattr(args, "apr_dropout", 0.1),
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
    
    # 统计 AdaptedFFM 模块的可训练参数
    adapted_ffm_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and 'adapted_ffm' in n
    )
    if adapted_ffm_params > 0:
        print(f'AdaptedFFM trainable params: {adapted_ffm_params}')

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

    model_type = 'norag' if args.disable_rag else 'rag'
    if args.ablation_visual:
        output_dir = '{}_{}M_{}_ablation'.format(model_type, args.attention_size, args.decoder_name)
    else:
        output_dir = '{}_{}M_{}'.format(model_type, args.attention_size, args.decoder_name)

    output_dir = os.path.join(args.experiments_dir, output_dir)
    use_ar_adapter = not args.disable_ar_adapter
    use_dual_path_refiner = getattr(args, "use_dual_path_refiner", False)
    
    # 根据使用的增强模块添加路径标识
    if use_dual_path_refiner:
        output_dir = output_dir + "_dualpath"
    elif use_ar_adapter:
        output_dir = output_dir + "_ar"

    def collate_fn(batch):
        # batch: list of samples (dict)
        # extract retrieved_captions (may be list[str] per sample)
        retrieved = [example.get("retrieved_captions", None) for example in batch]
        collated = default_data_collator(batch)

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
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        tokenizer=feature_extractor,
    )

    trainer.train()


if __name__ == '__main__':
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

    parser.add_argument("--ablation_visual", action="store_true", default=False,
                        help="Whether to blank visual features")
    parser.add_argument("--disable_ar_adapter", action="store_true", default=False,
                        help="Disable AR adapter enhancement on visual tokens")
    parser.add_argument("--ar_down_ratio", type=int, default=4,
                        help="Bottleneck ratio for AR adapter hidden size")
    parser.add_argument("--ar_dropout", type=float, default=0.1,
                        help="Dropout rate inside AR adapter")
    parser.add_argument("--disable_ar_gate", action="store_true", default=False,
                        help="Disable learnable gate inside AR adapter")

    # === 双路径增强模块 (DualPathRefiner) 参数 ===
    parser.add_argument("--use_dual_path_refiner", action="store_true", default=False,
                        help="Enable dual path refiner (LayerNorm + AdapterResidual + AttentionPoolingRefiner)")
    parser.add_argument("--dual_path_init_alpha", type=float, default=0.1,
                        help="Initial value for alpha1 and alpha2 (dual path weights)")
    parser.add_argument("--apr_n_queries", type=int, default=1,
                        help="Number of queries for AttentionPoolingRefiner")
    parser.add_argument("--apr_n_heads", type=int, default=8,
                        help="Number of attention heads for AttentionPoolingRefiner")
    parser.add_argument("--apr_proj_back", action="store_true", default=True,
                        help="Whether to project back to token space in AttentionPoolingRefiner")
    parser.add_argument("--apr_dropout", type=float, default=0.1,
                        help="Dropout rate inside AttentionPoolingRefiner")
    # === 双路径增强模块参数结束 ===


    args = parser.parse_args()

    main(args)
