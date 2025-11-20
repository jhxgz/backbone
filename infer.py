import pandas as pd
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import h5py
from PIL import ImageFile
import torch
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers import CLIPTextModel, CLIPTokenizer


from src.utils import load_data_for_inference, prep_strings, postprocess_preds
ImageFile.LOAD_TRUNCATED_IMAGES = True


PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

def evaluate_norag_model(args, feature_extractor, tokenizer, model, eval_df):
    """Models without retrival augmentation can be evaluated with a batch of length >1."""
    out = []
    bs = args.batch_size

    for idx in tqdm(range(0, len(eval_df), bs)):
        file_names = eval_df['file_name'][idx:idx+bs]
        image_ids = eval_df['image_id'][idx:idx+bs]
        decoder_input_ids = [prep_strings('', tokenizer, is_test=True) for _ in range(len(image_ids))] 
                
        # load image 
        images = [Image.open(args.images_dir + file_name).convert("RGB") for file_name in file_names]
        pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
        with torch.no_grad():
            preds = model.generate(pixel_values.to(args.device), 
                               decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                               **args.generation_kwargs)
        preds = tokenizer.batch_decode(preds)
 
        for image_id, pred in zip(image_ids, preds):
            pred = postprocess_preds(pred, tokenizer)
            out.append({"image_id": int(image_id), "caption": pred})

    return out

def evaluate_rag_model(args, feature_extractor, tokenizer, model, eval_df):
    """RAG models can only be evaluated with a batch of length 1."""

    template = open(args.template_path).read().strip() + ' '

    if args.features_path is not None:
        features = h5py.File(args.features_path, 'r')

    out = []
    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        image_id = eval_df['image_id'][idx]
        caps = eval_df['caps'][idx]  # list[str]

        # --- 1) encode retrieved captions into CLIP text pooled embeddings ---
        # if no caps available, use None
        if caps is None or len(caps) == 0:
            retrieved_tensor = None
            retrieved_mask = None
        else:
            # use model.text_tokenizer and model.text_encoder if attached (load_model already attaches them)
            toks = model.text_tokenizer(caps, padding=True, truncation=True, return_tensors="pt")
            toks = {k: v.to(args.device) for k, v in toks.items()}
            with torch.no_grad():
                out_enc = model.text_encoder(**toks)
                pooled = out_enc.last_hidden_state[:, 0, :].detach()  # [k, D_text]
            # reshape to [1, k, D] (batch size 1)
            retrieved_tensor = pooled.unsqueeze(0)
            retrieved_mask = torch.ones(1, pooled.size(0), dtype=torch.long, device=args.device)

        # build decoder_input ids as before
        decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                         k=int(args.k), is_test=True)

        # now pass retrieved_tensor (not raw strings) into generate
        if args.features_path is not None:
            encoder_last_hidden_state = torch.FloatTensor([features[image_id][()]])
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))
            with torch.no_grad():
                pred = model.generate(encoder_outputs=encoder_outputs,
                                      decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                                      retrieved_captions=retrieved_tensor,
                                      retrieved_captions_mask=retrieved_mask,
                                      **args.generation_kwargs)

        else:
            image = Image.open(args.images_dir + file_name).convert("RGB")
            pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                pred = model.generate(pixel_values.to(args.device),
                                      decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                                      retrieved_captions=retrieved_tensor,
                                      retrieved_captions_mask=retrieved_mask,
                                      **args.generation_kwargs)
        # decode as before
        pred = tokenizer.decode(pred[0])
        pred = postprocess_preds(pred, tokenizer)
        out.append({"image_id": int(image_id), "caption": pred})

    return out

def load_model(args, checkpoint_path):
    import torch
    import os
    
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    
    # 从检查点的state_dict中推断fusion_dim和text_hidden
    state_dict_path = None
    for filename in ['pytorch_model.bin', 'model.safetensors']:
        if os.path.exists(checkpoint_path + '/' + filename):
            state_dict_path = checkpoint_path + '/' + filename
            break
    
    fusion_dim = None
    text_hidden = None
    
    if state_dict_path:
        if state_dict_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            except:
                state_dict = None
        else:
            state_dict = torch.load(state_dict_path, map_location='cpu')
        
        if state_dict:
            # 从prompt_proj的权重形状推断
            # nn.Linear(in_dim, out_dim) 权重形状是 [out_dim, in_dim]
            # prompt_proj 是 PromptProjector(in_dim=text_hidden, out_dim=fusion_dim)
            if 'prompt_proj.proj.weight' in state_dict:
                weight_shape = state_dict['prompt_proj.proj.weight'].shape
                fusion_dim = weight_shape[0]  # out_dim = fusion_dim
                text_hidden = weight_shape[1]  # in_dim = text_hidden
                print(f"从检查点推断: fusion_dim={fusion_dim}, text_hidden={text_hidden} (prompt_proj.proj.weight 形状: {weight_shape})")
            elif 'visual_proj.proj.weight' in state_dict:
                weight_shape = state_dict['visual_proj.proj.weight'].shape
                fusion_dim = weight_shape[0]  # out_dim = fusion_dim
                print(f"从检查点推断: fusion_dim={fusion_dim} (visual_proj.proj.weight 形状: {weight_shape})")
    
    # 设置config中的fusion_dim
    if fusion_dim is not None:
        config.fusion_dim = fusion_dim
    
    # 先加载CLIP text encoder（用于获取正确的text_hidden）
    clip_text_tokenizer = CLIPTokenizer.from_pretrained(args.encoder_name)
    clip_text_encoder = CLIPTextModel.from_pretrained(args.encoder_name)
    for p in clip_text_encoder.parameters():
        p.requires_grad = False
    
    # 如果从检查点推断的text_hidden与CLIP text encoder的hidden_size不同，需要特殊处理
    actual_text_hidden = clip_text_encoder.config.hidden_size
    if text_hidden is not None and text_hidden != actual_text_hidden:
        print(f"警告: 检查点中的text_hidden={text_hidden}，但CLIP text encoder的hidden_size={actual_text_hidden}")
        print("将在加载后手动修复prompt_proj...")
    
    # 加载模型（允许形状不匹配，稍后手动修复）
    model = AutoModel.from_pretrained(
        checkpoint_path,
        config=config,
        ignore_mismatched_sizes=True  # 允许形状不匹配
    )
    model.config = config

    # attach text encoder
    model.text_tokenizer = clip_text_tokenizer
    model.text_encoder = clip_text_encoder
    
    # 如果text_hidden不匹配，需要重新创建prompt_proj并加载权重
    if text_hidden is not None and text_hidden != actual_text_hidden and state_dict:
        print("重新创建prompt_proj以匹配检查点...")
        from src.modules.multimodal_projection import PromptProjector
        # 重新创建prompt_proj
        model.prompt_proj = PromptProjector(in_dim=text_hidden, out_dim=fusion_dim)
        # 从state_dict加载权重
        if 'prompt_proj.proj.weight' in state_dict:
            model.prompt_proj.proj.weight.data = state_dict['prompt_proj.proj.weight'].clone()
            if 'prompt_proj.proj.bias' in state_dict:
                model.prompt_proj.proj.bias.data = state_dict['prompt_proj.proj.bias'].clone()
            if 'prompt_proj.ln.weight' in state_dict:
                model.prompt_proj.ln.weight.data = state_dict['prompt_proj.ln.weight'].clone()
            if 'prompt_proj.ln.bias' in state_dict:
                model.prompt_proj.ln.bias.data = state_dict['prompt_proj.ln.bias'].clone()
            print("prompt_proj权重已从检查点加载")
    
    # move both model and text encoder to device
    clip_text_encoder.to(args.device)
    model.to(args.device)
    model.eval()
    return model


def infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn):
    model = load_model(args, checkpoint_path)
    preds = infer_fn(args, feature_extractor, tokenizer, model, eval_df)
    with open(os.path.join(checkpoint_path, args.outfile_name), 'w') as outfile:
        json.dump(preds, outfile)

def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from src.vision_encoder_decoder import SmallCap, SmallCapConfig
    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src.opt import ThisOPTConfig, ThisOPTForCausalLM
    from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM

    AutoConfig.register("this_xglm", ThisXGLMConfig)
    AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
    AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    AutoConfig.register("this_opt", ThisOPTConfig)
    AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
    AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)
    
    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

def main(args):

    register_model_and_config()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.infer_test or args.disable_rag:
        args.features_path = None
    
    if args.features_path is not None:
        feature_extractor = None
    else:
        feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)

    if args.disable_rag:
        args.k=0
        infer_fn = evaluate_norag_model
    else:
        infer_fn = evaluate_rag_model

    if args.infer_test:
        split = 'test'
    else:
        split = 'val'

    data = load_data_for_inference(args.annotations_path, args.captions_path)

    eval_df = pd.DataFrame(data[split])
    args.outfile_name = '{}_preds.json'.format(split)

    # load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    
    # configure generation 
    args.generation_kwargs = {'max_new_tokens': CAPTION_LENGTH, 'no_repeat_ngram_size': 0, 'length_penalty': 0.,
                              'num_beams': 3, 'early_stopping': True, 'eos_token_id': tokenizer.eos_token_id}

    # run inference once if checkpoint specified else run for all checkpoints
    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)
        infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)
    else:
        for checkpoint_path in os.listdir(args.model_path):
            if 'runs' in checkpoint_path:
                continue
            checkpoint_path = os.path.join(args.model_path, checkpoint_path)
            if os.path.exists(os.path.join(checkpoint_path, args.outfile_name)):
                print('Found existing file for', checkpoint_path)
            else:
                infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--images_dir", type=str, default="data/images/train/", help="Directory where input image features are stored")
    parser.add_argument("--features_path", type=str, default='features/val.hdf5', help="H5 file with cached input image features")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
        
    parser.add_argument("--model_path", type=str, default=None, help="Path to model to use for inference")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")

    parser.add_argument("--infer_test", action="store_true", default=False, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_resnet50x64.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size; only matter if evaluating a norag model")

    args = parser.parse_args()

    main(args)
   
