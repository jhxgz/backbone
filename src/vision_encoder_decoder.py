# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
""" Classes to support Vision-Encoder-Text-Decoder architectures"""
import timeit

from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
# from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.utils import logging
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import VisionEncoderDecoderConfig
import inspect

from .gpt2 import ThisGPT2LMHeadModel
from .gpt2 import ThisGPT2Config
from .xglm import ThisXGLMForCausalLM
from .xglm import ThisXGLMConfig
from .opt import ThisOPTForCausalLM
from .opt import ThisOPTConfig
from .modules.AR import AdapterResidual
from .modules.multimodal_projection import CBSAVisionProjector, PromptProjector
from .modules.adapted_ffm import AdaptedFFM


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "SmallCapConfig"

VISION_ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model
    as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via
    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like image captioning.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained
    Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical
    character recognition (OCR) yields a significant performance improvement.

    After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any
    other models (see the examples for more information).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VISION_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using a feature extractor (e.g. if you use ViT as the encoder,
            you should use [`ViTFeatureExtractor`]). See [`ViTFeatureExtractor.__call__`] for details.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the
            right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):
            This tuple must consist of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) is a tensor
            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0,
            ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.Seq2SeqLMOutput`] instead of a plain tuple.
        kwargs: (*optional*) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:

            - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
            - With a *decoder_* prefix which will be input as `**decoder_kwargs` for the decoder forward function.
"""


class SmallCapConfig(VisionEncoderDecoderConfig):
    model_type = "smallcap"

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)


class SmallCap(PreTrainedModel):
    r"""
    [`VisionEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    one of the base vision model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = SmallCapConfig
    base_model_prefix = "smallcap"
    main_input_name = "pixel_values"

    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
            text_encoder: Optional[PreTrainedModel] = None,
            text_tokenizer: Optional[object] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = SmallCapConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal#"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder.vision_model
        self.encoder.main_input_name = 'pixel_values'
        self.decoder = decoder

        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer

        # 创建visual_proj / prompt_proj / adapted_ffm
        enc_hidden = (
            self.encoder.config.hidden_size
            if hasattr(self.encoder.config, "hidden_size")
            else self.encoder.config.vision_config.hidden_size
        )
        fusion_dim = getattr(self.config, "fusion_dim", enc_hidden)

        # 图像投影：CLIP vision encoder输出 -> fusion_dim
        self.visual_proj = CBSAVisionProjector(in_dim=enc_hidden, out_dim=fusion_dim)

        # prompt投影：需要根据text_encoder的hidden_size决定in_dim
        text_hidden = None
        if self.text_encoder is not None:
            try:
                text_hidden = self.text_encoder.config.hidden_size
            except:
                text_hidden = fusion_dim

        if text_hidden is None:
            text_hidden = fusion_dim

        self.prompt_proj = PromptProjector(in_dim=text_hidden, out_dim=fusion_dim)

        # ===> BEGIN: 添加AR模块用于图像特征增强 <===
        self.use_ar_adapter = getattr(self.config, "use_ar_adapter", True)
        if self.use_ar_adapter:
            self.image_ar_adapter = AdapterResidual(
                dim=fusion_dim,
                down_ratio=int(getattr(self.config, "ar_down_ratio", 4)),
                dropout=float(getattr(self.config, "ar_dropout", 0.1)),
                use_gate=bool(getattr(self.config, "ar_use_gate", True)),
            )
        # ===> END: AR 图像增强模块 <===

        # 融合模块
        self.adapted_ffm = AdaptedFFM(dim=fusion_dim, hidden=max(256, fusion_dim),
                                      num_heads=getattr(self.config, "ffm_heads", 4), )

        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for VisionEncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
            cls,
            encoder_pretrained_model_name_or_path: str = None,
            decoder_pretrained_model_name_or_path: str = None,
            cross_attention_reduce_factor: int = None,
            *model_args,
            **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the image encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. An
                      example is `google/vit-base-patch16-224-in21k`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the text decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import VisionEncoderDecoderModel

        >>> # initialize a vit-bert from a pretrained ViT and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
        >>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "google/vit-base-patch16-224-in21k", "bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./vit-bert")
        >>> # load fine-tuned model
        >>> model = VisionEncoderDecoderModel.from_pretrained("./vit-bert")
        ```"""

        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                if "xglm" in decoder_pretrained_model_name_or_path:
                    decoder_config, kwargs_decoder = ThisXGLMConfig.from_pretrained(
                        decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                    )

                elif "opt" in decoder_pretrained_model_name_or_path:
                    decoder_config, kwargs_decoder = ThisOPTConfig.from_pretrained(
                        decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                    )

                else:
                    decoder_config, kwargs_decoder = ThisGPT2Config.from_pretrained(
                        decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                    )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True
                decoder_config.encoder_hidden_size = encoder.config.vision_config.hidden_size
                decoder_config.cross_attention_reduce_factor = cross_attention_reduce_factor
                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            # decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
            if "xglm" in decoder_pretrained_model_name_or_path:
                decoder = ThisXGLMForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

            elif "opt" in decoder_pretrained_model_name_or_path:
                decoder = ThisOPTForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
            else:
                decoder = ThisGPT2LMHeadModel.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # pull optional text encoder/tokenizer from kwargs before building config
        if isinstance(kwargs, dict):
            text_encoder = kwargs.pop("text_encoder", None)
            text_tokenizer = kwargs.pop("text_tokenizer", None)
        else:
            text_encoder = None
            text_tokenizer = None

        # instantiate config with corresponding kwargs
        config = SmallCapConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False

        return cls(encoder=encoder, decoder=decoder, config=config, text_encoder=text_encoder,
                   text_tokenizer=text_tokenizer)

    def forward(
            self,
            pixel_values=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        >>> import requests
        >>> from PIL import Image
        >>> import torch

        >>> processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        >>> # load image from the IAM dataset
        >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> # training
        >>> model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        >>> model.config.pad_token_id = processor.tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> pixel_values = processor(image, return_tensors="pt").pixel_values
        >>> text = "hello world"
        >>> labels = processor.tokenizer(text, return_tensors="pt").input_ids
        >>> outputs = model(pixel_values=pixel_values, labels=labels)
        >>> loss = outputs.loss

        >>> # inference (generation)
        >>> generated_ids = model.generate(pixel_values)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 从kwargs中提取retrieved_captions相关参数（这些不应该传递给encoder）
        retrieved_caps = kwargs.pop("retrieved_captions", None)
        retrieved_caps_mask = kwargs.pop("retrieved_captions_mask", None)

        kwargs_encoder = {
            argument: value 
            for argument, value in kwargs.items() 
            if not argument.startswith("decoder_")
            # 排除retrieved_captions相关参数，它们不应该传递给encoder
            and argument not in ["retrieved_captions", "retrieved_captions_mask"]
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)
        else:
            encoder_outputs = BaseModelOutput(encoder_outputs, None)

        encoder_hidden_states = encoder_outputs[0]

        encoder_attention_mask = None
        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # ===== Step 1: 使用CLIP vision encoder生成image embedding并投影 =====
        # 将 encoder_hidden_states (B, N, C_clip) 投影到 fusion_dim
        img_feats = self.visual_proj(encoder_hidden_states)  # [B, N, fusion_dim]

        # ===== Step 2: 使用AR模块增强图像特征 =====
        if getattr(self, "use_ar_adapter", False):
            img_feats = img_feats + self.image_ar_adapter(img_feats)

        # 从 kwargs 获取检索到的 captions 字段（Trainer 的 batch 需包含 'retrieved_captions'）
        # 这些参数已经从 kwargs 中移除，所以这里不再需要获取

        if retrieved_caps is not None and self.text_tokenizer is not None and self.text_encoder is not None:
            # ===== Step 3: 使用CLIP text encoder生成caption embedding =====
            # encode_prompts 会接受 tensor 或字符串列表
            # 返回 prompt_embeds: [B, L, D_text] 以及 attn_mask: [B, L]
            prompt_embeds, attn_mask = self.encode_prompts(retrieved_caps)

            # 如果外部传入了 retrieved_caps_mask （来自 dataset/collate），优先使用它（按需）
            if retrieved_caps_mask is not None:
                # ensure device and dtype
                retrieved_caps_mask = retrieved_caps_mask.to(
                    attn_mask.device) if attn_mask is not None else retrieved_caps_mask.to(
                    next(self.parameters()).device)
                # 如果 attn_mask 为 None（例如 encode_prompts 返回 pooled only），用 retrieved_caps_mask 作为 attn_mask
                if attn_mask is None:
                    attn_mask = retrieved_caps_mask
                else:
                    # 两者都存在时，使用按位相与（若需）
                    attn_mask = (attn_mask.long() & retrieved_caps_mask.long()).long()

            # ===== Step 4: 投影caption embedding到融合维度 =====
            # prompt_embeds: [B, L, D_text] or [B, D_text] (proj 会扩维)
            prompt_feats = self.prompt_proj(prompt_embeds)  # [B, L, fusion_dim]

            # ===== Step 5: 使用adapted_ffm对image embedding和caption embedding进行融合 =====
            # 调用 AdaptedFFM 进行跨模态融合，返回增强后的图像特征 [B, N, fusion_dim]
            img_feats = self.adapted_ffm(img_feats, prompt_feats, prompt_mask=attn_mask)

        # 最终把 img_feats 作为 encoder_hidden_states 传入 decoder
        encoder_hidden_states = img_feats

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        # 从kwargs中提取retrieved_captions相关参数，传递给forward
        if "retrieved_captions" in kwargs:
            input_dict["retrieved_captions"] = kwargs.pop("retrieved_captions")
        if "retrieved_captions_mask" in kwargs:
            input_dict["retrieved_captions_mask"] = kwargs.pop("retrieved_captions_mask")
        # 其他kwargs参数也传递
        input_dict.update(kwargs)
        return input_dict

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_tensor, model_kwargs, model_input_name=None
    ):
        """
        重写此方法，过滤掉不应传递给encoder的参数（如retrieved_captions）
        """
        # 从model_kwargs中提取retrieved_captions相关参数（这些不应该传递给encoder）
        retrieved_captions = model_kwargs.pop("retrieved_captions", None)
        retrieved_captions_mask = model_kwargs.pop("retrieved_captions_mask", None)
        
        # 调用父类方法（此时model_kwargs已经不包含retrieved_captions了）
        # 注意：如果父类没有这个方法，我们需要自己实现
        try:
            encoder_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(
                input_tensor, model_kwargs, model_input_name
            )
        except AttributeError:
            # 如果父类没有这个方法，我们自己实现
            # 这是从transformers库的generation_utils.py中简化版本
            if model_input_name is None:
                model_input_name = self.main_input_name
            
            encoder_kwargs = {}
            if model_input_name in model_kwargs:
                encoder_kwargs[model_input_name] = model_kwargs.pop(model_input_name)
            
            # 过滤掉不应传递给encoder的参数
            encoder_kwargs.update({
                k: v for k, v in model_kwargs.items()
                if k not in ["retrieved_captions", "retrieved_captions_mask", "decoder_input_ids", 
                            "decoder_attention_mask", "past_key_values", "use_cache"]
                and not k.startswith("decoder_")
            })
        
        # 将retrieved_captions参数添加回model_kwargs，以便传递给forward
        if retrieved_captions is not None:
            model_kwargs["retrieved_captions"] = retrieved_captions
        if retrieved_captions_mask is not None:
            model_kwargs["retrieved_captions_mask"] = retrieved_captions_mask
        
        # 确保encoder_kwargs不包含这些参数
        encoder_kwargs.pop("retrieved_captions", None)
        encoder_kwargs.pop("retrieved_captions_mask", None)
        
        return encoder_kwargs

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the VisionEncoderDecoderModel directly is not supported.Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

    def encode_prompts(self, prompt_input):
        """
            Accept:
              - None
              - list[str] (长度 batch_size) OR list[list[str]] (每个 sample 为多个 retrieved captions)
              - torch.Tensor of shape [B, K, D_text] (precomputed pooled embeddings) OR [B, D_text]
            Return:
              prompt_embeds: torch.Tensor [B, L, D_text]
              attn_mask: torch.Tensor [B, L] (1 for valid token, 0 for pad) or None if not applicable
        """
        if self.text_tokenizer is None or self.text_encoder is None:
            raise ValueError(
                "text_tokenizer/text_encoder not set in model. Pass them via from_encoder_decoder_pretrained kwargs."
            )

        device = next(self.parameters()).device

        # None case
        if prompt_input is None:
            return None, None

        # If already tensor: assume pooled or token-level embeddings provided
        if torch.is_tensor(prompt_input):
            x = prompt_input.to(device)
            # if 2D -> [B, D] treat as pooled single prompt per image
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [B, 1, D_text]
                attn_mask = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=device)
                return x, attn_mask
            # if 3D -> [B, K, D_text] already token-level/pool-level per caption
            if x.dim() == 3:
                attn_mask = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=device)
                return x, attn_mask
            raise ValueError("Unsupported tensor dim for prompt_input: got dim = %d" % x.dim())

        # Else expect list[str] or list[list[str]]
        prompt_texts = prompt_input
        if len(prompt_texts) == 0:
            return None, None

        # If inner lists (multiple captions per sample), join them into one sequence per sample by default
        # (alternative: you may want to keep them separate; adjust if needed)
        if isinstance(prompt_texts[0], (list, tuple)):
            # join each sample's captions into a single string
            prompt_texts = [" ".join(p) for p in prompt_texts]

        enc = self.text_tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        # forward through text encoder
        outputs = self.text_encoder(**enc)
        prompt_embeds = outputs.last_hidden_state  # [B, L_text, D_text]
        attn_mask = enc.get("attention_mask", None)
        return prompt_embeds, attn_mask

