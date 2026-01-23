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
from .modules.Film import FiLMTokenizerFusion
from .modules.LiteAdaLNZero import LiteAdaLNZero
from .modules.GatedFusion import GatedFusion
from .modules.dual_path_refiner import DualPathRefiner
from .modules.TAR import VisualGuidedTAR
from .modules.DPG import ChannelWiseDPG
from .modules.CBSA import CBSA
from .modules.MambaAdapter import MambaAdapter
from .modules.MoEAdapter import MoEAdapter
from .modules.VisualSelector import TextGuidedVisualSelector

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

        # 创建visual_proj / prompt_proj / gate_fusion
        enc_hidden = (
            self.encoder.config.hidden_size
            if hasattr(self.encoder.config, "hidden_size")
            else self.encoder.config.vision_config.hidden_size
        )
        fusion_dim = getattr(self.config, "fusion_dim", enc_hidden)

        # 定义要提取的多层特征索引（倒数第1、6、12层）
        self.feature_layers = [-1, -6, -12]

        # 图像投影：CLIP vision encoder输出 -> fusion_dim
        # 输入维度为 enc_hidden * len(self.feature_layers)，以接收拼接后的多层特征
        self.visual_proj = CBSAVisionProjector(in_dim=enc_hidden * len(self.feature_layers), out_dim=fusion_dim)

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

        # ===> BEGIN: 添加DPG模块用于增强text encoder输出的文本特征 <===
        self.use_dpg = getattr(self.config, "use_dpg", False)
        if self.use_dpg and self.text_encoder is not None:
            # === 使用 DPG (Detail-Oriented Prompt Generation) ===
            print("Initializing DPG Module (Detail-Oriented Prompt Generation)...")
            self.dpg_module = ChannelWiseDPG(
                dim=text_hidden,
                dropout=float(getattr(self.config, "dpg_dropout", 0.1))
            )

            # 维度对齐（如果视觉和文本维度不同）
            if fusion_dim != text_hidden:
                self.dpg_visual_proj = nn.Linear(fusion_dim, text_hidden)
            else:
                self.dpg_visual_proj = None
            # === 新增: 专门给 DPG 输入的视觉特征做归一化 ===
            self.dpg_visual_ln = nn.LayerNorm(text_hidden)
        else:
            self.dpg_module = None
            self.dpg_visual_proj = None
            self.dpg_visual_ln = None
        # ===> END: DPG模块 <===


        # ===> BEGIN: 添加上下文混合器 (Contextual Mixer) <===
        # 用于平衡DPG过滤后的特征和原始全局语义
        if self.use_dpg and self.text_encoder is not None:
            self.mixer_alpha = nn.Parameter(torch.ones(1) * 0.5)
        else:
            self.mixer_alpha = None
        # ===> END: 上下文混合器 <===

        # ===> BEGIN: 添加双路径增强模块 (DualPathRefiner) <===
        # 在 CLIP image encoder 输出后使用双路径增强模块
        self.use_dual_path_refiner = getattr(self.config, "use_dual_path_refiner", False)
        if self.use_dual_path_refiner:
            self.dual_path_refiner = DualPathRefiner(
                dim=fusion_dim,
                # AdapterResidual 参数
                adapter_down_ratio=int(getattr(self.config, "ar_down_ratio", 4)),
                adapter_dropout=float(getattr(self.config, "ar_dropout", 0.1)),
                adapter_use_gate=bool(getattr(self.config, "ar_use_gate", True)),
                # AttentionPoolingRefiner 参数
                apr_n_queries=int(getattr(self.config, "apr_n_queries", 1)),
                apr_n_heads=int(getattr(self.config, "apr_n_heads", 8)),
                apr_proj_back=bool(getattr(self.config, "apr_proj_back", True)),
                apr_dropout=float(getattr(self.config, "apr_dropout", 0.1)),
                # 双路径权重参数
                alpha1_init=float(getattr(self.config, "dual_path_init_alpha", 0.1)),
                alpha2_init=float(getattr(self.config, "dual_path_init_alpha", 0.1)),
            )
        # ===> END: 双路径增强模块 <===

        # ===> BEGIN: 使用VS模块（VisualSelector）进行视觉特征增强 <===
        self.use_visual_selector = getattr(self.config, "use_visual_selector", True)
        if self.use_visual_selector:
            print("Initializing TextGuidedVisualSelector (VS) for visual enhancement...")
            self.visual_selector = TextGuidedVisualSelector(
                dim=fusion_dim,
                dropout=float(getattr(self.config, "vs_dropout", 0.1))
            )
        else:
            self.visual_selector = None
        # ===> END: VS视觉增强模块 <===

        # === 融合模块初始化：使用 GatedFusion 模块 ===
        self.use_gated_fusion = getattr(self.config, "use_gated_fusion", False)

        if self.use_gated_fusion:
            print("Initializing GatedFusion module...")
            self.gate_fusion = GatedFusion(
                dim=fusion_dim,
                dropout=float(getattr(self.config, "gate_fusion_dropout", 0.1))
            )
        else:
            print("GatedFusion disabled. Parameters will not be loaded.")
            self.gate_fusion = None

        # ============================================================
        # 【在这里添加】Baseline Fusion 线性层
        # 用于消融实验：当不使用 GatedFusion 时，使用此层将拼接后的特征投影回原来的维度
        # 输入维度是 fusion_dim * 2 (Visual + Text 拼接)，输出维度是 fusion_dim
        # ============================================================
        self.baseline_proj = nn.Linear(fusion_dim * 2, fusion_dim)

        # ===> BEGIN: 辅助对比损失 (Auxiliary Contrastive Loss) <===
        # 用于监督融合层的中间特征，强制融合后的特征与 GT 文本的语义保持一致
        self.use_aux_loss = getattr(self.config, "use_aux_loss", True)
        if self.use_aux_loss:
            # 获取 CLIP text encoder 的隐藏维度（用于 GT text embeds 的维度）
            clip_embed_dim = text_hidden  # CLIP text encoder 的 hidden_size

            # 投影头：将融合特征映射到与 CLIP 文本嵌入相同的维度
            self.aux_projector = nn.Linear(fusion_dim, clip_embed_dim)

            # 损失函数权重（可以通过 config 配置，默认 0.1）
            self.lambda_aux = float(getattr(self.config, "lambda_aux", 0.1))

            # 余弦嵌入损失函数
            self.aux_loss_fn = nn.CosineEmbeddingLoss()
        # ===> END: 辅助对比损失 <===

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
            gt_text_embeds=None,
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

        # 从kwargs中提取gt_text_embeds（如果通过kwargs传递）
        if gt_text_embeds is None:
            gt_text_embeds = kwargs.pop("gt_text_embeds", None)

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

            # 强制开启 output_hidden_states 以获取所有层的特征
            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=True,  # 强制开启以获取多层特征
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)
        else:
            encoder_outputs = BaseModelOutput(encoder_outputs, None)

        # 多层视觉特征融合：从 hidden_states 中提取指定层的特征并拼接
        if hasattr(encoder_outputs, 'hidden_states') and encoder_outputs.hidden_states is not None:
            # 提取指定层的特征
            selected_features = []
            for layer_idx in self.feature_layers:
                # 使用负数索引从后往前访问（-1是最后一层，-6是倒数第6层，-12是倒数第12层）
                selected_features.append(encoder_outputs.hidden_states[layer_idx])
            
            # 在通道维度（dim=-1）拼接多层特征
            encoder_hidden_states = torch.cat(selected_features, dim=-1)  # [B, N, enc_hidden * len(feature_layers)]
        else:
            # 兼容性保底：如果没有 hidden_states，使用最后一层输出
            # 注意：这种情况下维度可能不匹配，因为 visual_proj 期望的输入维度是 enc_hidden * len(feature_layers)
            encoder_hidden_states = encoder_outputs[0]
            # 计算期望的输入维度
            expected_dim = (
                self.encoder.config.hidden_size
                if hasattr(self.encoder.config, "hidden_size")
                else self.encoder.config.vision_config.hidden_size
            ) * len(self.feature_layers)
            # 如果维度不匹配，需要重复或扩展特征（这里假设不会发生，因为我们强制开启了 hidden_states）
            if encoder_hidden_states.shape[-1] != expected_dim:
                # 如果维度不匹配，重复特征以匹配期望的输入维度
                repeat_factor = expected_dim // encoder_hidden_states.shape[-1]
                encoder_hidden_states = encoder_hidden_states.repeat(1, 1, repeat_factor)

        encoder_attention_mask = None
        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # ===== Step 1: 使用CLIP vision encoder生成image embedding并投影 =====
        # 将 encoder_hidden_states (B, N, C_clip) 投影到 fusion_dim
        img_feats = self.visual_proj(encoder_hidden_states)  # [B, N, fusion_dim]

        encoder_hidden_states = img_feats

        # 注意：VS模块（VisualSelector）将在文本特征处理后在Step 5中调用

        # 从 kwargs 获取检索到的 captions 字段（Trainer 的 batch 需包含 'retrieved_captions'）
        # 这些参数已经从 kwargs 中移除，所以这里不再需要获取

        if retrieved_caps is not None and self.text_tokenizer is not None and self.text_encoder is not None:
            # ===== Step 3: 使用CLIP text encoder生成caption embedding =====
            # encode_prompts 会接受 tensor 或字符串列表
            # 返回 (prompt_embeds_refined, prompt_embeds_raw, attn_mask)
            prompt_embeds_refined, prompt_embeds_raw, attn_mask, alpha = self.encode_prompts(retrieved_caps,
                                                                                      visual_feats=img_feats)

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

            # ===== Step 3.5: 使用上下文混合器平衡DPG过滤后的特征和原始全局语义 =====
            if self.mixer_alpha is not None and prompt_embeds_refined is not None and prompt_embeds_raw is not None:
                # f_refined_tokens: DPG模块的输出 [B, N, C]
                f_refined_tokens = prompt_embeds_refined
                # f_raw_text_feats: 原始的、未经过DPG处理的文本特征 [B, T, C]
                f_raw_text_feats = prompt_embeds_raw

                # 计算全局嵌入：对 f_raw_text_feats 进行平均池化，得到 f_global [B, C]
                f_global = f_raw_text_feats.mean(dim=1)  # [B, C]

                # 扩展全局嵌入到Token维度，使其形状变为 [B, N, C]
                # 其中 N 是 f_refined_tokens 的序列长度
                f_global_expanded = f_global.unsqueeze(1).expand_as(f_refined_tokens)  # [B, N, C]

                # 门控混合
                alpha = torch.sigmoid(self.mixer_alpha)
                f_mixed_prompt = alpha * f_refined_tokens + (1 - alpha) * f_global_expanded

                # 将混合后的特征替换原来的 f_refined_tokens
                prompt_embeds = f_mixed_prompt
            else:
                # 如果没有上下文混合器，直接使用DPG的输出（或原始特征）
                prompt_embeds = prompt_embeds_refined if prompt_embeds_refined is not None else prompt_embeds_raw

            # ===== Step 4: 投影caption embedding到融合维度 =====
            # prompt_embeds: [B, L, D_text] or [B, D_text] (proj 会扩维)
            prompt_feats = self.prompt_proj(prompt_embeds)  # [B, L, fusion_dim]

            # ===== Step 5: Fusion Pipeline =====
            
            # Step B (GatedFusion): Global modulation and fusion
            # 调用 GatedFusion 进行跨模态融合，返回增强后的图像特征 [B, N, fusion_dim]
            # GatedFusion需要img_feats和text_feats都是[B, N, C]形状
            # 将prompt_feats池化为[B, C]，然后扩展到[B, N, C]以匹配img_feats的序列长度
            B, N, C = img_feats.shape
            if prompt_feats.dim() == 3:
                # [B, L, C] -> [B, C] (平均池化)
                prompt_feats_pooled = prompt_feats.mean(dim=1)
            else:
                # [B, C] 已经是池化后的
                prompt_feats_pooled = prompt_feats
            # 扩展到 [B, N, C] 以匹配img_feats的序列长度
            prompt_feats_expanded = prompt_feats_pooled.unsqueeze(1).expand(B, N, C)

            # ================= 消融实验控制区 =================

            # 1. 设置开关 (实际使用时建议写在 config 里，这里为了演示直接写变量)
            use_vs_module = getattr(self.config, "use_visual_selector", False)  # 是否用 VS
            use_smart_fusion = getattr(self.config, "use_gated_fusion", False)  # True=你的GatedFusion, False=基准融合

            # --- 模块 A: Visual Selector (视觉去噪) ---
            if use_vs_module and self.visual_selector is not None:
                # 你的创新点：用文本过滤视觉
                # 传入 DPG 计算出的 alpha 作为 reliability，用于可靠性控制
                img_feats = self.visual_selector(img_feats, prompt_feats, reliability=alpha)
            else:
                # Baseline：不做任何视觉过滤，原封不动
                pass

            # --- 模块 B: Fusion Strategy (融合方式) ---
            if use_smart_fusion:
                # === 你的创新点：Reliability-Aware Gated Fusion ===

                # 准备 reliability (来自 DPG 的 alpha)
                global_reliability = None
                if alpha is not None:
                    # 只对 Sequence 维度 (dim=1) 进行平均，保留 Channel 维度以便进行细粒度的去噪
                    global_reliability = alpha.mean(dim=1, keepdim=True)

                # 调用你的高级融合模块
                img_feats = self.gate_fusion(img_feats, prompt_feats_expanded, reliability=global_reliability)

            else:
                # === Baseline Fusion：简单的拼接 + 线性层 ===
                # 模拟最普通的 SmallCap 或其它 RAG 模型做法
                # 临时定义一个线性层 (注意：为了严谨，最好在 __init__ 里定义 self.baseline_linear)
                print("DEBUG: Using Baseline Fusion (Linear Projection)")
                if not hasattr(self, 'baseline_proj'):
                    # 懒加载：如果是测试代码，可以在这里临时定义，或者去 __init__ 加一行
                    self.baseline_proj = nn.Linear(C * 2, C).to(img_feats.device)

                # 简单的拼接融合
                concat_feats = torch.cat([img_feats, prompt_feats_expanded], dim=-1)  # [B, N, 2C]
                img_feats = self.baseline_proj(concat_feats)  # [B, N, C]

                # 或者更简单的：直接相加 (Add Fusion)
                # img_feats = img_feats + prompt_feats_expanded

            # ===============================================

            # 最终把 img_feats 传给 decoder
            encoder_hidden_states = img_feats

        # ===== Step 6: 计算辅助对比损失 (Auxiliary Contrastive Loss) =====
        loss_aux = None
        if self.use_aux_loss and gt_text_embeds is not None and labels is not None:
            # 融合后的特征 img_feats 的形状: [Batch, Seq_Len, fusion_dim]
            fusion_features = img_feats  # [B, N, fusion_dim]

            # 1. 池化：对融合特征进行平均池化，去掉序列维度
            pooled_feat = fusion_features.mean(dim=1)  # [B, fusion_dim]

            # 2. 投影：将融合特征映射到 CLIP 文本嵌入维度
            projected_feat = self.aux_projector(pooled_feat)  # [B, clip_embed_dim]

            # 3. 确保 gt_text_embeds 的维度正确
            if gt_text_embeds.dim() == 3:
                # 如果是 [B, L, D]，进行池化
                gt_text_embeds = gt_text_embeds.mean(dim=1)  # [B, clip_embed_dim]
            elif gt_text_embeds.dim() == 2:
                # 如果是 [B, D]，已经是正确的形状
                pass
            else:
                # 如果是其他维度，报错或跳过
                raise ValueError(f"Unsupported gt_text_embeds dimension: {gt_text_embeds.dim()}, expected 2 or 3")

            # 4. 计算余弦嵌入损失（目标标签为1，表示应相似）
            batch_size = projected_feat.size(0)
            device = projected_feat.device
            target = torch.ones(batch_size, device=device)
            loss_aux = self.aux_loss_fn(projected_feat, gt_text_embeds, target)

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
        loss_caption = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss_caption = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

            # 合并辅助损失到总损失
            if loss_aux is not None:
                loss = loss_caption + self.lambda_aux * loss_aux
            else:
                loss = loss_caption
        else:
            # 即使没有labels，如果有辅助损失，也可以计算（虽然通常不会发生）
            if loss_aux is not None:
                loss = self.lambda_aux * loss_aux

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        # 将辅助损失添加到返回字典中以便监控
        # 注意：Seq2SeqLMOutput 没有 loss_aux 字段，所以我们返回 loss
        # 如果需要监控 loss_aux，可以通过自定义返回或日志记录
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

        # === [新增] 辅助函数：根据 input_ids 的 Batch Size 扩展自定义参数 ===
        def expand_to_match_beams(tensor, target_bsz):
            if tensor is None:
                return None
            curr_bsz = tensor.shape[0]
            # 如果当前 Batch Size 小于目标 Batch Size (说明开启了 Beam Search)
            if target_bsz > curr_bsz:
                num_beams = target_bsz // curr_bsz
                # 使用 repeat_interleave 进行复制 (e.g., [A, B] -> [A, A, A, B, B, B])
                return tensor.repeat_interleave(num_beams, dim=0)
            return tensor

        # 获取当前实际的 Batch Size (包含 Beam Search 扩展后的)
        target_bsz = input_ids.shape[0]

        # 从kwargs中提取retrieved_captions相关参数，并进行必要的扩展
        if "retrieved_captions" in kwargs:
            val = kwargs.pop("retrieved_captions")
            input_dict["retrieved_captions"] = expand_to_match_beams(val, target_bsz)

        if "retrieved_captions_mask" in kwargs:
            val = kwargs.pop("retrieved_captions_mask")
            input_dict["retrieved_captions_mask"] = expand_to_match_beams(val, target_bsz)

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

    def encode_prompts(self, prompt_input, visual_feats=None):
        """
            Accept:
              - None
              - list[str] (长度 batch_size) OR list[list[str]] (每个 sample 为多个 retrieved captions)
              - torch.Tensor of shape [B, K, D_text] (precomputed pooled embeddings) OR [B, D_text]
            Args:
              visual_feats: (B, T_vis, fusion_dim) - 视觉特征，用于VG-TAR跨模态交互
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
            return None, None, None, None

        # If already tensor: assume pooled or token-level embeddings provided
        if torch.is_tensor(prompt_input):
            x = prompt_input.to(device)
            x_raw = x  # 保存原始特征
            alpha = None
            # if 2D -> [B, D] treat as pooled single prompt per image
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [B, 1, D_text]
                x_raw = x  # 更新原始特征
                attn_mask = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=device)
                if self.dpg_module is not None and visual_feats is not None:
                    visual_feats = visual_feats.to(device)
                    if self.dpg_visual_proj is not None:
                        visual_feats = self.dpg_visual_proj(visual_feats)
                    if self.dpg_visual_ln is not None:
                        visual_feats = self.dpg_visual_ln(visual_feats)
                    x, alpha = self.dpg_module(x, visual_feats)
                return x, x_raw, attn_mask, alpha
            # if 3D -> [B, K, D_text] already token-level/pool-level per caption
            if x.dim() == 3:
                attn_mask = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=device)
                if self.dpg_module is not None and visual_feats is not None:
                    visual_feats = visual_feats.to(device)
                    if self.dpg_visual_proj is not None:
                        visual_feats = self.dpg_visual_proj(visual_feats)
                    if self.dpg_visual_ln is not None:
                        visual_feats = self.dpg_visual_ln(visual_feats)
                    x, alpha = self.dpg_module(x, visual_feats)
                return x, x_raw, attn_mask, alpha
            raise ValueError("Unsupported tensor dim for prompt_input: got dim = %d" % x.dim())

        # Else expect list[str] or list[list[str]]
        prompt_texts = prompt_input
        if len(prompt_texts) == 0:
            return None, None, None

        # If inner lists (multiple captions per sample), join them into one sequence per sample by default
        # (alternative: you may want to keep them separate; adjust if needed)
        if isinstance(prompt_texts[0], (list, tuple)):
            # join each sample's captions into a single string
            prompt_texts = [" ".join(p) for p in prompt_texts]

        enc = self.text_tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        # forward through text encoder
        outputs = self.text_encoder(**enc)
        prompt_embeds_raw = outputs.last_hidden_state  # [B, L_text, D_text] - 原始特征

        # ===> BEGIN: DPG模块增强text encoder输出的文本特征 <===
        prompt_embeds_refined = prompt_embeds_raw  # 默认使用原始特征
        alpha = None
        if self.dpg_module is not None and visual_feats is not None:
            visual_feats = visual_feats.to(device)
            if self.dpg_visual_proj is not None:
                visual_feats = self.dpg_visual_proj(visual_feats)
            if self.dpg_visual_ln is not None:
                visual_feats = self.dpg_visual_ln(visual_feats)
            prompt_embeds_refined, alpha = self.dpg_module(prompt_embeds_raw, visual_feats)
        # ===> END: DPG模块增强 <===

        attn_mask = enc.get("attention_mask", None)
        # 返回 (refined_embeds, raw_embeds, attn_mask) 用于上下文混合器
        return prompt_embeds_refined, prompt_embeds_raw, attn_mask, alpha
