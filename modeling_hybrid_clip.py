# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple
import os
from flax.serialization import to_bytes, from_bytes
import flax.linen as nn
import jax
import jax.numpy as jnp
from configuration_hybrid_clip import HybridCLIPConfig
from flax.core.frozen_dict import FrozenDict
from transformers import FLAX_MODEL_MAPPING
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.utils import logging
import flax
from transformers.file_utils import ModelOutput
import jaxlib.xla_extension as jax_xla
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from typing import Any, Optional, Tuple, Union

logger = logging.get_logger(__name__)

@flax.struct.dataclass
class FlaxSEOutput(ModelOutput):
    logits_per_text1: jax_xla.DeviceArray = None
    logits_per_text2: jax_xla.DeviceArray = None
    text_embeds1: jax_xla.DeviceArray = None
    text_embeds2: jax_xla.DeviceArray = None
    text_model_output1: FlaxBaseModelOutputWithPooling = None
    text_model_output2: FlaxBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output1", "text_model_output2"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class FlaxSEModule(nn.Module):
    config: HybridCLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        text_config = self.config.text_config
        self.text_embed_dim = text_config.hidden_size
        text_module = FLAX_MODEL_MAPPING[self.config.text_config.__class__].module_class
        self.text_model = text_module(text_config, dtype=self.dtype)
        self.logit_scale = self.param("logit_scale", jax.nn.initializers.ones, [])

    def __call__(
        self,
        input_ids1=None,
        attention_mask1=None,
        position_ids1=None,
        token_type_ids1=None,
        input_ids2=None,
        attention_mask2=None,
        position_ids2=None,
        token_type_ids2=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        text_outputs1 = self.text_model(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids1,
            position_ids=position_ids1,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs2 = self.text_model(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
            position_ids=position_ids2,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #CLS pooling
        #text_embeds1 = text_outputs1.last_hidden_state[:, 0]
        #text_embeds2 = text_outputs2.last_hidden_state[:, 0]

        #Pooling output
        text_embeds1 = text_outputs1[1]
        text_embeds2 = text_outputs2[1]


        # normalized features
        text_embeds1 = text_embeds1 / jnp.linalg.norm(text_embeds1, axis=-1, keepdims=True)
        text_embeds2 = text_embeds2 / jnp.linalg.norm(text_embeds2, axis=-1, keepdims=True)

        # cosine similarity as logits
#         logit_scale = jnp.exp(self.logit_scale)
        logit_scale = 20
        logits_per_text1 = jnp.matmul(text_embeds1, text_embeds2.T) * logit_scale
        logits_per_text2 = logits_per_text1.T

        if not return_dict:
            return (logits_per_text1, logits_per_text2, text_embeds1, text_embeds2, text_outputs1, text_outputs2)

        return FlaxSEOutput(
            logits_per_text1=logits_per_text1,
            logits_per_text2=logits_per_text2,
            text_embeds1=text_embeds1,
            text_embeds2=text_embeds2,
            text_model_output1=text_outputs1,
            text_model_output2=text_outputs2,
        )


class FlaxSE(FlaxPreTrainedModel):
    config_class = HybridCLIPConfig
    module_class = FlaxSEModule

    def __init__(
        self,
        config: HybridCLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        if input_shape is None:
            input_shape = [(1, 1), (1, 1)]

        self.module_obj = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, self.module_obj, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensor
        input_ids1 = jnp.zeros(input_shape[0], dtype="i4")
        position_ids1 = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids1).shape[-1]), input_shape[0])
        token_type_ids1 = jnp.ones_like(input_ids1)
        attention_mask1 = jnp.ones_like(input_ids1)

        input_ids2 = jnp.zeros(input_shape[1], dtype="i4")
        position_ids2 = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids2).shape[-1]), input_shape[0])
        token_type_ids2 = jnp.ones_like(input_ids1)
        attention_mask2 = jnp.ones_like(input_ids1)


        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_ids1, attention_mask1, position_ids1, token_type_ids1, input_ids2, attention_mask2, position_ids2, token_type_ids2)["params"]

    def __call__(
        self,
        input_ids1,
        attention_mask1=None,
        position_ids1=None,
        token_type_ids1=None,
        input_ids2=None,
        attention_mask2=None,
        position_ids2=None,
        token_type_ids2=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict


        if position_ids1 is None:
            position_ids1 = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids1).shape[-1]), input_ids1.shape)

        if position_ids2 is None:
            position_ids2 = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids2).shape[-1]), input_ids1.shape)

        if token_type_ids1 is None:
            token_type_ids1 = jnp.zeros_like(input_ids1)

        if token_type_ids2 is None:
            token_type_ids2 = jnp.zeros_like(input_ids2)

        if attention_mask1 is None:
            attention_mask1 = jnp.ones_like(input_ids1)

        if attention_mask2 is None:
            attention_mask2 = jnp.ones_like(input_ids2)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids1, dtype="i4"),
            jnp.array(attention_mask1, dtype="i4"),
            jnp.array(position_ids1, dtype="i4"),
            jnp.array(token_type_ids1, dtype="i4"),
            jnp.array(input_ids2, dtype="i4"),
            jnp.array(attention_mask2, dtype="i4"),
            jnp.array(position_ids2, dtype="i4"),
            jnp.array(token_type_ids2, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )

    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        dropout_rng: jax.random.PRNGKey = None,
        train=False,
    ):
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, input_ids, attention_mask, position_ids, token_type_ids, deterministic):
            text_outputs = module.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                deterministic=deterministic,
            )
            pooled_output = text_outputs[1]
            text_features = module.text_projection(pooled_output)
            return text_features

        return self.module.apply(
            {"params": self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            not train,
            method=_get_features,
            rngs=rngs,
        )

    """
    def save_pretrained(self, path, params):
        path = os.path.join(path, "flax_model.msgpack")
        with open(path, "wb") as f:
            f.write(to_bytes(params['text_model']))
    """

    @classmethod
    def from_text_pretrained(
        cls,
        text_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> FlaxPreTrainedModel:


        kwargs_text = {
            argument[len("text_") :]: value for argument, value in kwargs.items() if argument.startswith("text_")
        }

        # remove text, vision kwargs from kwargs
        for key in kwargs_text.keys():
            del kwargs["text_" + key]


        # Load and initialize the text and vision model
        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            assert (
                text_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
            from transformers import FlaxAutoModel

            if "config" not in kwargs_text:
                from transformers import AutoConfig

                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = FlaxAutoModel.from_pretrained(text_model_name_or_path, *model_args, **kwargs_text)


        # instantiate config with corresponding kwargs
        dtype = kwargs.pop("dtype", jnp.float32)
        config = HybridCLIPConfig.from_text_configs(text_model.config, **kwargs)

        # init model
        model = cls(config, *model_args, dtype=dtype, **kwargs)
        model.params["text_model"] = text_model.params

        return model
