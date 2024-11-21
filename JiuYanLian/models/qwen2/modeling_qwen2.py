# from transformers import Qwen2Config, Qwen2Tokenizer, Qwen2ForCausalLM
from typing import OrderedDict
from transformers.models.qwen2.modeling_qwen2 import *

class Qwen2ForCausalLM4PP(Qwen2ForCausalLM):
    def forward(self,*args, **kwargs):
        output = super().forward(*args, **kwargs)        
        return output.logits

class AdaptiveQwen2SdpaAttention(Qwen2Attention):
    """
    from Qwen2FlashAttention2: self.num_heads  -->  -1
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        # print("apply_rotary_pos_emb", query_states.shape, key_states.shape, cos.shape, sin.shape)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    
import transformers.models.qwen2.modeling_qwen2 as transformers_modeling_qwen2
transformers_modeling_qwen2.QWEN2_ATTENTION_CLASSES['sdpa']=AdaptiveQwen2SdpaAttention
# Qwen2ForCausalLM4TP = transformers_modeling_qwen2.Qwen2ForCausalLM
class Qwen2ForCausalLM4TP( transformers_modeling_qwen2.Qwen2ForCausalLM):
  def forward(
      self,
      input_ids: torch.LongTensor = None,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
      past_key_values: Optional[List[torch.FloatTensor]] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.LongTensor] = None,
      use_cache: Optional[bool] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      cache_position: Optional[torch.LongTensor] = None,
      num_logits_to_keep: int = 0,
      **loss_kwargs,
  ) -> Union[Tuple, CausalLMOutputWithPast]:
      r"""
      Args:
          labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
              Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
              config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
              (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

          num_logits_to_keep (`int`, *optional*):
              Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
              `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
              token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

      Returns:

      Example:

      ```python
      >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

      >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
      >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

      >>> prompt = "Hey, are you conscious? Can you talk to me?"
      >>> inputs = tokenizer(prompt, return_tensors="pt")

      >>> # Generate
      >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
      >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
      "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
      ```"""

      # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

      if (input_ids is None) ^ (inputs_embeds is not None):
          raise ValueError("You must specify exactly one of input_ids or inputs_embeds")


      inputs_embeds=input_ids
      if self.model.embed_tokens:
          inputs_embeds = self.model.embed_tokens(input_ids)

      if cache_position is None:
          past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
          cache_position = torch.arange(
              past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
          )
      if position_ids is None:
          position_ids = cache_position.unsqueeze(0)

      causal_mask = self.model._update_causal_mask(
          attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
      )

      hidden_states = inputs_embeds

      # create position embeddings to be shared across the decoder layers
      position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

      # decoder layers
      for decoder_layer in self.model.layers:
          if self.model.gradient_checkpointing and self.training:
              layer_outputs = self.model._gradient_checkpointing_func(
                  decoder_layer.__call__,
                  hidden_states,
                  causal_mask,
                  position_ids,
                  past_key_values,
                  output_attentions,
                  use_cache,
                  cache_position,
                  position_embeddings,
              )
          else:
              layer_outputs = decoder_layer(
                  hidden_states,
                  attention_mask=causal_mask,
                  position_ids=position_ids,
                  past_key_value=past_key_values,
                  output_attentions=output_attentions,
                  use_cache=use_cache,
                  cache_position=cache_position,
                  position_embeddings=position_embeddings,
              )
          hidden_states = layer_outputs[0]
      if self.model.norm:
        hidden_states = self.model.norm(hidden_states)

      # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
      if self.lm_head:
        hidden_states = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
      return hidden_states
      
if __name__=='__main__':
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
    model=LlamaConfig.from_pretrained("Meta-Llama-3-8B")
    # model=LlamaForCausalLM.from_pretrained("/Meta-Llama-3-8B")
    # model=Qwen2Config.from_pretrained("Qwen2.5-7B")
    # model=Qwen2ForCausalLM.from_pretrained("Qwen2.5-7B")
    print(model)

"""
for TP, num_heads, pos_embed, 

LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "vocab_size": 128256
}

Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 131072,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_mrope": false,
  "use_sliding_window": false,
  "vocab_size": 152064
}


LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)

Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 3584)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
          (k_proj): Linear(in_features=3584, out_features=512, bias=True)
          (v_proj): Linear(in_features=3584, out_features=512, bias=True)
          (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((3584,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3584, out_features=152064, bias=False)
)
"""