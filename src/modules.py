import torch
import torch.nn as nn
from typing import Optional, Tuple
import math
import inspect

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

@dataclass
class MaskedLMOutput():

    logits: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]]
    attentions: Optional[Tuple[torch.FloatTensor]]
    
@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions():

    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]]
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]
    attentions: Optional[Tuple[torch.FloatTensor]]

@dataclass
class BaseModelOutputWithPastAndCrossAttentions():

    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]
    hidden_states: Optional[Tuple[torch.FloatTensor]]
    attentions: Optional[Tuple[torch.FloatTensor]]
#------------------------------------------------------------------------------ A (BERT Model)
class BertModel(nn.Module):

    def __init__(self, config, add_pooling_layer=False):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        
        self.encoder = BertEncoder(config)


    def get_extended_attention_mask(self, attention_mask, input_shape, device) -> torch.Tensor:
        

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
            
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask #.to(dtype=self.dtype)  # fp16 compatibility
        self.dtype = extended_attention_mask.dtype
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        
        return extended_attention_mask
    
    
    def forward(self, attention_mask=None, token_type_ids=None, head_mask=None, inputs_embeds=None,
                encoder_attention_mask=None, output_attentions=None, output_hidden_states=None, past_key_values=None, use_cache=False):

        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify inputs_embeds")

        device = inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)


        encoder_extended_attention_mask = None


        head_mask = [head_mask] * self.config.num_hidden_layers

        
        embedding_output = self.embeddings(inputs_embeds=inputs_embeds)
        

        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask, encoder_extended_attention_mask,
                                       past_key_values, use_cache, output_attentions, output_hidden_states)
        
        
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = None

        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, past_key_values=encoder_outputs.past_key_values,
                                                            attentions=encoder_outputs.attentions)

    
class BertForMaskedLM(nn.Module):


    def __init__(self, config):
        super().__init__()


        self.bert = BertModel(config, add_pooling_layer=False)
        
        self.cls = BertOnlyMLMHead(config)

    def forward(self, attention_mask=None, token_type_ids=None, head_mask=None, inputs_embeds=None, encoder_attention_mask=None, output_attentions=None, 
                output_hidden_states=None, past_key_values=None, use_cache=False):
      

        outputs = self.bert(attention_mask, token_type_ids, head_mask, inputs_embeds, encoder_attention_mask, output_attentions, output_hidden_states, past_key_values, use_cache)
        
        
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)


        return MaskedLMOutput(logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# ----------------------------------------------------------------------------- A
class PositionalEncoding(nn.Module): ########### A11

    def __init__(self, model_hs: int, max_len: int = 5000):
        super().__init__()
        
        dropout=float = 0.1
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_hs, 2) * (-math.log(10000.0) / model_hs))
        pe = torch.zeros(max_len, 1, model_hs)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.pe[:x.size(1)]
        return self.dropout(x)

class BertEmbeddings(nn.Module): ########### A1
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)

        self.position_embeddings = PositionalEncoding(model_hs=config.hidden_size, max_len=config.max_position_embeddings)

        
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        
    def forward(self, inputs_embeds):
        embeddings = self.word_embeddings(inputs_embeds)
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings += position_embeddings.swapaxes(0, 1)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    

#---------


class BertSelfAttention(nn.Module): ############## A211
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                             f"heads ({config.num_attention_heads})")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size


        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)



        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None, head_mask: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, output_attentions: Optional[bool] = False) -> Tuple[torch.Tensor]:
        
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # scaling

        if attention_mask is not None: # Not none
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask


        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            
        return outputs
    
class PreLNBertSelfOutput(nn.Module): ############## A212
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor 
        return hidden_states
    
    
class PreLNBertAttention(nn.Module): #################### A21
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self = BertSelfAttention(config)
        self.output = PreLNBertSelfOutput(config)
        

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, past_key_value=None, encoder_attention_mask=None):
        
        hidden_states = self.LayerNorm(hidden_states)
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_attention_mask, past_key_value, output_attentions)
        
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:] # skip connection
        return outputs

    
#---------
#---------


class PreLNBertIntermediate(nn.Module): ##################### A22
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
            
    def forward(self, hidden_states):
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    
#---------
#---------


class PreLNBertOutput(nn.Module): ###################### A23
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states
    
    

#---------
#---------
    
def apply_chunking_to_forward(forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors) -> torch.Tensor:

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
                         "tensors are given")

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(f"All input tenors have to be of the same shape: {tensor_shape}, "
                                 f"found shape {input_tensor.shape[chunk_dim]}")

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


#--------------


class BertLayer(nn.Module):   ###################### A2
    def __init__(self, config):
        super().__init__()
        
        self.chunk_size_feed_forward = config.chunk_size_feed_forward # = 0
        self.seq_len_dim = 1



        self.attention = PreLNBertAttention(config) #BertAttention
        self.intermediate = PreLNBertIntermediate(config) # BertIntermediate
        self.output = PreLNBertOutput(config) #BertOutput

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        
        # decoder uni-directional self-attention cached key/values tuple is at positions 1, 2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None # None

        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]


        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights


        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        #print(layer_output)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output




class BertEncoder(nn.Module):  ###################### A2
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False):
        
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None


        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_attention_mask, past_key_value, output_attentions)
                

            hidden_states = layer_outputs[0]

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states,  past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions)

# -----------------------------------------HEAD---------------------------------------------------------

# --------------------------------------PRETRAINING-----------------------------------------------------
    
class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = nn.Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        
        return output


# --------------------------------------FINETUNING - DENOISING--------------------------------------------


class DenoisingHead(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = nn.Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        
        return output


# ----------------------------------VELOCITY PREDICTION ----------------------------------------------------


class VelocitypredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vel_size)
        self.predictions.decoder = nn.Identity()
        self.vel_min = config.vel_min
        self.vel_max = config.vel_max
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        output = torch.mean(output[:, :1, :], dim=1)
        output = self.vel_min + (output + 1) * (self.vel_max - self.vel_min) * 0.5
        
        return output
    