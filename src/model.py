import math
import numpy as np
import copy
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

#Local Context Focus implementation of BERT
#See https://mdpi-res.com/d_attachment/applsci/applsci-09-03389/article_deploy/applsci-09-03389-v3.pdf
class LCF_BERT(nn.Module):
    def __init__(self, bert, dropout, bert_dim, polarities_dim, max_seq_len, device, SRD, local_context_focus):
        super(LCF_BERT, self).__init__()
        
        #Define and assign parameters, layers, etc.
        self.bert_spc = bert
        # for dual bert channel
        # self.bert_local = copy.deepcopy(bert)
        self.bert_local = bert
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.bert_SA = SelfAttention(bert.config, max_seq_len, self.device)
        self.linear_double = nn.Linear(bert_dim * 2, bert_dim)
        self.linear_single = nn.Linear(bert_dim, bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(bert_dim, polarities_dim)
        self.max_seq_len = max_seq_len
        self.bert_dim = bert_dim
        self.SRD = SRD
        self.local_context_focus = local_context_focus

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.max_seq_len, self.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= self.SRD:
                mask_begin = asp_begin - self.SRD
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + self.SRD, self.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.bert_dim), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.max_seq_len, self.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i])-1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index)+asp_len/2
                                        - self.SRD)/np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.device)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, text_local_indices, aspect_indices = inputs

        bert_spc_out, _ = self.bert_spc(
            text_bert_indices, token_type_ids=bert_segments_ids, return_dict=False)
        bert_spc_out = self.dropout(bert_spc_out)

        bert_local_out, _ = self.bert_local(text_local_indices, return_dict=False)
        bert_local_out = self.dropout(bert_local_out)

        if self.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)
        mean_pool = self.linear_double(out_cat)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)

        return dense_out


class SelfAttention(nn.Module):
    def __init__(self, config, max_seq_len, device):
        super(SelfAttention, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.SA = BertSelfAttention(config)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob if hasattr(
            config, 'attention_probs_dropout_prob') else 0)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.self.device).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.self.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + \
                    relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
