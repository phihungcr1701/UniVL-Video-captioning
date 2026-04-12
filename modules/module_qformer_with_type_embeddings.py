"""
Q-Former wrapper with type embeddings for text/video distinction.

Reference reuse:
- BLIP2 init_Qformer pattern (BertConfig + BertLMHeadModel + query tokens)
- UniVL-style type embeddings over concatenated text/video features
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertLMHeadModel


class QFormerBackbone(nn.Module):
	"""Container for BLIP2-style Q-Former backbone and learnable query tokens."""

	def __init__(self, qformer, query_tokens):
		super().__init__()
		self.bert = qformer.bert
		self.query_tokens = query_tokens


def init_qformer_backbone(
	num_query_tokens=32,
	vision_width=768,
	pretrained_name="bert-base-uncased",
	cross_attention_freq=2,
):
	"""
	CHANGE: Ported from BLIP2 init_Qformer with minimal edits for UniVL modules.
	"""
	encoder_config = BertConfig.from_pretrained(pretrained_name)
	encoder_config.encoder_width = vision_width
	# CHANGE: Newer transformers requires decoder mode when using cross-attention in BertLayer.
	encoder_config.is_decoder = True
	encoder_config.add_cross_attention = True
	encoder_config.cross_attention_freq = cross_attention_freq
	encoder_config.query_length = num_query_tokens

	qformer = BertLMHeadModel.from_pretrained(pretrained_name, config=encoder_config)
	query_tokens = nn.Parameter(
		torch.zeros(1, num_query_tokens, encoder_config.hidden_size)
	)
	query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

	# CHANGE: Keep this BLIP2 pruning behavior to align runtime and memory.
	qformer.cls = None
	qformer.bert.embeddings.word_embeddings = None
	qformer.bert.embeddings.position_embeddings = None
	for layer in qformer.bert.encoder.layer:
		layer.output = None
		layer.intermediate = None

	return QFormerBackbone(qformer, query_tokens)


class QFormerWithTypeEmbeddings(nn.Module):
	"""
	CHANGE: UniVL-specific wrapper adds type embeddings (0=text, 1=video)
	before feeding encoder_hidden_states to Q-Former.
	"""

	def __init__(self, qformer_backbone, hidden_size=768, max_position_embeddings=512):
		super().__init__()
		self.qformer = qformer_backbone
		self.type_embeddings = nn.Embedding(2, hidden_size)
		self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
		self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
		self.dropout = nn.Dropout(0.1)

	def forward(self, concat_features, concat_type, attention_mask):
		batch_size, seq_len = concat_features.shape[:2]

		position_ids = torch.arange(seq_len, dtype=torch.long, device=concat_features.device)
		position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

		position_embeddings = self.position_embeddings(position_ids)
		type_embeddings = self.type_embeddings(concat_type)

		embeddings = concat_features + position_embeddings + type_embeddings
		embeddings = self.layer_norm(embeddings)
		embeddings = self.dropout(embeddings)

		query_tokens = self.qformer.query_tokens.expand(batch_size, -1, -1)
		query_attention_mask = torch.ones(
			query_tokens.size()[:-1],
			dtype=torch.long,
			device=query_tokens.device,
		)

		# CHANGE: HuggingFace BertModel uses `inputs_embeds`, not BLIP2's `query_embeds`.
		query_output = self.qformer.bert(
			inputs_embeds=query_tokens,
			attention_mask=query_attention_mask,
			encoder_hidden_states=embeddings,
			encoder_attention_mask=attention_mask,
			return_dict=True,
		)

		return query_output.last_hidden_state
