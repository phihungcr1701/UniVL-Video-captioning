"""
FLAN-T5 wrapper with LoRA and SCST helpers.

Reference reuse:
- mllm-video-captioner/lavis/models/blip2_models/blip2_t5.py
"""

import itertools
import logging

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast

logger = logging.getLogger(__name__)

try:
	from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
	from pycocoevalcap.cider.cider import Cider
	HAS_COCO_EVAL = True
except Exception:
	HAS_COCO_EVAL = False


def _tokenize_for_cider(refs, cands):
	tokenizer = PTBTokenizer()
	refs = {idx: [{"caption": r} for r in c_refs] for idx, c_refs in enumerate(refs)}
	cands = {idx: [{"caption": c}] for idx, c in enumerate(cands)}
	refs = tokenizer.tokenize(refs)
	cands = tokenizer.tokenize(cands)
	return refs, cands


class T5Model(nn.Module):
	def __init__(
		self,
		t5_model_name="google/flan-t5-xl",
		lora=True,
		lora_r=8,
		lora_alpha=16,
		lora_dropout=0.05,
		scst=False,
		beam_size=5,
		max_txt_len=32,
	):
		super().__init__()

		self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)
		t5_config = T5Config.from_pretrained(t5_model_name)
		t5_config.dense_act_fn = "gelu"
		self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name, config=t5_config)

		for _, param in self.t5_model.named_parameters():
			param.requires_grad = False
			param.data = param.data.bfloat16()

		if lora:
			peft_config = LoraConfig(
				task_type=TaskType.SEQ_2_SEQ_LM,
				inference_mode=False,
				r=lora_r,
				lora_alpha=lora_alpha,
				lora_dropout=lora_dropout,
				target_modules=["q", "v"],
			)
			self.t5_model = get_peft_model(self.t5_model, peft_config)
			self.t5_model.print_trainable_parameters()

		self.scst = scst
		self.beam_size = beam_size
		self.max_txt_len = max_txt_len

	def build_prompt_embeds(self, prompt_text, batch_size, device):
		prompt_list = [prompt_text] * batch_size
		prompt_tokens = self.t5_tokenizer(
			prompt_list,
			padding="longest",
			truncation=True,
			max_length=self.max_txt_len,
			return_tensors="pt",
		).to(device)
		prompt_embeds = self.t5_model.encoder.embed_tokens(prompt_tokens.input_ids)
		return prompt_embeds, prompt_tokens.attention_mask

	def forward_xe(
		self,
		inputs_embeds,
		attention_mask,
		decoder_input_ids,
		decoder_attention_mask,
	):
		outputs = self.t5_model(
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			return_dict=True,
		)
		return outputs

	def generate(
		self,
		inputs_embeds,
		attention_mask,
		num_beams=5,
		max_length=30,
		num_return_sequences=1,
		output_scores=False,
		return_dict_in_generate=False,
		**kwargs
	):
		return self.t5_model.generate(
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
			do_sample=False,
			top_p=0.9,
			temperature=1.0,
			num_beams=num_beams,
			max_new_tokens=max_length,
			repetition_penalty=1.0,
			length_penalty=1.0,
			num_return_sequences=num_return_sequences,
			output_scores=output_scores,
			return_dict_in_generate=return_dict_in_generate,
			**kwargs
		)

	def compute_scst_loss(self, inputs_embeds, attention_mask, references, device, beam_size=None):
		beam_size = beam_size if beam_size is not None else self.beam_size
		batch_size = inputs_embeds.size(0)

		outputs = self.generate(
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
			num_beams=beam_size,
			max_length=32,
			num_return_sequences=beam_size,
			output_scores=True,
			return_dict_in_generate=True,
		)

		transition_scores = self.t5_model.compute_transition_scores(
			outputs.sequences,
			outputs.scores,
			outputs.beam_indices,
			normalize_logits=False,
		)
		output_length = torch.sum(transition_scores < 0, dim=1)
		sequences_scores = transition_scores.sum(dim=1) / (output_length ** 1.0)
		sequences_scores = sequences_scores.view(batch_size, -1)

		if not HAS_COCO_EVAL:
			# CHANGE: fallback when pycocoevalcap is unavailable.
			logger.warning("pycocoevalcap not found; SCST fallback reward uses score mean baseline.")
			reward = sequences_scores.detach()
			reward_baseline = reward.mean(dim=-1, keepdim=True)
			return (-(sequences_scores) * (reward - reward_baseline)).mean()

		caps_gen = self.t5_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
		caps_gen = [text.strip() for text in caps_gen]
		caps_gt = list(itertools.chain(*([c] * beam_size for c in references)))
		caps_gt = [[c] for c in caps_gt]

		caps_gen, caps_gt = _tokenize_for_cider(caps_gt, caps_gen)
		reward = Cider().compute_score(caps_gt, caps_gen)[1].astype(np.float32)
		reward = torch.from_numpy(reward).to(device).view(batch_size, beam_size)
		reward_baseline = torch.mean(reward, -1, keepdim=True)
		return (-(sequences_scores) * (reward - reward_baseline)).mean()
