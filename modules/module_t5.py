"""
FLAN-T5 wrapper with LoRA and SCST helpers.

Reference reuse:
- mllm-video-captioner/lavis/models/blip2_models/blip2_t5.py
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
		_ = beam_size
		max_new_tokens = self.max_txt_len
		sampled_ids, sampled_logprob_sum, sampled_lengths = self._sample_with_logprobs(
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
			max_new_tokens=max_new_tokens,
		)

		with torch.no_grad():
			greedy_ids = self.t5_model.generate(
				inputs_embeds=inputs_embeds,
				attention_mask=attention_mask,
				do_sample=False,
				num_beams=1,
				max_new_tokens=max_new_tokens,
			)

		sampled_text = self.t5_tokenizer.batch_decode(sampled_ids, skip_special_tokens=True)
		sampled_text = [text.strip() for text in sampled_text]
		greedy_text = self.t5_tokenizer.batch_decode(greedy_ids, skip_special_tokens=True)
		greedy_text = [text.strip() for text in greedy_text]

		reward_sample = self._compute_reward(references, sampled_text, device)
		reward_greedy = self._compute_reward(references, greedy_text, device)
		advantage = (reward_sample - reward_greedy).detach()

		normalized_logprob = sampled_logprob_sum / sampled_lengths.clamp(min=1.0)
		loss = -(advantage * normalized_logprob).mean()
		return loss

	def _compute_reward(self, references, candidates, device):
		if HAS_COCO_EVAL:
			refs = [[ref] for ref in references]
			refs_tok, cands_tok = _tokenize_for_cider(refs, candidates)
			scores = Cider().compute_score(refs_tok, cands_tok)[1].astype(np.float32)
			return torch.from_numpy(scores).to(device)

		logger.warning("pycocoevalcap not found; using exact-match reward for SCST.")
		scores = []
		for ref, cand in zip(references, candidates):
			scores.append(float(ref.strip().lower() == cand.strip().lower()))
		return torch.tensor(scores, device=device, dtype=torch.float32)

	def _sample_with_logprobs(self, inputs_embeds, attention_mask, max_new_tokens):
		batch_size = inputs_embeds.size(0)
		device = inputs_embeds.device
		start_token_id = self.t5_model.config.decoder_start_token_id
		eos_token_id = self.t5_model.config.eos_token_id

		if start_token_id is None:
			start_token_id = self.t5_tokenizer.pad_token_id
		if eos_token_id is None:
			eos_token_id = self.t5_tokenizer.eos_token_id

		decoder_input_ids = torch.full(
			(batch_size, 1),
			start_token_id,
			dtype=torch.long,
			device=device,
		)
		unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
		logprob_sum = torch.zeros(batch_size, device=device)
		lengths = torch.zeros(batch_size, device=device)

		for _ in range(max_new_tokens):
			outputs = self.t5_model(
				inputs_embeds=inputs_embeds,
				attention_mask=attention_mask,
				decoder_input_ids=decoder_input_ids,
				return_dict=True,
			)
			next_token_logits = outputs.logits[:, -1, :].float()
			log_probs = F.log_softmax(next_token_logits, dim=-1)
			probs = log_probs.exp()
			sampled_token = torch.multinomial(probs, num_samples=1)

			token_logprob = log_probs.gather(1, sampled_token).squeeze(1)
			active = unfinished.float()
			logprob_sum = logprob_sum + token_logprob * active
			lengths = lengths + active

			next_token = sampled_token.squeeze(1)
			next_token = torch.where(
				unfinished,
				next_token,
				torch.full_like(next_token, eos_token_id),
			)
			decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(1)], dim=1)
			unfinished = unfinished & (next_token != eos_token_id)
			if not unfinished.any():
				break

		return decoder_input_ids, logprob_sum, lengths
