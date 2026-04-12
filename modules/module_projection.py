"""Projection bridge from Q-Former hidden size to FLAN-T5 hidden size."""

import torch.nn as nn


class ProjectionLayer(nn.Module):
	def __init__(self, input_dim=768, output_dim=2048):
		super().__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return self.linear(x)
