import torch

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel


class BertForContinualFineTuning(BertPreTrainedModel):
	"""docstring for BertForContinualFineTuning"""
	def __init__(self, config):
		super().__init__(config)
		# self.bert = BertModel(configs[list(configs.keys())[0]])
		# self.dropouts = {}
		# self.classifiers = {}
		# self.num_labels = {}
		# for task, config in configs.items():
		# 	self.dropouts[task] = nn.Dropout(config.hidden_dropout_prob)
		# 	self.classifiers[task] = nn.Linear(config.hidden_size, config.num_labels).to(device)
		# 	self.num_labels[task] = config.num_labels
		# self.init_weights()

	def load_pretrained(self, configs, device):
		self.bert = BertModel(configs[list(configs.keys())[0]])
		self.dropouts = {}
		self.classifiers = {}
		self.num_labels = {}
		for task, config in configs.items():
			self.dropouts[task] = nn.Dropout(config.hidden_dropout_prob)
			self.classifiers[task] = nn.Linear(config.hidden_size, config.num_labels).to(device)
			self.num_labels[task] = config.num_labels
		self.bert = self.bert.from_pretrained('bert-base-uncased')
		self.init_weights()

	def forward(
		self,
		task,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
	):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)

		sequence_output = outputs[1]
		sequence_output = (self.dropouts[task])(sequence_output)
		logits = (self.classifiers[task])(sequence_output)

		outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
		if labels is not None:
			if self.num_labels == 1:
				#  We are doing regression
				loss_fct = MSELoss()
				loss = loss_fct(logits.view(-1), labels.view(-1))
			else:
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels[task]), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs