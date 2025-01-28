from transformers import EsmModel, EsmConfig, EsmTokenizer
import pytorch_lightning as pl
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision, BinaryAUROC, BinaryMatthewsCorrCoef, \
	BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
from collections import defaultdict


class IdentityWithKwargs(nn.Identity):
	def __init__(self):
		super().__init__()

	def forward(self, inputs_embeds, *args, **kwargs):
		return inputs_embeds


class EsmFreezeUnfreeze(pl.callbacks.BaseFinetuning):
	def __init__(self, unfreeze_at_epoch=0):
		super().__init__()
		self._unfreeze_at_epoch = unfreeze_at_epoch

	def freeze_before_training(self, pl_module):
		self.freeze(pl_module.model)
		print('Model frozen')

	def finetune_function(self, pl_module, current_epoch, optimizer):
		if current_epoch == self._unfreeze_at_epoch:
			print('Unfreezing model at epoch ', current_epoch)
			self.unfreeze_and_add_param_group(
				 modules=pl_module.model,
				 optimizer=optimizer,
				 train_bn=True,
			)


class TransFactor(pl.LightningModule):
	def __init__(self,
				 pretrained: str = None,
				 esm_config: dict = None,
				 head_config: dict = None,
				 lora_config: dict = None,
				 train_config: dict = None,
				 data_config: dict = None,
				 comment: str = '',
				 batch_size: int = 2,
				 **kwargs):
		super().__init__()
		self.save_hyperparameters()

		self.train_dataset = None
		self.val_dataset = None
		self.test_dataset = None
		self.batch_size = batch_size

		assert (pretrained is None) ^ (esm_config is None), 'Please specify exactly one of pretrained or config'
		if pretrained is not None:
			self.model = EsmModel.from_pretrained(pretrained)
			self.tokenizer = EsmTokenizer.from_pretrained(pretrained)
			esm_config = self.model.config.to_dict()
		else:
			self.model = EsmModel(EsmConfig(**esm_config))
			self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')

		if lora_config is not None:
			lora_config['target_modules'] = [name.replace('base_model.model.', '')
											 for name, module in self.model.named_modules()
											 if isinstance(module, nn.Linear) and 'attention' in name]
			self.model = get_peft_model(self.model, LoraConfig(**lora_config))

		self.model.embeddings.token_dropout = False

		if head_config['pool_type'] == 'start_token':
			self.pooler = lambda x: x  # use pooler_output instead of last_hidden_state
		elif head_config['pool_type'] == 'mean':
			self.pooler = lambda x: torch.mean(x, dim=1)
		elif head_config['pool_type'] == 'max':
			self.pooler = lambda x: torch.max(x, dim=1)[0]
		else:
			raise ValueError(f'Invalid pool_type: {head_config["pool_type"]}')

		# If num_hidden_layers == 0, then just a linear layer
		hidden_neurons = [head_config['hidden_neurons']] * head_config['num_hidden_layers']
		hidden_neurons = [self.model.embeddings.word_embeddings.embedding_dim] + hidden_neurons + [1]
		layers = []
		for i in range(len(hidden_neurons) - 2):
			layers.append(nn.Linear(hidden_neurons[i], hidden_neurons[i + 1]))
			if head_config['batch_norm']:
				layers.append(nn.BatchNorm1d(hidden_neurons[i + 1]))

			if head_config['activation'] == 'relu':
				layers.append(nn.ReLU())
			elif head_config['activation'] == 'tanh':
				layers.append(nn.Tanh())
			elif head_config['activation'] == 'sigmoid':
				layers.append(nn.Sigmoid())
			elif head_config['activation'] == 'linear':
				pass
			else:
				raise ValueError(f'Invalid activation: {head_config["activation"]}')
			if head_config['dropout'] > 0:
				layers.append(nn.Dropout(head_config['dropout']))

		layers.append(nn.Linear(hidden_neurons[-2], hidden_neurons[-1]))

		self.head = nn.Sequential(*layers)
		self.sigmoid = nn.Sigmoid()  # not used in training for better stability, but needed for predict

		self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([train_config['pos_weight']]) if 'pos_weight' in train_config and train_config['pos_weight'] is not None else None)

		self.outputs = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}

		self.evaluation = nn.ModuleDict({'accuracy': BinaryAccuracy(), 'precision': BinaryPrecision(),
										 'recall': BinaryRecall(), 'f1': BinaryF1Score(), 'auc': BinaryAUROC(),
										 'aps': BinaryAveragePrecision(), 'mcc': BinaryMatthewsCorrCoef()})
		self.confusion_matrix = BinaryConfusionMatrix()

		self.pretrained = pretrained
		self.esm_config = esm_config
		self.head_config = head_config
		self.lora_config = lora_config
		self.train_config = train_config
		self.data_config = data_config

	def forward(self, x_original, attention_mask=None):
		x = self.model.embeddings.word_embeddings(x_original)
		x = self.model(inputs_embeds=x, attention_mask=attention_mask)  # Position ids not used, because of rotary positional embeddings

		if self.head_config['pool_type'] == 'start_token':
			x = x['pooler_output']
		else:
			x = x['last_hidden_state']
			x = self.pooler(x)
		x = self.head(x).squeeze(1)

		return x

	def predict(self, x, attention_mask=None):
		x = self.sigmoid(self(x, attention_mask))
		return x

	def set_train_dataset(self, train_dataset):
		self.train_dataset = train_dataset

	def train_dataloader(self):
		if self.train_dataset is None:
			raise ValueError('train_dataset is None, please first set it using set_train_dataset()')
		return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

	def set_val_dataset(self, val_dataset):
		self.val_dataset = val_dataset

	def val_dataloader(self):
		if self.val_dataset is None:
			raise ValueError('val_dataset is None, please first set it using set_val_dataset()')
		return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

	def set_test_dataset(self, test_dataset):
		self.test_dataset = test_dataset

	def test_dataloader(self):
		if self.test_dataset is None:
			raise ValueError('test_dataset is None, please first set it using set_train_dataset()')
		return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

	def on_train_start(self) -> None:
		self.logger.log_hyperparams({'num_params_backbone': sum(p.numel() for p in self.model.parameters()),
									 'num_params_head': sum(p.numel() for p in self.head.parameters()),
									 'batch_size': self.batch_size,})

	def basic_step(self, batch, batch_idx, mode):
		seq, attention_mask, y = batch
		y_hat_logits = self(seq, attention_mask)
		loss = self.loss(y_hat_logits, y)

		# log metrics
		y_hat = self.sigmoid(y_hat_logits)
		self.log(f'{mode}_loss', loss, on_step=False, on_epoch=True, logger=True)

		self.outputs[mode]['y_hat'].append(y_hat.detach())
		self.outputs[mode]['y'].append(y.detach())

		return loss

	def basic_epoch_end(self, mode):
		outputs = self.outputs[mode]
		y_hat = torch.cat(outputs['y_hat'])
		y = torch.cat(outputs['y'])
		for name, metric in self.evaluation.items():
			self.log(f'{mode}_{name}', metric(y_hat, y.int()), on_step=False, on_epoch=True, logger=True,
					 prog_bar=True if mode == 'val' else False)

		self.outputs[mode] = defaultdict(list)  # reset outputs to free memory

	def training_step(self, batch, batch_idx):
		return self.basic_step(batch, batch_idx, 'train')

	def on_train_epoch_end(self):
		self.basic_epoch_end('train')

	def validation_step(self, batch, batch_idx):
		return self.basic_step(batch, batch_idx, 'val')

	def on_validation_epoch_end(self):
		self.basic_epoch_end('val')

	def test_step(self, batch, batch_idx):
		return self.basic_step(batch, batch_idx, 'test')

	def on_test_epoch_end(self):
		self.basic_epoch_end('test')

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.train_config['lr'])
