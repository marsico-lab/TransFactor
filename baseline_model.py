from transformers import EsmTokenizer
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision, BinaryAUROC, BinaryMatthewsCorrCoef, \
	BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
from collections import defaultdict


class CNNBlock(nn.Module):
	"""
	CNN models were trained with the following parameters:
	number of filters=128, kernel size=37,
	pooling size=4, dropout rate=0.2, LSTM output size=50, batch size=128.
	"""
	def __init__(self,
				 model_config: dict = None,
				 ** kwargs):
		super().__init__()

		self.cnn = nn.Conv1d(model_config['num_filters'], model_config['num_filters'], model_config['kernel_size'], stride=1, padding=model_config['kernel_size']//2)
		self.activation = nn.ReLU() if model_config['activation'] == 'relu' else nn.Identity()
		self.dropout = nn.Dropout(model_config['dropout'])
		self.max_pool = nn.MaxPool1d(model_config['max_pool_kernel_size'], stride=model_config['max_pool_kernel_size'])
		self.dropout = nn.Dropout(model_config['dropout'])

	def forward(self, x):
		# x needs to be of shape (batch_size, in_channels, seq_len)
		x = self.cnn(x)  # (batch_size, out_channels, seq_len)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.max_pool(x)
		x = self.dropout(x)
		return x


class CNNLSTMPredictor(pl.LightningModule):
	def __init__(self,
				 model_config: dict = None,
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

		# input_size, hidden_size, num_layers = 1, bias = True, batch_first = False, dropout = 0.0, bidirectional = False, proj_size = 0, device = None, dtype = None
		self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')

		self.embedding = nn.Embedding(num_embeddings=self.tokenizer.vocab_size, embedding_dim=model_config['num_filters'], padding_idx=self.tokenizer.pad_token_id)
		self.cnn = nn.Sequential(*[CNNBlock(model_config)] * model_config['num_blocks'])
		self.lstm = nn.LSTM(input_size=model_config['num_filters'], hidden_size=model_config['lstm_hidden_size'], batch_first=True, )

		self.dropout = nn.Dropout(model_config['dropout'])
		self.linear = nn.Linear(model_config['lstm_hidden_size'], 1)
		self.sigmoid = nn.Sigmoid()

		self.model_config = model_config
		self.model_config = model_config
		self.train_config = train_config
		self.data_config = data_config

		self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([train_config['pos_weight']]) if 'pos_weight' in train_config and train_config['pos_weight'] is not None else None)

		self.outputs = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}

		self.evaluation = nn.ModuleDict({'accuracy': BinaryAccuracy(), 'precision': BinaryPrecision(),
										 'recall': BinaryRecall(), 'f1': BinaryF1Score(), 'auc': BinaryAUROC(),
										 'aps': BinaryAveragePrecision(), 'mcc': BinaryMatthewsCorrCoef()})
		self.confusion_matrix = BinaryConfusionMatrix()

	def forward(self, x_original, attention_mask=None, output_weights=False):
		x = self.embedding(x_original)  # (batch_size, seq_len, in_channels)
		x = self.cnn(x.permute(0, 2, 1))  # (batch_size, out_channels, seq_len)
		x = x.permute(0, 2, 1)  # (batch_size, seq_len, out_channels)

		out, (h, c) = self.lstm(x)  # (batch_size, seq_len, hidden_size) for x, (num_layers, batch_size, hidden_size) for h and c
		x = h[-1]  # (batch_size, hidden_size) since using last layer
		x = self.dropout(x)  # (batch_size, hidden_size)
		x = self.linear(x).squeeze(1)  # (batch_size)

		return x

	def predict(self, x, attention_mask=None):
		x = self.sigmoid(self(x))
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
		self.logger.log_hyperparams({'num_params': sum(p.numel() for p in self.parameters()),
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
