import numpy as np
import torch
import pandas as pd


class ProteinDataset(torch.utils.data.Dataset):
	def __init__(self, config, split='train', tokenizer=None, df=None):
		super().__init__()
		if df is None:
			data = pd.read_pickle(config['path'])
		else:
			data = df

		if split is not None:
			data = data[data[config['split_col']] == split].copy()

		data = data[~data['seq'].isna()]
		if tokenizer is not None:
			if data['seq'].str.len().max() > config['max_seq_len']:
				print(f"Some sequences are longer than {config['max_seq_len']}. They will be truncated. "
					  f"If this is not intended, please adjust the `max_seq_len` in the config.")
			seq_data = tokenizer(data['seq'].to_list(), padding=True,
								 truncation=True, add_special_tokens=True, max_length=config['max_seq_len'])
			data['seq_data'] = seq_data['input_ids']
			data['attention_mask'] = seq_data['attention_mask']
		else:
			unique_aa = ['-'] + sorted(set(''.join(data['seq'].to_list())))  # padding token + unique amino acids
			self.aa2label = dict(zip(unique_aa, range(len(unique_aa))))
			data['seq_data'] = data['seq'].apply(lambda x: np.array([self.aa2label[c] for c in [*x]]))
			data['seq_data'] = data['seq_data'].apply(lambda x: np.pad(x, (0, max(0, config['max_seq_len'] - x.shape[0])), 'constant', constant_values=0)[:config['max_seq_len']].T)
			data['attention_mask'] = data['seq_data'].apply(lambda x: np.pad(x, (0, max(0, config['max_seq_len'] - x.shape[0])), 'constant', constant_values=0)[:config['max_seq_len']].T)
			data['attention_mask'] = data['seq_data'].apply(lambda x: np.where(x == 0, 0, 1))

		if config['sample_size'] is not None and data.shape[0] > config['sample_size']:
			data = data.sample(config['sample_size'], random_state=42).reset_index(drop=True)

		self.seq = torch.tensor(np.stack(data['seq_data'].values))
		self.attention_mask = torch.tensor(np.stack(data['attention_mask'].values))
		self.label = torch.tensor(data[config['label_col']].values)

	def __len__(self):
		return len(self.label)

	def __getitem__(self, idx):
		return self.seq[idx], self.attention_mask[idx], self.label[idx]
