# Example usage:
# python predict.py --ckpt_path manuscript/saved_models/transfactor/transfactor_0.ckpt --out_dir results/test_predictions --subsample 100

import os
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from dataloader import ProteinDataset
from transfactor_model import TransFactor
from baseline_model import CNNLSTMPredictor


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--model_type', type=str, default='esm')
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--data_path', type=str, default='data/all_with_candidates.pickle')
parser.add_argument('--out_dir', type=str, default='results/predictions')
parser.add_argument('--subsample', type=int, default=None)
args = parser.parse_args()

if args.device is None:
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
	device = args.device

model_name = os.path.basename(args.ckpt_path).replace('.ckpt', '')
if args.model_type == 'esm':
	model = TransFactor.load_from_checkpoint(args.ckpt_path)
elif args.model_type == 'cnn_lstm':
	model = CNNLSTMPredictor.load_from_checkpoint(args.ckpt_path)

model.eval()
model = model.to(device)

df = pd.read_pickle(args.data_path)
# Replace candidate with 0.5 to make column numeric
df[model.data_config['label_col']] = df[model.data_config['label_col']].replace('candidate', 0.5).astype(float)

if args.subsample is not None:
	df = df.sample(args.subsample).copy()

data_config = model.data_config.copy()
dataset = ProteinDataset(df=df, config=data_config, split=None, tokenizer=model.tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

y_hats = []
with torch.no_grad():
	for batch in tqdm(dataloader):
		batch = [item.to(device) for item in batch]
		seq, attention_mask, y = batch
		y_hat_logits = model(seq, attention_mask)
		y_hat = model.sigmoid(y_hat_logits)
		y_hats.append(y_hat.detach().cpu())
	y_hat = torch.cat(y_hats)

df[model_name] = y_hat

os.makedirs(args.out_dir, exist_ok=True)
df.to_pickle(os.path.join(f'{args.out_dir}', f'{model_name}.pickle.zip'))
print(f'Saved predictions to {os.path.join(f"{args.out_dir}", f"{model_name}.pickle.zip")}')
