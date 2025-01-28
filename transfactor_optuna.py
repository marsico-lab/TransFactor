import comet_ml
import optuna
import argparse
import randomname
import os
import torch
from importlib import import_module
from time import strftime

from transfactor_train import train
from utils import convert_str_to_bool


parser = argparse.ArgumentParser()

# ESM configs
parser.add_argument('--config', type=str, default='configs.optuna_transfactor', help='Path to optuna config file')
parser.add_argument('--pretrained', type=str, default=None,
					choices=['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D', 'False'],
					help='Name of pretrained model as str, None to let Optuna propose, False to not use pretrained model')
parser.add_argument('--esm_config', type=str, default='False',
					help='True to manually configure model, False to use pretrained model')

# Head configs
parser.add_argument('--head_config_dropout', type=float, default=0,
					help='Dropout probability for head, None for Optuna to propose')
parser.add_argument('--head_config_hidden_neurons', type=int, default=0,
					help='Number of neurons per hidden layer in head, None for Optuna to propose')
parser.add_argument('--head_config_num_hidden_layers', type=int, default=0,
					help='Number of hidden layers in head, None for Optuna to propose')
parser.add_argument('--head_config_pool_type', type=str, default='mean',
					choices=['start_token', 'mean', 'max'],
					help='Pooling type from [start_token, mean, max], None for Optuna to propose')

# Lora configs
parser.add_argument('--lora_config', type=str, default='True', help='True to use LoRA, False to not use LoRA')
parser.add_argument('--lora_config_r', type=int, default=None,
					help='Number of hops, None for Optuna to propose')
parser.add_argument('--lora_config_lora_alpha', type=int, default=None,
					help='Alpha for Lora, None for Optuna to propose')
parser.add_argument('--lora_config_lora_dropout', type=float, default=None,
					help='Dropout probability for Lora, None for Optuna to propose')

# Training configs
parser.add_argument('--train_config_lr', type=float, default=None, help='Learning rate, None for Optuna to propose')
parser.add_argument('--train_config_pos_weight', type=float, default=None, help='Loss weight for positive class, None for Optuna to propose')
parser.add_argument('--train_config_max_epochs', type=int, default=5000, help='Max epochs')
parser.add_argument('--train_config_early_stopping', type=int, default=25, help='Early stopping patience')
parser.add_argument('--train_config_unfreeze_at_epoch', type=int, default=25, help='Early stopping patience')

# Data configs
parser.add_argument('--data_config_path', type=str, default='data/training_data.pickle', help='Path to data')
parser.add_argument('--data_config_max_seq_len', type=int, default=1024, help='Max sequence length')
parser.add_argument('--data_config_sample_size', type=int, default=None, help='Downsample size, None to use all data')
parser.add_argument('--data_config_label_col', type=str, default='label', help='Column name for labels')
parser.add_argument('--data_config_split_col', type=str, default='group_split_0', help='Column name for split')

# Comment
parser.add_argument('--comment', type=str, default='', help='Comment for CometML')

args = parser.parse_args()

# convert String 'True'/'False' to Boolean True/False
args = convert_str_to_bool(args)

def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
	print('#' * 40 + ' NEW RUN ' + '#' * 40)
	print("Trial number: {}".format(trial.number))

	optuna_proposer = import_module(args.config, package=None)

	config = optuna_proposer.propose(args, trial)
	config['trial_number'] = trial.number

	trainer = train(config)
	best_model_score = trainer.checkpoint_callbacks[0].best_model_score

	return best_model_score


if __name__ == "__main__":
	start_time = strftime('%Y%m%d-%H%M%S')
	study_name = f'{start_time}_' + randomname.get_name()

	os.makedirs(os.path.dirname('optuna/'), exist_ok=True)

	study = optuna.create_study(storage=f'sqlite:///optuna/{study_name}.db', load_if_exists=True,
								sampler=optuna.samplers.TPESampler(seed=42),
								study_name=study_name, direction='maximize')

	timeout = 47.5 * 3600  # 47.5 hours
	study.optimize(lambda trial: objective(trial, args), n_trials=1000, timeout=timeout, n_jobs=1)

	print("Number of finished trials: {}".format(len(study.trials)))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: {}".format(trial.value))

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))
