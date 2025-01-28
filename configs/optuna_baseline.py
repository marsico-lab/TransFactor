import optuna


def propose(args, trial: optuna.trial.Trial) -> dict:

	config = {
		'model_config': {
			'num_filters': 2**trial.suggest_int('model_config/num_filters', 4, 10) if args.model_config_num_filters is None else args.model_config_num_filters,
			'kernel_size': 2**trial.suggest_int('model_config/kernel_size', 1, 6)+1 if args.model_config_kernel_size is None else args.model_config_kernel_size,
			'max_pool_kernel_size': trial.suggest_int('model_config/max_pool_kernel_size', 1, 2) if args.model_config_max_pool_kernel_size is None else args.model_config_max_pool_kernel_size,
			'dropout': trial.suggest_float('model_config/dropout', 0.0, 0.25, step=0.025) if args.model_config_dropout is None else args.model_config_dropout,
			'num_blocks': trial.suggest_int('model_config/num_blocks', 1, 6) if args.model_config_num_blocks is None else args.model_config_num_blocks,
			'activation': 'relu',
			'lstm_hidden_size': 2**trial.suggest_int('model_config/lstm_hidden_size', 4, 10) if args.model_config_lstm_hidden_size is None else args.model_config_lstm_hidden_size,
		},
		'train_config': {
			'lr': 10**(-trial.suggest_float('train_config/lr', 3.5, 6, step=0.25)) if args.train_config_lr is None else args.train_config_lr,
			'pos_weight': 10**(trial.suggest_float('train_config/pos_weight', 0.0, 1.0, step=0.5)) if args.train_config_pos_weight is None else args.train_config_pos_weight,
			'max_epochs': args.train_config_max_epochs,
			'early_stopping': args.train_config_early_stopping,
			'unfreeze_at_epoch': args.train_config_unfreeze_at_epoch,
		},
		'data_config': {
			'max_seq_len': args.data_config_max_seq_len,
			'sample_size': args.data_config_sample_size,
			'label_col': args.data_config_label_col,
			'split_col': args.data_config_split_col,
		},
		'comment': args.comment,
	}


	return config