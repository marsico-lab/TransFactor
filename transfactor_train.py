import comet_ml
import torch
import pytorch_lightning as pl
import randomname
import yaml
import argparse
import os
import platform
from pprint import pprint

from time import strftime
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner import Tuner

from dataloader import ProteinDataset
from transfactor_model import TransFactor, EsmFreezeUnfreeze
from utils import check_offline


torch.set_float32_matmul_precision('medium')


def train(config):
    start_time = strftime('%Y%m%d-%H%M%S')
    experiment_id = f'{start_time}_' + randomname.get_name()
    seed_everything(42)

    if isinstance(config, str):
        print(f'Loading config from {config}')
        config = yaml.safe_load(open(config, 'r'))
        pprint(config)
    elif isinstance(config, dict):
        print('Using config dict')
        pprint(config)
    else:
        raise ValueError('config must be either a path to a yaml file or a dict')

    if config['pretrained'] is not None and config['esm_config'] is not None:
        raise ValueError('Please specify exactly one of pretrained or esm, both are specified')
    elif config['pretrained'] is None and config['esm_config'] is None:
        raise ValueError('Please specify exactly one of pretrained or esm, none are specified')
    else:
        model = TransFactor(**config)

    train_dataset = ProteinDataset(config=config['data_config'], split='train', tokenizer=model.tokenizer)
    model.set_train_dataset(train_dataset)
    val_dataset = ProteinDataset(config=config['data_config'], split='val', tokenizer=model.tokenizer)
    model.set_val_dataset(val_dataset)
    test_dataset = ProteinDataset(config=config['data_config'], split='test', tokenizer=model.tokenizer)
    model.set_test_dataset(test_dataset)

    os.makedirs(f'logs/{experiment_id}', exist_ok=True)
    os.makedirs(f'saved_models/{experiment_id}', exist_ok=True)
    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=f'saved_models/{experiment_id}', monitor='val_auc', mode='max',
                                     filename='auc_{epoch}', save_last=False, save_top_k=1, verbose=False))
    callbacks.append(EarlyStopping(monitor='val_auc', patience=config['train_config']['early_stopping'],
                                   mode='max', verbose=True))
    callbacks.append(EsmFreezeUnfreeze(unfreeze_at_epoch=config['train_config']['unfreeze_at_epoch']))

    tuner = Tuner(pl.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu', num_sanity_val_steps=0))
    tuner.scale_batch_size(model, mode='power', init_val=2, max_trials=1 if platform.system() == 'Darwin' else 6)
    model.batch_size //= 2

    trainer = pl.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                         max_epochs=config['train_config']['max_epochs'], log_every_n_steps=1,
                         callbacks=callbacks, num_sanity_val_steps=0, max_time='01:23:50:00')

    trainer.fit(model)

    model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    trainer.test(model)

    return trainer
