# TransFactor
![architecture.png](architecture.png)
## Usage
### 1. Install
```git clone https://github.com/marsico-lab/TransFactor.git```

```conda env create -f environment.yml```

As creating an environment from a file fails on some devices, one can create an environment from scratch

```conda create -n transfactor python=3.11.8```
```conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 -c pytorch```
```pip install -r requirements.txt```

### 2. Run hyperparameter optimization with Optuna
```conda activate transfactor```

```python transfactor_optuna.py```

### 3. Predict using saved models
```conda activate transfactor```

```python predict.py --ckpt_path <path_to_model_checkpoint> --out_dir <out_dir>```

### 4. Train model on new data with full hyperparameter optimization
The data is expected to be in a pickled pandas DataFrame with the sequence in a column named `seq` and the label in a column named `label`.
```conda activate transfactor```
```python transfactor_optuna.py --data_config_path <path_to_dataset_pickle_file>```

## Reproduce results from manuscript
Please find the corresponding notebooks and scripts in `/manuscript`. To download the saved model checkpoints and alanine scan results, please download those here: 
https://1drv.ms/f/s!AhK2hrQjl9NtlIgf9ueMG5xV94JBUA?e=k1kHZF
