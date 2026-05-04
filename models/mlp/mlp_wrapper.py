import os

import numpy as np

from models.mlp.mlp_core import RandomizedWindowDataset, MLP
from models.preprocessing import pca
from utils import read_config, save_pickle, join_list_of_dict

MLP_CONFIG_PATH = "configs/model_configs/mlp-config.yaml"


def mlp_wrapper(sensor, sensor_val, treatment_data, treatment_data_val, ckpt_dir,random_seed=42, is_csv=False, **kwargs):
    os.makedirs(os.path.join(ckpt_dir), exist_ok=True)
    model_config = read_config(MLP_CONFIG_PATH)
    params = join_list_of_dict(model_config.model_params)
    data_dim = sensor.shape[1]
    n_treatments = len(np.unique(treatment_data['treatment']))

    # Preprocessing ------------------------------------------------
    if not is_csv and params['pca']:
        sensor, pca_parm = pca(sensor, num_comp=data_dim)
        save_pickle(os.path.join(ckpt_dir, 'pca_params.bin'), pca_parm)

    # Dataset & Training -------------------------------------------
    window = params['window']
    data = RandomizedWindowDataset(sensor, treatment_data["reward"], treatment_data['treatment'], window=window)
    data_val = RandomizedWindowDataset(sensor_val, treatment_data_val["reward"], treatment_data_val['treatment'], window=window)
    params['optimizer_params']['weight_decay'] = float(params['optimizer_params']['weight_decay'])
    model = MLP(input_dim=data_dim,
                output_dim=n_treatments,
                hidden_list=params['hidden_list'],
                average_means= True,
                lr=float(params['lr']),
                epoch_num=params['epoch_num'],
                batch_size=params['batch_size'],
                criterion_name=params['criterion_name'],
                criterion_params=params['criterion_params'],
                optimizer_name=params['optimizer_name'],
                optimizer_params=params['optimizer_params'],
                dropout_rate=params['dropout_rate'],
                batch_norm=params['batch_norm'],
                hidden_activation_name=params['hidden_activation_name'],
                output_activation_name=params['output_activation_name'],
                random_state=random_seed,
                device=None,
                use_compile=False,
                compile_mode='default',
                verbose=1,
                n=kwargs.get('n'),
                l=kwargs.get('l'),
                )
    history = model.fit(train_set=data, val_set=data_val)
    model.save(ckpt_dir)
    return model, data, data_val
