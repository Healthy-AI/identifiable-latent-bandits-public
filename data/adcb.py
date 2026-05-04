import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.utils import to_one_hot
from utils import read_config

ADCB_CONFIG = "configs/data_configs/adcb_config.yaml"


def adcb_preprocess(x, minmax, add_ptmarry):
    """Preprocess for ADCB for bandit time."""
    ptmarry = x.pop('PTMARRY')
    ptmarry_cols = [f'PTMARRY_{i}' for i in range(5)]
    ptmarry = np.eye(5)[ptmarry] # Hardcode 5 classes
    df = pd.DataFrame([x]) 
    df[df.columns] = minmax.transform(df)
    if add_ptmarry:
        df[ptmarry_cols] = ptmarry
    return df.to_numpy()


def get_patient_data(df, patient_id, timesteps=(0, None)):
    return df.loc[df['RID'] == patient_id][timesteps[0]: timesteps[1]]


def read_acdc_csv(filepath, n_patients, n_timesteps, reward_sigma=0.25):
    """Processes ACDC🎸 medical dataset."""
    acdc_random = np.random.RandomState(112)
    config = read_config(ADCB_CONFIG)
    is_add_ptmarry = config.add_ptmarry
    z_cols = config.z_cols
    x_cols = config.x_cols
    ptmarry_cols = config.ptmarry_cols
    action_col = config.action_col
    mu_col = config.mu_col
    id_col = config.id_col
    potoutcome_cols = config.potoutcome_cols
    reward_nonoise_col = config.reward_nonoise_col
    noise_col = config.noise_col
    df = pd.read_csv(filepath)
    df[noise_col] = acdc_random.normal(0, reward_sigma, df[noise_col].to_numpy().shape)
    df = df.iloc[:(df[id_col].max()+1) * (n_timesteps + 20)]
    train_ids = range(0, n_patients)
    test_ids = range(n_patients, int(n_patients + 50))

    df.loc[df['APOE4'] == 2, 'APOE4'] = 1 # Class 2 is underrepresented
    #df.loc[df['PTRACCAT'] > 0, 'PTRACCAT'] = 1 # Classes 2-6 are heavily underrepresented
    #df.loc[df['PTETHCAT'] != 1, 'PTETHCAT'] = 0 # Classes 0 and 2 is heavily underrepresented
    minmax = MinMaxScaler()
    ptmarry = to_one_hot(df['PTMARRY'], 5)[0] # Hardcode 5 classes
    df.drop(['PTMARRY'], axis=1, inplace=True)
    df[x_cols] = minmax.fit_transform(df[x_cols])
    df[ptmarry_cols] = ptmarry
    parsed_patients = [get_patient_data(df, i, (0, n_timesteps)) for i in train_ids]
    parsed_val = [get_patient_data(df, i, (n_timesteps, n_timesteps+20)) for i in train_ids]
    parsed_test = [get_patient_data(df, i, (0, n_timesteps)) for i in test_ids]
    if is_add_ptmarry:
        x_cols += ptmarry_cols
    x = np.concatenate([dfi[x_cols].values for dfi in parsed_patients])
    x_val = np.concatenate([dfi[x_cols].values for dfi in parsed_val])
    x_test = np.concatenate([dfi[x_cols].values for dfi in parsed_test])
    y = np.concatenate([np.repeat(i, n_timesteps) for i in train_ids])
    y_val = np.concatenate([np.repeat(i, 20) for i in train_ids])
    s = np.concatenate([dfi[z_cols].values for dfi in parsed_patients])
    source_val = np.concatenate([dfi[z_cols].values for dfi in parsed_val])
    source_test = np.concatenate([dfi[z_cols].values for dfi in parsed_test])
    mu = pd.concat([dfi[[id_col] + mu_col].iloc[[0]] for dfi in parsed_patients])
    mu_test = pd.concat([dfi[[id_col] + mu_col].iloc[[0]] for dfi in parsed_test])
    treatment = np.concatenate([dfi[action_col].values.reshape(1, -1) for dfi in parsed_patients])
    treatment_val = np.concatenate([dfi[action_col].values.reshape(1, -1) for dfi in parsed_val])
    treatment_test = np.concatenate([dfi[action_col].values.reshape(1, -1) for dfi in parsed_test])
    theta_zed = np.concatenate([dfi[reward_nonoise_col].values.reshape(1, -1) for dfi in parsed_patients])
    theta_zed_val = np.concatenate([dfi[reward_nonoise_col].values.reshape(1, -1) for dfi in parsed_val])
    theta_zed_test = np.concatenate([dfi[reward_nonoise_col].values.reshape(1, -1) for dfi in parsed_test])
    potential_outcomes = np.concatenate([np.expand_dims(dfi[potoutcome_cols].values, 0) for dfi in parsed_patients])
    potential_outcomes_val = np.concatenate([np.expand_dims(dfi[potoutcome_cols].values, 0) for dfi in parsed_val])
    potential_outcomes_test = np.concatenate([np.expand_dims(dfi[potoutcome_cols].values, 0) for dfi in parsed_test])
    noise = np.concatenate([dfi[noise_col].values.reshape(1, -1) for dfi in parsed_patients])
    noise_val = np.concatenate([dfi[noise_col].values.reshape(1, -1) for dfi in parsed_val])
    noise_test = np.concatenate([dfi[noise_col].values.reshape(1, -1) for dfi in parsed_test])
    treatment_data = {'theta': None,
                      'treatment': treatment,
                      'reward': theta_zed + noise,
                      'theta_zed': theta_zed,
                      'potential_outcomes': potential_outcomes,
                      'noise': noise,
                      }
    treatment_data_val = {'theta': None,
                      'treatment': treatment_val,
                      'reward': theta_zed_val + noise_val,
                      'theta_zed': theta_zed_val,
                      'potential_outcomes': potential_outcomes_val,
                      'noise': noise_val,
                      }
    treatment_data_test = {'theta': None,
                           'treatment': treatment_test,
                           'reward': theta_zed_test + noise_test,
                           'theta_zed': theta_zed_test,
                           'potential_outcomes': potential_outcomes_test,
                           'noise': noise_test,
                           }
    return pack_data_dict(x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test)


def pack_data_dict(x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test, **kwargs):
    data_dict = {
        'x': x,
        'y': y,
        's': s,
        'x_val': x_val,
        'y_val': y_val,
        'source_val': source_val,
        'x_test': x_test,
        'source_test': source_test,
        'mu': mu,
        'mu_test': mu_test,
        'minmax': minmax,
        'treatment_data': treatment_data,
        'treatment_data_val': treatment_data_val,
        'treatment_data_test': treatment_data_test,
        }
    return data_dict


def unpack_data_dict(data_dict):
    x = data_dict['x']
    y = data_dict['y']
    s = data_dict['s']
    x_val = data_dict.get('x_val')
    y_val = data_dict.get('y_val')
    source_val = data_dict.get('source_val')
    x_test = data_dict['x_test']
    source_test = data_dict['source_test']
    mu = data_dict['mu']
    mu_test = data_dict['mu_test']
    minmax = data_dict['minmax']
    treatment_data = data_dict['treatment_data']
    treatment_data_val = data_dict.get('treatment_data_val')
    treatment_data_test = data_dict['treatment_data_test']
    return x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test
