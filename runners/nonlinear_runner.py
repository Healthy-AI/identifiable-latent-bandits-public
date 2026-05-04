import os

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.metrics import mean_squared_error, r2_score

from data.ilb import ILBDataConfig, pack_data_dict, unpack_data_dict
from metrics.mcc import mean_corr_coef
from models.mlp.mlp_reward import MLPModel, train_reward_model, load_reward_model, estimate_rewards,estimate_potential_outcomes
from models.sequence_regressor import sequence_model_wrapper, load_model, prepare_data
from models.vae import vae_wrapper, VAE
from models.tcl.tcl_wrapper_gpu import TCL_wrapper, inference_tcl
from runners.plot_utils import plot_loss_and_r2
from utils import save_pickle, load_pickle, write_json_file


from typing import NamedTuple
import matplotlib.pyplot as plt


class TCL(NamedTuple):

    num_comp: int
    num_segment: int
    num_segmentdata: int

    def n_data(self):
        return int(self.num_segment * self.num_segmentdata)

    @classmethod
    def init_from_args(cls, data_dim, n_segments, n_obs_per_seg):
        return cls(num_comp=data_dim, num_segment=n_segments, num_segmentdata=n_obs_per_seg)


def visualize_data(source_or_senor, FLAGS, name='source', filepath=None):
    df = pd.DataFrame(source_or_senor)
    filepath = os.path.join('.', f'{name}.png') if filepath is None else filepath
    axes = df.plot(kind='line', title=name.capitalize(), subplots=True)
    for ax in axes:
        for line in np.arange(0, FLAGS.n_data(), FLAGS.num_segmentdata):
            ax.axvline(x=line+FLAGS.num_segmentdata, linestyle='dotted', color='black')
    plt.savefig(filepath, format='pdf', dpi=300)
    plt.clf()


def get_mu_hat(latent, n, n_segments):
    """n: len_treatment, n_segments: n_patients."""
    idx = np.arange(0, n*n_segments, n)
    return np.array([np.mean(np.take(latent, axis=0, indices=range(i, i+n)), 0) for i in idx])


def run_ilb_exp(args, config):
    """run TCL simulations"""
    stepDict = {1: [int(5e3), int(5e3)],
                2: [int(1e4), int(1e4)],
                3: [int(1e4), int(1e4)],
                4: [int(1e4), int(1e4)],
                5: [int(1e4), int(1e4)]}

    data_dim = config.data_dim
    num_comp = data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    data_seed = config.data_seed
    latent_sigma = config.latent_sigma
    latent_noise_sigma = config.latent_noise_sigma
    reward_sigma = config.reward_sigma
    nSims = args.nSims
    data_config = ILBDataConfig(config)

    results_accuracy = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    for l in n_layers:
        for n in n_obs_per_seg:
            # Generate mixing data and treatment data
            datadict = data_config.generate_data(n_time_steps=n, n_layers=l)
            x, y, s, x_val, y_val, source_val, mu, mixing, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(datadict)
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            os.makedirs(data_folder, exist_ok=True)
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            save_pickle(datadict_path, pack_data_dict(**locals()))
            save_pickle(os.path.join(data_folder, 'NonLinear.bin'), mixing) # Save for ease of access
            save_pickle(os.path.join(data_folder, 'theta_true.bin'), treatment_data['theta']) # Save for ease of access
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                res_TCL = TCL_wrapper(sensor=x.T, label=y, random_seed=seed,
                                      list_hidden_nodes=[num_comp * 2] * (l - 1) + [num_comp],
                                      max_steps=stepDict[l][0] * 2, max_steps_init=stepDict[l][1],
                                      ckpt_dir=ckpt_folder, test=False, n=n, l=l)
                # Estimate thetas
                latent = np.asarray(res_TCL[0].T)
                latent_ica = np.asarray(res_TCL[1].T)
                mu_hat = get_mu_hat(latent=latent, n=n, n_segments=n_segments)
                mu_hat_ica = get_mu_hat(latent=latent_ica, n=n, n_segments=n_segments)
                theta_hat_path = os.path.join(ckpt_folder, 'theta_model.bin')
                theta_hat_ica_path = os.path.join(ckpt_folder, 'theta_model_ica.bin')
                history = train_reward_model(mu_hat, treatment_data, treatment_data_val, theta_hat_path)
                history_ica = train_reward_model(mu_hat_ica, treatment_data, treatment_data_val, theta_hat_ica_path)
                write_json_file(history, os.path.join(ckpt_folder, 'theta_history.json'))
                write_json_file(history_ica, os.path.join(ckpt_folder, 'theta_history_ica.json'))
                # Accuracy results
                results_accuracy[l][n].append(res_TCL[2])
                print(f"Accuracy for n={n}, l={l} is: {res_TCL[2]} ")
    return {'data_dim': data_dim,
            'n_patients': n_segments,
            'n_treatment': n_treatment,
            'Train Results': {
                'Accuracy': results_accuracy,
                },
            }


def run_vae_exp(args, config):
    """run TCL simulations"""
    data_dim = config.data_dim
    num_comp = data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    data_seed = config.data_seed
    latent_sigma = config.latent_sigma
    latent_noise_sigma = config.latent_noise_sigma
    reward_sigma = config.reward_sigma
    nSims = args.nSims
    num_epochs = config.num_epochs
    data_config = ILBDataConfig(config)

    for l in n_layers:
        for n in n_obs_per_seg:
            # Generate mixing data and treatment data
            datadict = data_config.generate_data(n_time_steps=n, n_layers=l)
            x, y, s, x_val, y_val, source_val, mu, mixing, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(datadict)
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            os.makedirs(data_folder, exist_ok=True)
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            save_pickle(datadict_path, pack_data_dict(**locals()))
            save_pickle(os.path.join(data_folder, 'NonLinear.bin'), mixing) # Save for ease of access
            save_pickle(os.path.join(data_folder, 'theta_true.bin'), treatment_data['theta']) # Save for ease of access
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                os.makedirs(ckpt_folder, exist_ok=True)
                latent = vae_wrapper(sensor=x.T, sensor_test=x_test, latent_dim=data_dim,
                                    list_hidden=[num_comp * 2] * (l - 1) + [num_comp],
                                    lr=1e-3,
                                    epoch_num=num_epochs, #stepDict[l][0] * 2,
                                    batch_size=32,
                                    optimizer_name='adam',
                                    verbose=1,
                                    optimizer_params={'weight_decay': 1e-3},
                                    beta=1.0,
                                    capacity=0.0,
                                    hidden_activation_name='relu',
                                    output_activation_name='sigmoid',
                                    batch_norm=False,
                                    dropout_rate=0.0,
                                    is_csv=False,
                                    seed=seed,
                                    ckpt_dir=ckpt_folder,
                                    n=n, l=l)
                ica = FastICA(random_state=seed).fit(latent)
                latent_ica = ica.transform(latent)
                save_pickle(os.path.join(ckpt_folder, 'fast_ica.bin'), ica)

                # Estimate thetas
                mu_hat = get_mu_hat(latent=latent, n=n, n_segments=n_segments)
                mu_hat_ica = get_mu_hat(latent=latent_ica, n=n, n_segments=n_segments)
                theta_hat_path = os.path.join(ckpt_folder, 'theta_model.bin')
                theta_hat_ica_path = os.path.join(ckpt_folder, 'theta_model_ica.bin')
                history = train_reward_model(mu_hat, treatment_data, treatment_data_val, theta_hat_path)
                history_ica = train_reward_model(mu_hat_ica, treatment_data, treatment_data_val, theta_hat_ica_path)
                write_json_file(history, os.path.join(ckpt_folder, 'theta_history.json'))
                write_json_file(history_ica, os.path.join(ckpt_folder, 'theta_history_ica.json'))
    return {'data_dim': data_dim,
            'n_patients': n_segments,
            'n_treatment': n_treatment,
            }


def run_LSTM_exp(args, config):
    """run ILB simulations"""
    data_dim = config.data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    nSims = args.nSims
    num_epochs = config.num_epochs
    data_config = ILBDataConfig(config)

    for l in n_layers:
        for n in n_obs_per_seg:
            # Generate mixing data and treatment data
            datadict = data_config.generate_data(n_time_steps=n, n_layers=l)
            x, y, s, x_val, y_val, source_val, mu, mixing, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(datadict)
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            os.makedirs(data_folder, exist_ok=True)
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            save_pickle(datadict_path, pack_data_dict(**locals()))
            save_pickle(os.path.join(data_folder, 'NonLinear.bin'), mixing) # Save for ease of access
            save_pickle(os.path.join(data_folder, 'theta_true.bin'), treatment_data['theta']) # Save for ease of access
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                os.makedirs(ckpt_folder, exist_ok=True)
                history = sequence_model_wrapper(sensor=x,
                                                treatment_data=treatment_data,
                                                n_layers=l,
                                                n_obs_per_seg=n,
                                                num_epochs=num_epochs,
                                                ckpt_dir=ckpt_folder,
                                                random_seed=seed,
                                                val=(x_test, treatment_data_test),
                                                n=n,l=l)
                # Call the function to plot loss and validation R2
                plot_loss_and_r2(history, ckpt_folder)
                write_json_file(history, os.path.join(ckpt_folder, 'history.json'))
    Results = {
        'data_dim': data_dim,
        'n_patients': n_segments,
        'n_treatment': n_treatment,
        'Train Results': {},
        }
    return Results


def estimate_actions(model, mu_hat, len_treatment):
    n_patients = mu_hat.shape[0]
    actions = np.zeros(shape=(n_patients, len_treatment), dtype=int)
    for p in range(n_patients):
        for t in range(len_treatment):
            po = model.get_potential_outcomes(mu_hat[p]).cpu().numpy().flatten()
            actions[p, t] = np.argmax(po)
    return actions


def evaluate_actions(model, mu_hat, treatment_data):
    potentials = treatment_data['potential_outcomes'][:, 0, :]
    arm_order = np.argsort(potentials, axis=-1)
    actions = estimate_actions(model, mu_hat, len_treatment=treatment_data['treatment'].shape[1])
    action_order = np.zeros_like(actions)
    n_patients, len_treatment = actions.shape
    for p in range(n_patients):
        order = arm_order[p][::-1]
        patient_actions = actions[p]
        for t in range(len_treatment):
            action_order[p, t] = np.where(order == patient_actions[t])[0][0] + 1 # Rank of the action
    unique, counts = np.unique(action_order.flatten(), return_counts=True)
    return dict(zip(unique.tolist(), (counts / (n_patients * len_treatment) * 100).tolist())) # Percentage of actions


def evaluate_latent_state(res_TCL, mu, mu_hat, mu_hat_ica, s, quick_eval=False):
    if quick_eval:
        return {'mcc_mu_ica': mean_corr_coef(mu_hat_ica, mu),
                'mcc_mu_no_ica': mean_corr_coef(mu_hat, mu)}
    mcc_no_ica = mean_corr_coef(res_TCL[0].T, s)
    mcc_ica = mean_corr_coef(res_TCL[1].T, s)
    mcc_mu_no_ica = mean_corr_coef(mu_hat, mu)
    mcc_mu_ica = mean_corr_coef(mu_hat_ica, mu)
    return {'mcc_ica': mcc_ica,
            'mcc_no_ica': mcc_no_ica,
            'mcc_mu_ica': mcc_mu_ica,
            'mcc_mu_no_ica': mcc_mu_no_ica,
            }


def evaluate_reward(treatment_data, reward_hat, po_hat):
    # reward w/o noise
    mse_thetazed = mean_squared_error(treatment_data['theta_zed'].flatten(), reward_hat.flatten())
    r2_thetazed = r2_score(treatment_data['theta_zed'].flatten(), reward_hat.flatten())
    # reward w/ noise
    mse_reward = mean_squared_error(treatment_data['reward'].flatten(), reward_hat.flatten())
    r2_reward = r2_score(treatment_data['reward'].flatten(), reward_hat.flatten())
    # potential outcomes w/o noise
    mse_po = mean_squared_error(treatment_data['potential_outcomes'].flatten(), po_hat.flatten())
    r2_po = r2_score(treatment_data['potential_outcomes'].flatten(), po_hat.flatten())
    # potential outcomes w/ noise
    po_noise = treatment_data['potential_outcomes'] + np.tile(np.expand_dims(treatment_data['noise'], -1), [1, 1, po_hat.shape[-1]])
    mse_po_noise = mean_squared_error(po_noise.flatten(), po_hat.flatten())
    r2_po_noise = r2_score(po_noise.flatten(), po_hat.flatten())
    return {'MSE-thetazed': mse_thetazed,
            'R2-thetazed': r2_thetazed,
            'MSE-Reward': mse_reward,
            'R2-Reward': r2_reward,
            'MSE-Potential_Outcomes': mse_po,
            'R2-Potential_Outcomes': r2_po,
            'MSE-PO_noise': mse_po_noise,
            'R2_PO_noise': r2_po_noise,
            }


def df_report():
    model_cols = ['data_dim', 'n_patients', 'n_treatment', 'n_layers', 'len_treatment', 'seed']
    reward_cols = [
        'MSE-thetazed',
        'R2-thetazed',
        'MSE-Reward',
        'R2-Reward',
        'MSE-Potential_Outcomes',
        'R2-Potential_Outcomes',
        'MSE-PO_noise',
        'R2_PO_noise',
        ]
    reward_cols_ica = [colname + '-ica' for colname in reward_cols]
    latent_cols = [
        'mcc_ica',
        'mcc_no_ica',
        'mcc_mu_ica',
        'mcc_mu_no_ica',
        ]
    action_cols = [
        'best_action',
        'second_best_action',
        'worst_action',
        ]
    action_cols_ica = [colname + '-ica' for colname in action_cols]
    return pd.DataFrame(columns=model_cols + reward_cols + reward_cols_ica + latent_cols + action_cols + action_cols_ica)


def append_report(df, config, unique, reward_results, reward_results_ica, latent_results, action_percent, action_percent_ica):
    model_cols = {'data_dim': config.data_dim,
                  'n_patients': config.n_segments,
                  'n_treatment': config.n_treatment,
                  'n_layers': unique['l'],
                  'len_treatment': unique['n'],
                  'seed': unique['seed'],
                  }
    actions = {'best_action': action_percent[1],
               'second_best_action': action_percent.get(2, 0),
               'worst_action': action_percent[max(action_percent.keys())],
               }
    actions_ica = {'best_action-ica': action_percent_ica[1],
               'second_best_action-ica': action_percent_ica.get(2, 0),
               'worst_action-ica': action_percent_ica[max(action_percent_ica.keys())],
               }
    reward_results_ica = {k + '-ica': v for k, v in reward_results_ica.items()}
    row = {**model_cols, **latent_results, **reward_results, **reward_results_ica, **actions, **actions_ica}
    df.loc[len(df)] = row


def eval_ilb_exp(args, config):
    """run TCL simulations"""
    stepDict = {1: [int(5e3), int(5e3)],
                2: [int(1e4), int(1e4)],
                3: [int(1e4), int(1e4)],
                4: [int(1e4), int(1e4)],
                5: [int(1e4), int(1e4)]}

    data_dim = config.data_dim
    num_comp = data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    nSims = args.nSims
    quick_eval = config.quick_eval

    df = df_report()
    df_test = df_report()
    results_accuracy = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_latent = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_reward = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_reward_ica = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action_ica = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    results_latent_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_reward_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_reward_ica_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action_ica_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    def _image_name(is_source, n, l, args):
        if is_source:
            img_name = f'source_{n}segment_{l}layers.pdf'
            img_name = 'source', os.path.join(args.checkpoints, img_name)
        else:
            img_name = f'sensor_{n}segment_{l}layers.pdf'
            img_name = 'sensor', os.path.join(args.checkpoints, img_name)
        return img_name

    for l in n_layers:
        for n in n_obs_per_seg:
            # generate mixing data
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            x, y, s, x_val, y_val, source_val, mu, mixing, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(load_pickle(datadict_path))
            #np.testing.assert_array_almost_equal(treatment_data['theta'], treatment_data_test['theta'])
            tcl = TCL.init_from_args(data_dim, n_segments, n)
            visualize_data(x, tcl, *_image_name(False, n, l, args))
            visualize_data(s, tcl, *_image_name(True, n, l, args))
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Testing exp with L={} and n={}; seed={}'.format(l, n, seed))
                # Run Inference for train data
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                res_TCL = TCL_wrapper(sensor=x.T, label=y, random_seed=seed,
                                      list_hidden_nodes=[num_comp * 2] * (l - 1) + [num_comp],
                                      max_steps=stepDict[l][0] * 2, max_steps_init=stepDict[l][1],
                                      ckpt_dir=ckpt_folder, test=True)
                # Post Processing
                results_accuracy[l][n].append(res_TCL[2])
                latent = np.asarray(res_TCL[0].T) # (n_segments * n_patients, data_dim)
                latent_ica = np.asarray(res_TCL[1].T) # (n_segments * n_patients, data_dim)
                mu_hat = get_mu_hat(latent, n=n, n_segments=n_segments)
                mu_hat_ica = get_mu_hat(latent_ica, n=n, n_segments=n_segments)

                # estimate rewards
                model = load_reward_model(data_dim, n_treatment, os.path.join(ckpt_folder, 'theta_model.bin'))
                model_ica = load_reward_model(data_dim, n_treatment, os.path.join(ckpt_folder, 'theta_model_ica.bin'))
                reward_hat = estimate_rewards(model, mu_hat, treatment_data['treatment'])
                reward_hat_ica = estimate_rewards(model_ica, mu_hat_ica, treatment_data['treatment'])
                po_hat = estimate_potential_outcomes(model, mu_hat, treatment_data['treatment'])
                po_hat_ica = estimate_potential_outcomes(model_ica, mu_hat_ica, treatment_data['treatment'])

                # train results
                latent_results = evaluate_latent_state(res_TCL, mu, mu_hat, mu_hat_ica, s, quick_eval=quick_eval)
                reward_results = evaluate_reward(treatment_data, reward_hat, po_hat)
                reward_results_ica = evaluate_reward(treatment_data, reward_hat_ica, po_hat_ica)
                action_percent = evaluate_actions(model, mu_hat, treatment_data)
                action_percent_ica = evaluate_actions(model_ica, mu_hat_ica, treatment_data)

                # Reporting
                results_accuracy[l][n].append(res_TCL[2])
                results_latent[l][n].append(latent_results)
                results_reward[l][n].append(reward_results)
                results_reward_ica[l][n].append(reward_results_ica)
                results_action[l][n].append(action_percent)
                results_action_ica[l][n].append(action_percent_ica)
                print(f"Results Train for n={n}, l={l}, seed={seed} : \n----------")
                print(f"Accuracy is: {res_TCL[2]}")
                if not quick_eval:
                    print(f"ILB Data Recovery (no ICA): {latent_results['mcc_no_ica']} \t  with ICA {latent_results['mcc_ica']}")
                print(f"Patient Means Recovery (no ICA): {latent_results['mcc_mu_no_ica']} \t with ICA: {latent_results['mcc_mu_ica']}")
                print(f"Reward Recovery MSE (w/o noise): {reward_results['MSE-thetazed']} \t R2: {reward_results['R2-thetazed']}")
                print(f"Reward Recovery MSE (w/o noise) ICA: {reward_results_ica['MSE-thetazed']} \t R2: {reward_results_ica['R2-thetazed']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent}")
                print(f"Percentages of chosen actions ICA (best to worst): {action_percent_ica}")
                print('----------')

                # Run inference for test data
                ica = load_pickle(os.path.join(ckpt_folder, 'fast_ica.bin'))
                z_hat = inference_tcl(x_test.T, np.zeros(x_test.shape[0]), [num_comp * 2] * (l - 1) + [num_comp], ckpt_folder, is_csv=False)
                z_hat_ica = ica.transform(z_hat)
                mu_hat_test = get_mu_hat(z_hat, n=n, n_segments=50)
                mu_hat_ica_test = get_mu_hat(z_hat_ica, n=n, n_segments=50)

                # estimate rewards
                reward_hat_test = estimate_rewards(model, mu_hat_test, treatment_data_test['treatment'])
                reward_hat_ica_test = estimate_rewards(model_ica, mu_hat_ica_test, treatment_data_test['treatment'])
                po_hat_test = estimate_potential_outcomes(model, mu_hat_test, treatment_data_test['treatment'])
                po_hat_ica_test = estimate_potential_outcomes(model_ica, mu_hat_ica_test, treatment_data_test['treatment'])

                # test results
                latent_results_test = evaluate_latent_state([z_hat.T, z_hat_ica.T], mu_test, mu_hat_test, mu_hat_ica_test, source_test, quick_eval=quick_eval)
                reward_results_test = evaluate_reward(treatment_data_test, reward_hat_test, po_hat_test)
                reward_results_ica_test = evaluate_reward(treatment_data_test, reward_hat_ica_test, po_hat_ica_test)
                action_percent_test = evaluate_actions(model, mu_hat_test, treatment_data_test)
                action_percent_ica_test = evaluate_actions(model_ica, mu_hat_ica_test, treatment_data_test)

                # Reporting
                results_latent_test[l][n].append(latent_results_test)
                results_reward_test[l][n].append(reward_results_test)
                results_reward_ica_test[l][n].append(reward_results_ica_test)
                results_action_test[l][n].append(action_percent_test)
                results_action_ica_test[l][n].append(action_percent_ica_test)
                print(f"Results Test for n={n}, l={l}, seed={seed}: \n----------")
                if not quick_eval:
                    print(f"ILB Data Recovery (no ICA): {latent_results_test['mcc_no_ica']} \t  with ICA {latent_results_test['mcc_ica']}")
                print(f"Patient Means Recovery (no ICA): {latent_results_test['mcc_mu_no_ica']} \t with ICA: {latent_results_test['mcc_mu_ica']}")
                print(f"Reward Recovery MSE (w/o noise): {reward_results_test['MSE-thetazed']} \t R2: {reward_results_test['R2-thetazed']}")
                print(f"Reward Recovery MSE (w/o noise) ICA: {reward_results_ica_test['MSE-thetazed']} \t R2: {reward_results_ica_test['R2-thetazed']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent_test}")
                print(f"Percentages of chosen actions ICA (best to worst): {action_percent_ica_test}")
                print('----------')
                append_report(df, config, {'l': l, 'n': n, 'seed': seed}, reward_results, reward_results_ica, latent_results, action_percent, action_percent_ica)
                append_report(df_test, config, {'l': l, 'n': n, 'seed': seed}, reward_results_test, reward_results_ica_test, latent_results_test, action_percent_test, action_percent_ica_test)

    df.to_csv(os.path.join(args.checkpoints, "Results_LVM_Synthetic-train.csv"))
    df_test.to_csv(os.path.join(args.checkpoints, "Results_LVM_Synthetic-test.csv"))
    Results = {
        'data_dim': data_dim,
        'n_patients': n_segments,
        'n_treatment': n_treatment,
        'Train Results': {
            'Accuracy': results_accuracy,
            'Latent Recovery': results_latent,
            'Reward Recovery': results_reward,
            'Reward Recovery ICA': results_reward_ica,
            'Action Recovery': results_action,
            'Action Recovery ICA': results_action_ica,
        },
        'Test Results': {
            'Latent Recovery': results_latent_test,
            'Reward Recovery': results_reward_test,
            'Reward Recovery ICA': results_reward_ica_test,
            'Action Recovery': results_action_test,
            'Action Recovery ICA': results_action_ica_test,
        },
    }
    return Results


def eval_vae_exp(args, config):
    """run TCL simulations"""
    data_dim = config.data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    nSims = args.nSims
    quick_eval = config.quick_eval

    df = df_report()
    df_test = df_report()
    results_latent = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_reward = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_reward_ica = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action_ica = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    results_latent_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_reward_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_reward_ica_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action_ica_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    def _image_name(is_source, n, l, args):
        if is_source:
            img_name = f'source_{n}segment_{l}layers.pdf'
            img_name = 'source', os.path.join(args.checkpoints, img_name)
        else:
            img_name = f'sensor_{n}segment_{l}layers.pdf'
            img_name = 'sensor', os.path.join(args.checkpoints, img_name)
        return img_name

    for l in n_layers:
        for n in n_obs_per_seg:
            # generate mixing data
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            x, y, s, x_val, y_val, source_val, mu, mixing, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(load_pickle(datadict_path))
            #np.testing.assert_array_almost_equal(treatment_data['theta'], treatment_data_test['theta'])
            tcl = TCL.init_from_args(data_dim, n_segments, n)
            visualize_data(x, tcl, *_image_name(False, n, l, args))
            visualize_data(s, tcl, *_image_name(True, n, l, args))
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Testing exp with L={} and n={}; seed={}'.format(l, n, seed))
                # Run Inference for train data
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                modelvae = VAE.load(ckpt_folder)
                latent, _ = modelvae.evaluate(X=x)
                ica = load_pickle(os.path.join(ckpt_folder, 'fast_ica.bin'))
                latent_ica = ica.transform(latent)
                # Post Processing
                mu_hat = get_mu_hat(latent, n=n, n_segments=n_segments)
                mu_hat_ica = get_mu_hat(latent_ica, n=n, n_segments=n_segments)

                # estimate rewards
                model = load_reward_model(data_dim, n_treatment, os.path.join(ckpt_folder, 'theta_model.bin'))
                model_ica = load_reward_model(data_dim, n_treatment, os.path.join(ckpt_folder, 'theta_model_ica.bin'))
                reward_hat = estimate_rewards(model, mu_hat, treatment_data['treatment'])
                reward_hat_ica = estimate_rewards(model_ica, mu_hat_ica, treatment_data['treatment'])
                po_hat = estimate_potential_outcomes(model, mu_hat, treatment_data['treatment'])
                po_hat_ica = estimate_potential_outcomes(model_ica, mu_hat_ica, treatment_data['treatment'])

                # train results
                latent_results = evaluate_latent_state([latent.T, latent_ica.T], mu, mu_hat, mu_hat_ica, s, quick_eval=quick_eval)
                reward_results = evaluate_reward(treatment_data, reward_hat, po_hat)
                reward_results_ica = evaluate_reward(treatment_data, reward_hat_ica, po_hat_ica)
                action_percent = evaluate_actions(model, mu_hat, treatment_data)
                action_percent_ica = evaluate_actions(model_ica, mu_hat_ica, treatment_data)

                # Reporting
                results_latent[l][n].append(latent_results)
                results_reward[l][n].append(reward_results)
                results_reward_ica[l][n].append(reward_results_ica)
                results_action[l][n].append(action_percent)
                results_action_ica[l][n].append(action_percent_ica)
                print(f"Results Train for n={n}, l={l}, seed={seed} : \n----------")
                if not quick_eval:
                    print(f"ILB Data Recovery (no ICA): {latent_results['mcc_no_ica']} \t  with ICA {latent_results['mcc_ica']}")
                print(f"Patient Means Recovery (no ICA): {latent_results['mcc_mu_no_ica']} \t with ICA: {latent_results['mcc_mu_ica']}")
                print(f"Reward Recovery MSE (w/o noise): {reward_results['MSE-thetazed']} \t R2: {reward_results['R2-thetazed']}")
                print(f"Reward Recovery MSE (w/o noise) ICA: {reward_results_ica['MSE-thetazed']} \t R2: {reward_results_ica['R2-thetazed']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent}")
                print(f"Percentages of chosen actions ICA (best to worst): {action_percent_ica}")
                print('----------')

                # Run inference for test data
                z_hat,_ = modelvae.evaluate(X=x_test)
                z_hat_ica = ica.transform(z_hat)
                mu_hat_test = get_mu_hat(z_hat, n=n, n_segments=50)
                mu_hat_ica_test = get_mu_hat(z_hat_ica, n=n, n_segments=50)

                # estimate rewards
                reward_hat_test = estimate_rewards(model, mu_hat_test, treatment_data_test['treatment'])
                reward_hat_ica_test = estimate_rewards(model_ica, mu_hat_ica_test, treatment_data_test['treatment'])
                po_hat_test = estimate_potential_outcomes(model, mu_hat_test, treatment_data_test['treatment'])
                po_hat_ica_test = estimate_potential_outcomes(model_ica, mu_hat_ica_test, treatment_data_test['treatment'])

                # test results
                latent_results_test = evaluate_latent_state([z_hat.T, z_hat_ica.T], mu_test, mu_hat_test, mu_hat_ica_test, source_test, quick_eval=quick_eval)
                reward_results_test = evaluate_reward(treatment_data_test, reward_hat_test, po_hat_test)
                reward_results_ica_test = evaluate_reward(treatment_data_test, reward_hat_ica_test, po_hat_ica_test)
                action_percent_test = evaluate_actions(model, mu_hat_test, treatment_data_test)
                action_percent_ica_test = evaluate_actions(model_ica, mu_hat_ica_test, treatment_data_test)

                # Reporting
                results_latent_test[l][n].append(latent_results_test)
                results_reward_test[l][n].append(reward_results_test)
                results_reward_ica_test[l][n].append(reward_results_ica_test)
                results_action_test[l][n].append(action_percent_test)
                results_action_ica_test[l][n].append(action_percent_ica_test)
                print(f"Results Test for n={n}, l={l}, seed={seed}: \n----------")
                if not quick_eval:
                    print(f"ILB Data Recovery (no ICA): {latent_results_test['mcc_no_ica']} \t  with ICA {latent_results_test['mcc_ica']}")
                print(f"Patient Means Recovery (no ICA): {latent_results_test['mcc_mu_no_ica']} \t with ICA: {latent_results_test['mcc_mu_ica']}")
                print(f"Reward Recovery MSE (w/o noise): {reward_results_test['MSE-thetazed']} \t R2: {reward_results_test['R2-thetazed']}")
                print(f"Reward Recovery MSE (w/o noise) ICA: {reward_results_ica_test['MSE-thetazed']} \t R2: {reward_results_ica_test['R2-thetazed']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent_test}")
                print(f"Percentages of chosen actions ICA (best to worst): {action_percent_ica_test}")
                print('----------')
                append_report(df, config, {'l': l, 'n': n, 'seed': seed}, reward_results, reward_results_ica, latent_results, action_percent, action_percent_ica)
                append_report(df_test, config, {'l': l, 'n': n, 'seed': seed}, reward_results_test, reward_results_ica_test, latent_results_test, action_percent_test, action_percent_ica_test)

    df.to_csv(os.path.join(args.checkpoints, "Results_VAE_Synthetic-train.csv"))
    df_test.to_csv(os.path.join(args.checkpoints, "Results_VAE_Synthetic-test.csv"))
    Results = {
        'data_dim': data_dim,
        'n_patients': n_segments,
        'n_treatment': n_treatment,
        'Train Results': {
            'Latent Recovery': results_latent,
            'Reward Recovery': results_reward,
            'Reward Recovery ICA': results_reward_ica,
            'Action Recovery': results_action,
            'Action Recovery ICA': results_action_ica,
        },
        'Test Results': {
            'Latent Recovery': results_latent_test,
            'Reward Recovery': results_reward_test,
            'Reward Recovery ICA': results_reward_ica_test,
            'Action Recovery': results_action_test,
            'Action Recovery ICA': results_action_ica_test,
        },
    }
    return Results


def evaluate_LSTM_actions(potential_outcomes, treatment_data):
    potentials = treatment_data['potential_outcomes'][:, 0, :]
    arm_order = np.argsort(potentials, axis=-1)
    actions = np.argmax(potential_outcomes, axis=-1)
    action_order = np.zeros_like(actions)
    n_patients, len_treatment = actions.shape
    for p in range(n_patients):
        order = arm_order[p][::-1]
        patient_actions = actions[p]
        for t in range(len_treatment):
            action_order[p, t] = np.where(order == patient_actions[t])[0][0] + 1 # Rank of the action
    unique, counts = np.unique(action_order.flatten(), return_counts=True)
    return dict(zip(unique.tolist(), (counts / (n_patients * len_treatment) * 100).tolist())) # Percentage of actions


def LSTM_report():
    model_cols = ['data_dim', 'n_patients', 'n_treatment', 'n_layers', 'len_treatment', 'seed']
    reward_cols = [
        'MSE-thetazed',
        'R2-thetazed',
        'MSE-Reward',
        'R2-Reward',
        'MSE-Potential_Outcomes',
        'R2-Potential_Outcomes',
        'MSE-PO_noise',
        'R2_PO_noise',
        ]
    action_cols = [
        'best_action',
        'second_best_action',
        'worst_action',
        ]
    return pd.DataFrame(columns=model_cols + reward_cols + action_cols)


def append_LSTM_report(df, config, unique, reward_results, action_percent):
    model_cols = {'data_dim': config.data_dim,
                  'n_patients': config.n_segments,
                  'n_treatment': config.n_treatment,
                  'n_layers': unique['l'],
                  'len_treatment': unique['n'],
                  'seed': unique['seed'],
                  }
    actions = {'best_action': action_percent[1],
               'second_best_action': action_percent.get(2, 0),
               'worst_action': action_percent[max(action_percent.keys())],
               }
    row = {**model_cols, **reward_results, **actions}
    df.loc[len(df)] = row


def eval_LSTM_exp(args, config):
    """run TCL simulations"""
    data_dim = config.data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    nSims = args.nSims

    df = LSTM_report()
    df_test = LSTM_report()
    results_reward = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    results_reward_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    def _image_name(is_source, n, l, args):
        if is_source:
            img_name = f'source_{n}segment_{l}layers.pdf'
            img_name = 'source', os.path.join(args.checkpoints, img_name)
        else:
            img_name = f'sensor_{n}segment_{l}layers.pdf'
            img_name = 'sensor', os.path.join(args.checkpoints, img_name)
        return img_name

    for l in n_layers:
        for n in n_obs_per_seg:
            # generate mixing data
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            x, y, s, x_val, y_val, source_val, mu, mixing, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(load_pickle(datadict_path))
            #np.testing.assert_array_almost_equal(treatment_data['theta'], treatment_data_test['theta'])
            tcl = TCL.init_from_args(data_dim, n_segments, n)
            visualize_data(x, tcl, *_image_name(False, n, l, args))
            visualize_data(s, tcl, *_image_name(True, n, l, args))
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Testing exp with L={} and n={}; seed={}'.format(l, n, seed))
                # Run Inference for train data
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                model = load_model(ckpt_folder, data_dim, n_treatment, l)
                x_time, treatment_id, _ = prepare_data(x, treatment_data, n)
                reward_hat = model(x_time, treatment_id).detach().numpy()
                po_hat = model.get_potential_outcomes(x_time).detach().numpy()
                reward_results = evaluate_reward(treatment_data, reward_hat, po_hat)
                action_percent = evaluate_LSTM_actions(po_hat, treatment_data)

                results_reward[l][n].append(reward_results)
                results_action[l][n].append(action_percent)
                print(f"Results Train for n={n}, l={l}, seed={seed} : \n----------")
                print(f"Reward Recovery MSE (w/o noise): {reward_results['MSE-thetazed']} \t R2: {reward_results['R2-thetazed']}")
                print(f"Potential Outcomes Recovery MSE (w/o noise): {reward_results['MSE-Potential_Outcomes']} \t R2: {reward_results['R2-Potential_Outcomes']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent}")
                print('----------')

                # Run inference for test data
                x_time_test, treatment_id_test, _ = prepare_data(x_test, treatment_data_test, n)
                reward_hat_test = model(x_time_test, treatment_id_test).detach().numpy()
                po_hat_test = model.get_potential_outcomes(x_time_test).detach().numpy()
                reward_results_test = evaluate_reward(treatment_data_test, reward_hat_test, po_hat_test)
                action_percent_test = evaluate_LSTM_actions(po_hat_test, treatment_data_test)

                results_reward_test[l][n].append(reward_results_test)
                results_action_test[l][n].append(action_percent_test)
                print(f"Results Test for n={n}, l={l}, seed={seed}: \n----------")
                print(f"Reward Recovery MSE (w/o noise): {reward_results_test['MSE-thetazed']} \t R2: {reward_results_test['R2-thetazed']}")
                print(f"Potential Outcomes Recovery MSE (w/o noise): {reward_results_test['MSE-Potential_Outcomes']} \t R2: {reward_results_test['R2-Potential_Outcomes']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent_test}")
                print('----------')
                append_LSTM_report(df, config, {'l': l, 'n': n, 'seed': seed}, reward_results, action_percent)
                append_LSTM_report(df_test, config, {'l': l, 'n': n, 'seed': seed}, reward_results_test, action_percent_test)

    df.to_csv(os.path.join(args.checkpoints, "Results_Regression_Synthetic-train.csv"))
    df_test.to_csv(os.path.join(args.checkpoints, "Results_Regression_Synthetic-test.csv"))
    # prepare output
    Results = {
        'data_dim': data_dim,
        'n_patients': n_segments,
        'n_treatment': n_treatment,
        'Train Results': {
            'Reward Recovery': results_reward,
            'Action Recovery': results_action,
        },
        'Test Results': {
            'Reward Recovery': results_reward_test,
            'Action Recovery': results_action_test,
            },
        }
    return Results
