import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from data.utils import to_one_hot
from data.adcb import read_acdc_csv, pack_data_dict, unpack_data_dict
from models.tcl.tcl_wrapper_gpu import TCL_wrapper, inference_tcl
from models.sequence_regressor import sequence_model_wrapper, load_model, prepare_data
from models.vae import vae_wrapper, VAE
from models.mlp import mlp_wrapper, MLP
from runners.plot_utils import plot_loss_and_r2
from utils import save_pickle, load_pickle, write_json_file


def tsne_plot(mu_hat, mu_col, colname, test=False, ckpt_folder='.'):
    mu_tsne = TSNE(n_components=2, perplexity=6).fit_transform(mu_hat)
    fig, ax = plt.subplots(1)
    sns.scatterplot(x=mu_tsne[:,0], y=mu_tsne[:,1], hue=mu_col[colname], ax=ax, palette='Paired')
    m_ = (mu_tsne.min() + mu_tsne.max()) / 2
    l = (mu_tsne.min() - mu_tsne.max()) / 2
    lim = (m_ + 1.2 * l, m_ - 1.2 * l)
    plt.xlabel('Tsne 1')
    plt.ylabel('Tsne 2')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    plt.legend(title=colname.capitalize())
    if test:
        colname += '_test'
    plt.savefig(os.path.join(ckpt_folder, f'tsne_{colname}.png'))
    plt.clf()


def estimate_thetas(mu, reward, treatment):
    n_treatment = treatment.max() + 1
    regression = {}
    for i in range(n_treatment):
        x, y = np.where(treatment == i)
        latents = mu[x]
        R = reward[x, y]
        regression[f'treatment_{i}'] = LinearRegression().fit(X=latents, y=R)
    theta_hat = np.array([v.coef_ for _, v in regression.items()])
    intercept = np.array([v.intercept_ for _, v in regression.items()])
    return {'W': theta_hat, 'b': intercept}


def inference_theta_hat(x, W, b):
    return W @ x.T + np.tile(b, (x.shape[0], 1)).T


def get_mu_hat(latent, n, n_segments):
    """n: len_treatment, n_segments: n_patients."""
    idx = np.arange(0, n*n_segments, n)
    return np.array([np.mean(np.take(latent, axis=0, indices=range(i, i+n)), 0) for i in idx])


def estimate_rewards(theta_dic, mu_hat, treatment):
    n_segment, n_segment_len =  treatment.shape
    reward_hat = np.zeros_like(treatment, dtype=np.float64)
    for p in range(n_segment):
        for t in range(n_segment_len):
            treatment_id = treatment[p, t]
            reward_hat[p, t] = theta_dic['W'][treatment_id] @ mu_hat[p] + theta_dic['b'][treatment_id]
    return reward_hat


def run_ilb_csv(args, config):
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

    results_accuracy = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    print(f"Data is loaded from {config.filepath} with {n_segments} patients.")
    for l in n_layers:
        for n in n_obs_per_seg:
            datadict = read_acdc_csv(config.filepath, n_patients=n_segments, n_timesteps=n, reward_sigma=config.reward_sigma)
            x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(datadict)
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            os.makedirs(data_folder, exist_ok=True)
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            save_pickle(datadict_path, datadict)
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                res_TCL = TCL_wrapper(sensor=x.T, label=y, random_seed=seed,
                                      list_hidden_nodes=[num_comp * 2] * (l - 1) + [num_comp],
                                      max_steps=stepDict[l][0] * 2, max_steps_init=stepDict[l][1],
                                      ckpt_dir=ckpt_folder, test=False, latent_comp=num_comp, is_csv=True,
                                      n=n, l=l)
                # Estimate thetas
                latent = np.asarray(res_TCL[0].T)
                latent_ica = np.asarray(res_TCL[1].T)
                mu_hat = get_mu_hat(latent=latent, n=n, n_segments=n_segments)
                mu_hat_ica = get_mu_hat(latent=latent_ica, n=n, n_segments=n_segments)
                theta_hat = estimate_thetas(mu_hat, treatment_data['reward'], treatment_data['treatment'])
                theta_hat_ica = estimate_thetas(mu_hat_ica, treatment_data['reward'], treatment_data['treatment'])
                save_pickle(os.path.join(ckpt_folder, 'thetas.bin'), theta_hat)
                save_pickle(os.path.join(ckpt_folder, 'thetas_ica.bin'), theta_hat_ica)
                save_pickle(os.path.join(ckpt_folder, 'minmax.bin'), minmax)
                # Summary
                results_accuracy[l][n].append(res_TCL[2])
                print(f"Accuracy for n={n}, l={l} is: {res_TCL[2]} ")
    return {'data_dim': data_dim,
            'n_patients': n_segments,
            'n_treatment': n_treatment,
            'Train Results': {
                'Accuracy': results_accuracy,
                },
            }


def run_vae_csv(args, config):
    """run TCL simulations"""
    data_dim = config.data_dim
    num_comp = data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    num_epochs = config.num_epochs
    nSims = args.nSims

    print(f"Data is loaded from {config.filepath} with {n_segments} patients.")
    for l in n_layers:
        for n in n_obs_per_seg:
            datadict = read_acdc_csv(config.filepath, n_patients=n_segments, n_timesteps=n, reward_sigma=config.reward_sigma)
            x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(datadict)
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            os.makedirs(data_folder, exist_ok=True)
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            save_pickle(datadict_path, datadict)
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
                                    optimizer_params={'weight_decay': 1e-5},
                                    beta=1.0,
                                    capacity=0.0,
                                    hidden_activation_name='relu',
                                    output_activation_name='sigmoid',
                                    batch_norm=False,
                                    dropout_rate=0.2,
                                    is_csv=True,
                                    seed=seed,
                                    ckpt_dir=ckpt_folder,
                                    n=n,l=l)
                ica = FastICA(random_state=seed).fit(latent)
                latent_ica = ica.transform(latent)
                save_pickle(os.path.join(ckpt_folder, 'fast_ica.bin'), ica)

                # Estimate thetas
                mu_hat = get_mu_hat(latent=latent, n=n, n_segments=n_segments)
                mu_hat_ica = get_mu_hat(latent=latent_ica, n=n, n_segments=n_segments)
                theta_hat = estimate_thetas(mu_hat, treatment_data['reward'], treatment_data['treatment'])
                theta_hat_ica = estimate_thetas(mu_hat_ica, treatment_data['reward'], treatment_data['treatment'])
                save_pickle(os.path.join(ckpt_folder, 'thetas.bin'), theta_hat)
                save_pickle(os.path.join(ckpt_folder, 'thetas_ica.bin'), theta_hat_ica)
                save_pickle(os.path.join(ckpt_folder, 'minmax.bin'), minmax)
    return {'data_dim': data_dim,
            'n_patients': n_segments,
            'n_treatment': n_treatment,
            'Train Results': {
                },
            }


def run_LSTM_csv(args, config):
    """run ILB simulations"""
    data_dim = config.data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    nSims = args.nSims
    num_epochs = config.num_epochs

    for l in n_layers:
        for n in n_obs_per_seg:
            datadict = read_acdc_csv(config.filepath, n_patients=n_segments, n_timesteps=n, reward_sigma=config.reward_sigma)
            x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(datadict)
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            os.makedirs(data_folder, exist_ok=True)
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            save_pickle(datadict_path, datadict)
            print(f"Data is loaded from {config.filepath} with {n_segments} patients and {n} time steps.")
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
                save_pickle(os.path.join(ckpt_folder, 'minmax.bin'), minmax)
                write_json_file(history, os.path.join(ckpt_folder, 'history.json'))
    Results = {
        'data_dim': data_dim,
        'n_patients': n_segments,
        'n_treatment': n_treatment,
        'Train Results': {},
        }
    return Results


def run_mlp_csv(args, config):
    """run ILB simulations"""
    data_dim = config.data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    nSims = args.nSims

    for l in n_layers:
        for n in n_obs_per_seg:
            datadict = read_acdc_csv(config.filepath, n_patients=n_segments, n_timesteps=n, reward_sigma=config.reward_sigma)
            x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(datadict)
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            os.makedirs(data_folder, exist_ok=True)
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            save_pickle(datadict_path, datadict)
            print(f"Data is loaded from {config.filepath} with {n_segments} patients and {n} time steps.")
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                os.makedirs(ckpt_folder, exist_ok=True)
                history = mlp_wrapper(sensor=x, sensor_val=x_val, treatment_data=treatment_data,
                                      treatment_data_val=treatment_data_val, ckpt_dir=ckpt_folder,
                                      random_seed=42, is_csv=False, n=n, l=l)
                # Call the function to plot loss and validation R2
                plot_loss_and_r2(history, ckpt_folder)
                save_pickle(os.path.join(ckpt_folder, 'minmax.bin'), minmax)
                write_json_file(history, os.path.join(ckpt_folder, 'history.json'))
    Results = {
        'data_dim': data_dim,
        'n_patients': n_segments,
        'n_treatment': n_treatment,
        'Train Results': {},
        }
    return Results


def estimate_potential_outcomes(theta_dic, mu_hat, treatment):
    n_patients, len_treatment =  treatment.shape
    n_treatment = len(theta_dic['b'])
    po_hat = np.zeros((n_patients, len_treatment, n_treatment), dtype=np.float64)
    for p in range(n_patients):
        for t in range(len_treatment):
            po_hat[p, t] = theta_dic['W'] @ mu_hat[p] + theta_dic['b']
    return po_hat


def estimate_actions(theta_dic, mu_hat, len_treatment):
    n_patients = mu_hat.shape[0]
    actions = np.zeros(shape=(n_patients, len_treatment), dtype=int)
    for p in range(n_patients):
        for t in range(len_treatment):
            po = theta_dic['W'] @ mu_hat[p] + theta_dic['b']
            actions[p, t] = np.argmax(po)
    return actions


def evaluate_actions(theta_dic, mu_hat, treatment_data):
    potentials = treatment_data['potential_outcomes'][:, 0, :]
    arm_order = np.argsort(potentials, axis=-1)
    actions = estimate_actions(theta_dic, mu_hat, len_treatment=treatment_data['treatment'].shape[1])
    action_order = np.zeros_like(actions)
    n_patients, len_treatment = actions.shape
    for p in range(n_patients):
        order = arm_order[p][::-1]
        patient_actions = actions[p]
        for t in range(len_treatment):
            action_order[p, t] = np.where(order == patient_actions[t])[0][0] + 1 # Rank of the action
    unique, counts = np.unique(action_order.flatten(), return_counts=True)
    return dict(zip(unique.tolist(), (counts / (n_patients * len_treatment) * 100).tolist())) # Percentage of actions


def fit_latent_regression(mu, mu_hat, mu_hat_ica):
    z_cols = ["ABETARatio", "PTETHCAT", "PTRACCAT", "PTGENDER", "APOE4"]
    lm = {z : (LogisticRegression(penalty=None).fit(X=mu_hat, y=mu[z]),
               LogisticRegression(penalty=None).fit(X=mu_hat_ica, y=mu[z]),
               ) for z in z_cols[1:]}
    lm["ABETARatio"] = (LinearRegression().fit(X=mu_hat, y=mu['ABETARatio']),
                        LinearRegression().fit(X=mu_hat_ica, y=mu['ABETARatio']),
                        )
    return lm


def evaluate_latent_state(lm, mu, mu_hat, mu_hat_ica, df_cat):
    results = {}
    for z, (lm, lm_ica) in lm.items():
        if z == 'ABETARatio':
            r2_latent = lm.score(X=mu_hat, y=mu['ABETARatio'])
            r2_latent_ica = lm_ica.score(X=mu_hat_ica, y=mu['ABETARatio'])
            results['ABETARatio-R2'] = (r2_latent, r2_latent_ica)
            continue
        pred = lm.predict_proba(X=mu_hat)
        pred_ica = lm_ica.predict_proba(X=mu_hat_ica)
        score = roc_auc_score(y_score=pred, y_true=to_one_hot(mu[z], df_cat[z])[0])
        score_ica = roc_auc_score(y_score=pred_ica, y_true=to_one_hot(mu[z], df_cat[z])[0])
        results[z + '-AUC'] = (score, score_ica)
    return results


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
    action_cols = [
        'best_action',
        'second_best_action',
        'worst_action',
        ]
    return pd.DataFrame(columns=model_cols + reward_cols + action_cols)


def append_report(df, config, unique, reward_results, action_percent):
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


def eval_ilb_csv(args, config):
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

    df = df_report()
    df_ica = df_report()
    df_test = df_report()
    df_ica_test = df_report()
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

    for l in n_layers:
        for n in n_obs_per_seg:
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(load_pickle(datadict_path))
            df_cat = mu.nunique()

            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Testing exp with L={} and n={}; seed={}'.format(l, n, seed))
                # Run Inference for train data
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                res_TCL = TCL_wrapper(sensor=x.T, label=y, random_seed=seed,
                                      list_hidden_nodes=[num_comp * 2] * (l - 1) + [num_comp],
                                      max_steps=stepDict[l][0] * 2, max_steps_init=stepDict[l][1],
                                      ckpt_dir=ckpt_folder, test=True, latent_comp=num_comp, is_csv=True, **{'n': n, 'l': l})
                # Post Processing
                results_accuracy[l][n].append(res_TCL[2])
                latent = np.asarray(res_TCL[0].T) # (n_segments * n_patients, data_dim)
                latent_ica = np.asarray(res_TCL[1].T) # (n_segments * n_patients, data_dim)
                mu_hat = get_mu_hat(latent, n=n, n_segments=n_segments)
                mu_hat_ica = get_mu_hat(latent_ica, n=n, n_segments=n_segments)
                theta_hat = load_pickle(os.path.join(ckpt_folder, 'thetas.bin'))
                theta_hat_ica = load_pickle(os.path.join(ckpt_folder, 'thetas_ica.bin'))
                lm = fit_latent_regression(mu, mu_hat, mu_hat_ica)

                # estimate rewards
                reward_hat = estimate_rewards(theta_hat, mu_hat, treatment_data['treatment'])
                reward_hat_ica = estimate_rewards(theta_hat_ica, mu_hat_ica, treatment_data['treatment'])
                po_hat = estimate_potential_outcomes(theta_hat, mu_hat, treatment_data['treatment'])
                po_hat_ica = estimate_potential_outcomes(theta_hat_ica, mu_hat_ica, treatment_data['treatment'])

                # train results
                latent_results = evaluate_latent_state(lm, mu, mu_hat, mu_hat_ica, df_cat)
                reward_results = evaluate_reward(treatment_data, reward_hat, po_hat)
                reward_results_ica = evaluate_reward(treatment_data, reward_hat_ica, po_hat_ica)
                action_percent = evaluate_actions(theta_hat, mu_hat, treatment_data)
                action_percent_ica = evaluate_actions(theta_hat_ica, mu_hat_ica, treatment_data)

                # tsne plots
                for colname in ['RID', 'PTETHCAT', 'PTRACCAT', 'PTGENDER', 'APOE4', 'ABETARatio']:
                    tsne_plot(mu_hat, mu, colname, False, ckpt_folder)

                # Reporting
                results_accuracy[l][n].append(res_TCL[2])
                results_latent[l][n].append(latent_results)
                results_reward[l][n].append(reward_results)
                results_reward_ica[l][n].append(reward_results_ica)
                results_action[l][n].append(action_percent)
                results_action_ica[l][n].append(action_percent_ica)
                print(f"Results Train for n={n}, l={l}, seed={seed} : \n----------")
                print(f"Accuracy is: {res_TCL[2]}")
                print(f"Reward Recovery MSE (w/o noise): {reward_results['MSE-thetazed']} \t R2: {reward_results['R2-thetazed']}")
                print(f"Reward Recovery MSE (w/o noise) ICA: {reward_results_ica['MSE-thetazed']} \t R2: {reward_results_ica['R2-thetazed']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent}")
                print(f"Percentages of chosen actions ICA (best to worst): {action_percent_ica}")
                print('----------')

                # Run inference for test data
                ica = load_pickle(os.path.join(ckpt_folder, 'fast_ica.bin'))
                z_hat = inference_tcl(x_test.T, np.zeros(x_test.shape[0]), [num_comp * 2] * (l - 1) + [num_comp], ckpt_folder, is_csv=True)
                z_hat_ica = ica.transform(z_hat)
                mu_hat_test = get_mu_hat(z_hat, n=n, n_segments=len(mu_test))
                mu_hat_ica_test = get_mu_hat(z_hat_ica, n=n, n_segments=len(mu_test))

                # estimate rewards
                reward_hat_test = estimate_rewards(theta_hat, mu_hat_test, treatment_data_test['treatment'])
                reward_hat_ica_test = estimate_rewards(theta_hat_ica, mu_hat_ica_test, treatment_data_test['treatment'])
                po_hat_test = estimate_potential_outcomes(theta_hat, mu_hat_test, treatment_data_test['treatment'])
                po_hat_ica_test = estimate_potential_outcomes(theta_hat_ica, mu_hat_ica_test, treatment_data_test['treatment'])

                # test results
                latent_results_test = evaluate_latent_state(lm, mu_test, mu_hat_test, mu_hat_ica_test, df_cat)
                reward_results_test = evaluate_reward(treatment_data_test, reward_hat_test, po_hat_test)
                reward_results_ica_test = evaluate_reward(treatment_data_test, reward_hat_ica_test, po_hat_ica_test)
                action_percent_test = evaluate_actions(theta_hat, mu_hat_test, treatment_data_test)
                action_percent_ica_test = evaluate_actions(theta_hat_ica, mu_hat_ica_test, treatment_data_test)

                # tsne plots
                for colname in ['RID', 'PTETHCAT', 'PTRACCAT', 'PTGENDER', 'APOE4', 'ABETARatio']:
                    tsne_plot(mu_hat_test, mu_test, colname, True, ckpt_folder)

                # Reporting
                results_latent_test[l][n].append(latent_results_test)
                results_reward_test[l][n].append(reward_results_test)
                results_reward_ica_test[l][n].append(reward_results_ica_test)
                results_action_test[l][n].append(action_percent_test)
                results_action_ica_test[l][n].append(action_percent_ica_test)
                print(f"Results Test for n={n}, l={l}, seed={seed}: \n----------")
                print(f"Reward Recovery MSE (w/o noise): {reward_results_test['MSE-thetazed']} \t R2: {reward_results_test['R2-thetazed']}")
                print(f"Reward Recovery MSE (w/o noise) ICA: {reward_results_ica_test['MSE-thetazed']} \t R2: {reward_results_ica_test['R2-thetazed']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent_test}")
                print(f"Percentages of chosen actions ICA (best to worst): {action_percent_ica_test}")
                print('----------')
                append_report(df, config, {'l': l, 'n': n, 'seed': seed}, reward_results, action_percent)
                append_report(df_ica, config, {'l': l, 'n': n, 'seed': seed}, reward_results_ica, action_percent_ica)
                append_report(df_test, config, {'l': l, 'n': n, 'seed': seed}, reward_results_test, action_percent_test)
                append_report(df_ica_test, config, {'l': l, 'n': n, 'seed': seed}, reward_results_ica_test, action_percent_ica_test)

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


def eval_vae_csv(args, config):
    """run TCL simulations"""
    data_dim = config.data_dim
    num_comp = data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    nSims = args.nSims

    df = df_report()
    df_ica = df_report()
    df_test = df_report()
    df_ica_test = df_report()
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

    for l in n_layers:
        for n in n_obs_per_seg:
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(load_pickle(datadict_path))
            df_cat = mu.nunique()

            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Testing exp with L={} and n={}; seed={}'.format(l, n, seed))
                # Run Inference for train data
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))
                model = VAE.load(ckpt_folder)
                latent, _ = model.evaluate(X=x)
                ica = load_pickle(os.path.join(ckpt_folder, 'fast_ica.bin'))
                latent_ica = ica.transform(latent)

                # Post Processing
                mu_hat = get_mu_hat(latent, n=n, n_segments=n_segments)
                mu_hat_ica = get_mu_hat(latent_ica, n=n, n_segments=n_segments)
                theta_hat = load_pickle(os.path.join(ckpt_folder, 'thetas.bin'))
                theta_hat_ica = load_pickle(os.path.join(ckpt_folder, 'thetas_ica.bin'))
                lm = fit_latent_regression(mu, mu_hat, mu_hat_ica)

                # estimate rewards
                reward_hat = estimate_rewards(theta_hat, mu_hat, treatment_data['treatment'])
                reward_hat_ica = estimate_rewards(theta_hat_ica, mu_hat_ica, treatment_data['treatment'])
                po_hat = estimate_potential_outcomes(theta_hat, mu_hat, treatment_data['treatment'])
                po_hat_ica = estimate_potential_outcomes(theta_hat_ica, mu_hat_ica, treatment_data['treatment'])

                # train results
                latent_results = evaluate_latent_state(lm, mu, mu_hat, mu_hat_ica, df_cat)
                reward_results = evaluate_reward(treatment_data, reward_hat, po_hat)
                reward_results_ica = evaluate_reward(treatment_data, reward_hat_ica, po_hat_ica)
                action_percent = evaluate_actions(theta_hat, mu_hat, treatment_data)
                action_percent_ica = evaluate_actions(theta_hat_ica, mu_hat_ica, treatment_data)

                # tsne plots
                for colname in ['RID', 'PTETHCAT', 'PTRACCAT', 'PTGENDER', 'APOE4', 'ABETARatio']:
                    tsne_plot(mu_hat, mu, colname, False, ckpt_folder)

                # Reporting
                results_latent[l][n].append(latent_results)
                results_reward[l][n].append(reward_results)
                results_reward_ica[l][n].append(reward_results_ica)
                results_action[l][n].append(action_percent)
                results_action_ica[l][n].append(action_percent_ica)
                print(f"Results Train for n={n}, l={l}, seed={seed} : \n----------")
                print(f"Reward Recovery MSE (w/o noise): {reward_results['MSE-thetazed']} \t R2: {reward_results['R2-thetazed']}")
                print(f"Reward Recovery MSE (w/o noise) ICA: {reward_results_ica['MSE-thetazed']} \t R2: {reward_results_ica['R2-thetazed']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent}")
                print(f"Percentages of chosen actions ICA (best to worst): {action_percent_ica}")
                print('----------')

                # Run inference for test data
                z_hat, _ = model.evaluate(X=x_test)
                z_hat_ica = ica.transform(z_hat)
                mu_hat_test = get_mu_hat(z_hat, n=n, n_segments=len(mu_test))
                mu_hat_ica_test = get_mu_hat(z_hat_ica, n=n, n_segments=len(mu_test))

                # estimate rewards
                reward_hat_test = estimate_rewards(theta_hat, mu_hat_test, treatment_data_test['treatment'])
                reward_hat_ica_test = estimate_rewards(theta_hat_ica, mu_hat_ica_test, treatment_data_test['treatment'])
                po_hat_test = estimate_potential_outcomes(theta_hat, mu_hat_test, treatment_data_test['treatment'])
                po_hat_ica_test = estimate_potential_outcomes(theta_hat_ica, mu_hat_ica_test, treatment_data_test['treatment'])

                # test results
                latent_results_test = evaluate_latent_state(lm, mu_test, mu_hat_test, mu_hat_ica_test, df_cat)
                reward_results_test = evaluate_reward(treatment_data_test, reward_hat_test, po_hat_test)
                reward_results_ica_test = evaluate_reward(treatment_data_test, reward_hat_ica_test, po_hat_ica_test)
                action_percent_test = evaluate_actions(theta_hat, mu_hat_test, treatment_data_test)
                action_percent_ica_test = evaluate_actions(theta_hat_ica, mu_hat_ica_test, treatment_data_test)

                # tsne plots
                for colname in ['RID', 'PTETHCAT', 'PTRACCAT', 'PTGENDER', 'APOE4', 'ABETARatio']:
                    tsne_plot(mu_hat_test, mu_test, colname, True, ckpt_folder)

                # Reporting
                results_latent_test[l][n].append(latent_results_test)
                results_reward_test[l][n].append(reward_results_test)
                results_reward_ica_test[l][n].append(reward_results_ica_test)
                results_action_test[l][n].append(action_percent_test)
                results_action_ica_test[l][n].append(action_percent_ica_test)
                print(f"Results Test for n={n}, l={l}, seed={seed}: \n----------")
                print(f"Reward Recovery MSE (w/o noise): {reward_results_test['MSE-thetazed']} \t R2: {reward_results_test['R2-thetazed']}")
                print(f"Reward Recovery MSE (w/o noise) ICA: {reward_results_ica_test['MSE-thetazed']} \t R2: {reward_results_ica_test['R2-thetazed']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent_test}")
                print(f"Percentages of chosen actions ICA (best to worst): {action_percent_ica_test}")
                print('----------')
                append_report(df, config, {'l': l, 'n': n, 'seed': seed}, reward_results, action_percent)
                append_report(df_ica, config, {'l': l, 'n': n, 'seed': seed}, reward_results_ica, action_percent_ica)
                append_report(df_test, config, {'l': l, 'n': n, 'seed': seed}, reward_results_test, action_percent_test)
                append_report(df_ica_test, config, {'l': l, 'n': n, 'seed': seed}, reward_results_ica_test, action_percent_ica_test)

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


def eval_LSTM_csv(args, config):
    """run ILB simulations"""
    data_dim = config.data_dim
    num_comp = data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    nSims = args.nSims
    num_epochs = config.num_epochs

    df = df_report()
    df_test = df_report()
    results_reward = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    results_reward_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    for l in n_layers:
        for n in n_obs_per_seg:
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(load_pickle(datadict_path))
            print(f"Data is loaded from {config.filepath} with {n_segments} patients and {n} time steps.")
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Eval exp with L={} and n={}; seed={}'.format(l, n, seed))
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))

                # train results
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
                append_report(df, config, {'l': l, 'n': n, 'seed': seed}, reward_results, action_percent)
                append_report(df_test, config, {'l': l, 'n': n, 'seed': seed}, reward_results_test, action_percent_test)

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


def eval_mlp_csv(args, config):
    """Eval ADCB simulations"""
    data_dim = config.data_dim
    num_comp = data_dim
    n_segments = config.n_segments
    n_treatment = config.n_treatment
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    nSims = args.nSims
    num_epochs = config.num_epochs

    df = df_report()
    df_test = df_report()
    results_reward = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    results_reward_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}
    results_action_test = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    for l in n_layers:
        for n in n_obs_per_seg:
            data_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n))
            datadict_path = os.path.join(data_folder, 'data_dict.bin')
            x, y, s, x_val, y_val, source_val, mu, minmax, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test = unpack_data_dict(load_pickle(datadict_path))
            print(f"Data is loaded from {config.filepath} with {n_segments} patients and {n} time steps.")
            for seed in range(args.start_seed, args.start_seed + nSims):
                print('Eval exp with L={} and n={}; seed={}'.format(l, n, seed))
                ckpt_folder = os.path.join(args.checkpoints, args.dataset, str(l), str(n), str(seed))

                # train results
                model = MLP.load(ckpt_folder)
                x_time, treatment_id, _ = prepare_data(x, treatment_data, n)
                reward_hat = model.predict(x_time, treatment_id)
                po_hat = model.get_potential_outcomes(x_time)
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
                reward_hat_test = model.predict(x_time_test, treatment_id_test)
                po_hat_test = model.get_potential_outcomes(x_time_test)
                reward_results_test = evaluate_reward(treatment_data_test, reward_hat_test, po_hat_test)
                action_percent_test = evaluate_LSTM_actions(po_hat_test, treatment_data_test)

                results_reward_test[l][n].append(reward_results_test)
                results_action_test[l][n].append(action_percent_test)
                print(f"Results Test for n={n}, l={l}, seed={seed}: \n----------")
                print(f"Reward Recovery MSE (w/o noise): {reward_results_test['MSE-thetazed']} \t R2: {reward_results_test['R2-thetazed']}")
                print(f"Potential Outcomes Recovery MSE (w/o noise): {reward_results_test['MSE-Potential_Outcomes']} \t R2: {reward_results_test['R2-Potential_Outcomes']}")
                print(f"Percentages of chosen actions (best to worst): {action_percent_test}")
                print('----------')
                append_report(df, config, {'l': l, 'n': n, 'seed': seed}, reward_results, action_percent)
                append_report(df_test, config, {'l': l, 'n': n, 'seed': seed}, reward_results_test, action_percent_test)

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
