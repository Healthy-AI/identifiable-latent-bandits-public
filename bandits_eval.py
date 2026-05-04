import os
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils import save_pickle, load_pickle

DIR = ""
REGRET = "$Reg_T$ (Cum. Regret)"
TIME = "Rounds ($T$)"
FILENAMES = {
    'greedy1':'greedy_LVM1_regret.pkl',
    'greedy2': 'greedy_LVM2_regret.pkl',
    'mab':'mab_regret.pkl',
    'mab_prior':'mab_prior_regret.pkl',
    'projected_thompson':'proj_thompson_regret.pkl',
    'greedy1_oracle': 'greedy_oracle1_regret.pkl',
    'greedy2_oracle': 'greedy_oracle2_regret.pkl',
    'projected_thompson_oracle':'proj_thompson_oracle_regret.pkl',
    'greedy1_oracle': 'greedy_oracle1_regret.pkl',
    'greedy1_VAE': 'greedy_vae1_regret.pkl',
    'greedy2_VAE': 'greedy_vae2_regret.pkl',
    'regression': 'regression_regret.pkl',
    'warmstart_thompson': 'warmstart_thompson.pkl',
    'linear_thompson': 'linear_thompson.pkl',
    'random_feature_bandit': 'rfb.pkl'
    }
NAMES = {
    'greedy1': 'CPG',
    'greedy2': 'FPG',
    'mab': 'MAB',
    'mab_prior': 'MAB Prior',
    'projected_thompson': 'FPG-TS',
    'greedy1_oracle': 'CPG (Oracle)',
    'greedy2_oracle': 'FPG (Oracle)',
    'projected_thompson_oracle': 'FPG-TS (Oracle)',
    'greedy1_VAE': 'CPG (VAE)',
    'greedy2_VAE': 'FPG (VAE)',
    'regression': 'Regression',
    'warmstart_thompson': 'Warmstart TS',
    'linear_thompson': 'LinUCB',
    'random_feature_bandit': 'Random Feature Bandit'
}
PALETTE = { 
    'mab': 'C0',
    'greedy1': 'C1',
    'greedy2': 'C2',
    'projected_thompson': 'C5',
    'greedy1_oracle': 'C3',
    'greedy2_oracle': 'C4',
    'mab_prior': 'C6',
    'greedy1_VAE': 'C7',
    'greedy2_VAE': 'C8',
    'regression': 'C9',
    'linear_thompson':'purple',
    'warmstart_thompson': 'pink',
}
Z_ORDER = {
    'mab': 17,
    'greedy1': 25,
    'greedy2': 26,
    'greedy1_oracle': 23,
    'greedy2_oracle': 24,
    'projected_thompson': 4,
    'mab_prior': 14,
    'greedy1_VAE': 20,
    'greedy2_VAE': 19,
    'regression': 2,
    'linear_thompson': 5,
    'warmstart_thompson': 3,
}


def save_regret_pickle(banditdir, bandit):
    banditdir = Path(banditdir)
    filename = FILENAMES.get(bandit)
    bandit_results_dir = banditdir / bandit
    x = np.array([load_pickle(f)['regret'] for f in bandit_results_dir.iterdir() if str(f).endswith('bin')])
    save_pickle(banditdir / filename, x)


def plot(regret, label, ax, colour, errorbar='sd', zorder=None, linestyle='-'):
    n_seeds, n_iter = regret.shape
    ne = errorbar.get("n_steps") if isinstance(errorbar, dict) else 100
    # Compute mean and std
    time_steps = np.arange(1, n_iter + 1)
    errorbar_indices = np.arange(ne-1, n_iter, ne)
    mean_regret = np.mean(regret, axis=0)
    std_regret = np.std(regret, axis=0) / np.sqrt(n_seeds)
    # Regret
    ax.plot(
        time_steps,
        mean_regret,
        label=label,
        color=colour,
        zorder=zorder,
        linestyle=linestyle,
        )
    ax.errorbar(
        time_steps[errorbar_indices],
        mean_regret[errorbar_indices],
        yerr=std_regret[errorbar_indices],
        fmt='o',
        capsize=3,
        color=colour,
        zorder=zorder,
        #label=f'Std Dev (every {ne} steps)'
    )


def cum_regret_table(banditdir, bandit_list):
    regret_dict = {}
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES[bandit]))
        cum_regret = np.cumsum(regret_dict[bandit], axis=1)
        mean = np.mean(cum_regret, axis=0)
        std = np.std(cum_regret, axis=0) / np.sqrt(np.shape(cum_regret)[0])
        print(f"{NAMES[bandit]}50 - {mean[50]:.2f} ± {std[50]:.2f}")
        print(f"{NAMES[bandit]}200 - {mean[200]:.2f}± {std[200]:.2f}")
        print(f"{NAMES[bandit]}500 - {mean[500]:.2f} ± {std[500]:.2f}")
        print(f"{NAMES[bandit]}1000 - {mean[1000]:.2f} ± {std[1000]:.2f}")


def cum_regret_plot(banditdir, bandit_list, errorbar=None, y_lim=300, figsize=(5,5), fonstize='x-small', savepath=None):
    errorbar = {'n_steps': 100}
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    regret_dict = {}
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1),  NAMES[bandit], ax, colour=PALETTE[bandit], errorbar=errorbar, zorder=Z_ORDER[bandit])
    # set axis lims and labels
    ax.set_xlim([0, len(regret_dict[bandit].T)])
    ax.set_ylim([0, y_lim])
    # set fontsize ticks
    ax.tick_params(axis="both", which="major")
    # set legend size
    ax.legend(loc="upper right", fontsize=fonstize).set_zorder(100)
    ax.set_xlabel(TIME)
    ax.set_ylabel(REGRET)
    # Drop legend from ax[1]
    # Move legend in ax[0] to upper right corner
    # pdf crop
    fig.tight_layout()
    if savepath is None:
        savepath = os.path.join(banditdir, 'Results.pdf')
    fig.savefig(savepath, dpi=300)
    plt.close()


def joint_regret_plot(banditdir, bandit_list, errorbar=None, y_lims=None, inset=None, inset_idx=[0, 100]):
    errorbar = ('se', 2) if errorbar is None else errorbar
    y_lims = [0.1, 40] if y_lims is None else y_lims
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    regret_dict = {}
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES[bandit]))
        plot(regret_dict[bandit], NAMES[bandit], ax[0], colour=PALETTE[bandit], errorbar=errorbar, zorder=Z_ORDER[bandit])
        plot(np.cumsum(regret_dict[bandit], axis=1), bandit, ax[1], colour=PALETTE[bandit], errorbar=("se", 2))
    # set axis lims and labels
    ax[0].set_xlim([0, len(regret_dict[bandit].T)])
    ax[1].set_xlim([0, len(regret_dict[bandit].T)])
    ax[0].set_ylim([0, y_lims[0]])
    ax[1].set_ylim([0, y_lims[1]])
    # set fontsize ticks
    ax[0].tick_params(axis="both", which="major")
    ax[1].tick_params(axis="both", which="major")
    # set legend size
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel(TIME)
    ax[1].set_xlabel(TIME)
    ax[0].set_ylabel(REGRET)
    ax[1].set_ylabel(REGRET)
    # Drop legend from ax[1]
    ax[1].legend().remove()
    # Move legend in ax[0] to upper right corner
    ax[0].legend(loc="upper right").set_zorder(100)
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(banditdir, 'Results.pdf'), dpi=300)
    plt.close()


def joint_regret_plot_zoom(banditdir, bandit_list, bandit_list2, errorbar=None, y_lims=None):
    y_lims = [0.1, 0.05, 40] if y_lims is None else y_lims
    errorbar = ('se', 2) if errorbar is None else errorbar
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    regret_dict = {}
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES[bandit]))
        plot(regret_dict[bandit], NAMES[bandit], ax[0], colour=PALETTE[bandit], errorbar=errorbar, zorder=Z_ORDER[bandit])
        plot(np.cumsum(regret_dict[bandit], axis=1), bandit, ax[2], colour=PALETTE[bandit], errorbar=("se", 2))
    for bandit in bandit_list2:
        plot(regret_dict[bandit], NAMES[bandit], ax[1], colour=PALETTE[bandit], errorbar=("se", 2),zorder=Z_ORDER[bandit])
    # set axis lims and labels
    ax[0].set_xlim([0, len(regret_dict[bandit].T)])
    ax[1].set_xlim([0, len(regret_dict[bandit].T)])
    ax[2].set_xlim([0, len(regret_dict[bandit].T)])
    ax[0].set_ylim([0, y_lims[0]])
    ax[1].set_ylim([0, y_lims[1]])
    ax[2].set_ylim([0, y_lims[2]])
    # set fontsize ticks
    ax[0].tick_params(axis="both", which="major")
    ax[1].tick_params(axis="both", which="major")
    ax[2].tick_params(axis="both", which="major")
    # set legend size
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Regret")
    ax[1].set_ylabel("Regret")
    ax[2].set_ylabel("Cumulative Regret")
    # Drop legend from ax[1]
    ax[1].legend().remove()
    ax[2].legend().remove()
    # Move legend in ax[0] to upper right corner
    ax[0].legend(loc="upper right")
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(banditdir, 'Results.pdf'), dpi=300)
    plt.close()


## For convenience after a run
#save_regret_pickle(banditdir, 'greedy1')
#save_regret_pickle(banditdir, 'greedy1_oracle')
#save_regret_pickle(banditdir, 'greedy2')
#save_regret_pickle(banditdir, 'greedy2_oracle')
#save_regret_pickle(banditdir, 'projected_thompson')
#save_regret_pickle(banditdir, 'projected_thompson_oracle')
#save_regret_pickle(banditdir, 'linear_thompson')
#save_regret_pickle(banditdir, 'mab')
#save_regret_pickle(banditdir, 'mab_prior')
#save_regret_pickle(banditdir, 'random_feature_bandit')
#save_regret_pickle(banditdir, 'regression')
#save_regret_pickle(banditdir, 'warmstart_thompson')
#bandit_list = ['mab', 'greedy1', 'greedy2', 'projected_thompson', 'mab_prior', 'linear_thompson', 'greedy1_VAE', 'greedy2_VAE', 'regression'] 


################################################
################################################
# Results for the Paper
################################################
################################################


################################################
# ADCB and ADCB - Z_01 Plots
################################################
ADCB = ""
Z_01 = ""
ADCB_NOAGE = ""
adcb_list = ['mab', 'greedy1', 'greedy2', 'projected_thompson', 'mab_prior', 'linear_thompson', 'greedy1_VAE', 'greedy2_VAE', 'regression']
adcb_list01 = ['mab', 'greedy1', 'greedy2', 'projected_thompson', 'mab_prior', 'linear_thompson', 'greedy1_VAE', 'greedy2_VAE', 'regression']
adcb_list2 = ['mab', 'greedy1', 'greedy2', 'projected_thompson', 'greedy1_VAE', 'greedy2_VAE'] 
save_adcb = os.path.join(DIR, "Results-ADCB.pdf")
save_z01 = os.path.join(DIR, "Results-ADCB-Z01.pdf")
save_noAge = os.path.join(DIR, "Results-ADCB_noAge.pdf")
#cum_regret_plot(banditdir=ADCB, bandit_list=adcb_list, y_lim=50, figsize=(7.5, 5), savepath=save_adcb)
#cum_regret_plot(banditdir=Z_01, bandit_list=adcb_list01, y_lim=50, figsize=(7.5, 5), savepath=save_z01) 
#cum_regret_plot(banditdir=ADCB_NOAGE, bandit_list=adcb_list2, y_lim=50, figsize=(7.5, 5), savepath=save_noAge) 


################################################
# Syntehtic Plots
################################################
Synthetic = ""
bandit_list_s = ['mab', 'greedy1', 'greedy2', 'greedy1_oracle', 'greedy2_oracle', 'projected_thompson', 'mab_prior', 'greedy1_VAE', 'greedy2_VAE', 'regression', 'linear_thompson', 'warmstart_thompson']


def synthetic_plot(banditdir=Synthetic, bandit_list=bandit_list_s, y_lim=200, figsize=(5,5), fonstize='x-small'):
    errorbar = {'n_steps': 100}
    #y_lims = [0.1, 40] if y_lims is None else y_lims
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    regret_dict = {}
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES[bandit]))
        #plot(regret_dict[bandit], NAMES[bandit], ax[0], colour=PALETTE[bandit], errorbar=errorbar, zorder=Z_ORDER[bandit])
        plot(np.cumsum(regret_dict[bandit], axis=1),  NAMES[bandit], ax, colour=PALETTE[bandit], errorbar=errorbar)
    # set axis lims and labels
    ax.set_xlim([0, len(regret_dict[bandit].T)])
    ax.set_ylim([0, y_lim])
    # set fontsize ticks
    ax.tick_params(axis="both", which="major")
    # set legend size
    ax.legend(loc="upper right", fontsize=fonstize).set_zorder(100)
    ax.set_xlabel(TIME)
    ax.set_ylabel(REGRET)
    # Drop legend from ax[1]
    # Move legend in ax[0] to upper right corner
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, 'Results-Synthetic.pdf'), dpi=300)
    plt.close()


#synthetic_plot(figsize=(8.5, 5))
#cum_regret_plot(banditdir=Synthetic, bandit_list=bandit_list_s, y_lim=200, figsize=(6, 4),)


################################################
# Reward $\sigma=1$ and $\sigma=2$ Plots
################################################
banditdir_r05=""
banditdir_r1=""
banditdir_r2=""
bandit_list = ['mab', 'greedy1', 'greedy2' 'projected_thompson',]# 'greedy1_VAE', 'greedy2_VAE', 'regression'] 
save_reward05 = os.path.join(DIR, "Results-Synthetic-Reward05.pdf")
save_reward1 = os.path.join(DIR, "Results-Synthetic-Reward1.pdf")
save_reward2 = os.path.join(DIR, "Results-Synthetic-Reward2.pdf")
cum_regret_plot(banditdir=banditdir_r05, bandit_list=bandit_list, y_lim=150, figsize=(7.5, 5), savepath=save_reward05)
cum_regret_plot(banditdir=banditdir_r1, bandit_list=bandit_list, y_lim=150, figsize=(7.5, 5), savepath=save_reward1)
cum_regret_plot(banditdir=banditdir_r2, bandit_list=bandit_list, y_lim=150, figsize=(7.5, 5), savepath=save_reward2) 


################################################
# Plot for FPG Overspec
################################################

def FPG_Overspec(bandit_list=['mab', 'greedy1', 'greedy1_VAE', 'greedy2', 'greedy2_VAE']):
    banditdir = ""
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    regret_dict = {}
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES[bandit]))
        plot(regret_dict[bandit], NAMES[bandit], ax[0], colour=PALETTE[bandit], errorbar=("se", 2))
        plot(np.cumsum(regret_dict[bandit], axis=1), bandit, ax[1], colour=PALETTE[bandit], errorbar=("se", 2))
    # set axis lims and labels
    ax[0].set_xlim([0, len(regret_dict[bandit].T)])
    ax[1].set_xlim([0, len(regret_dict[bandit].T)])
    ax[0].set_ylim([0, 1.])
    ax[1].set_ylim([0, 100])
    # set fontsize ticks
    ax[0].tick_params(axis="both", which="major")
    ax[1].tick_params(axis="both", which="major")
    # set legend size
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Regret")
    ax[1].set_ylabel("Cumulative Regret")
    # Drop legend from ax[1]
    ax[1].legend().remove()
    # Move legend in ax[0] to upper right corner
    ax[0].legend(loc="upper right")
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(banditdir, 'Results-FPG2.pdf'), dpi=300)
    plt.close()

### OOD Plotting
OOD = ""
OOD_List = ['greedy1_ood05', 'greedy2_ood05', 'regression_ood05',
            'greedy1_ood1', 'greedy2_ood1', 'regression_ood1',
            'greedy1_ood2', 'greedy2_ood2', 'regression_ood2']


def get_color_ood(bandit_name):
    if 'greedy1' in bandit_name:
        return PALETTE['greedy1']
    elif 'greedy2' in bandit_name:
        return PALETTE['greedy2']
    elif 'regression' in bandit_name:
        return PALETTE['regression']
    else:
        raise ValueError(bandit_name)

OOD_PALETTE = {o: get_color_ood(o) for o in OOD_List}
OOD05 = ['greedy1_ood05', 'greedy2_ood05', 'regression_ood05']
OOD1 = ['greedy1_ood1', 'greedy2_ood1', 'regression_ood1']
OOD2 = ['greedy1_ood2', 'greedy2_ood2', 'regression_ood2']
FILENAMES_OOD = {"greedy1_ood1" :"greedy_LVM1_regret-ood1.pkl",
            "greedy1_ood2" :"greedy_LVM1_regret-ood2.pkl",
            "greedy1_ood05" :"greedy_LVM1_regret-ood05.pkl",
            "greedy2_ood1" :"greedy_LVM2_regret-ood1.pkl",
            "greedy2_ood2" :"greedy_LVM2_regret-ood2.pkl",
            "greedy2_ood05" :"greedy_LVM2_regret-ood05.pkl",
            "regression_ood1" :"regression_regret-ood1.pkl",
            "regression_ood2" :"regression_regret-ood2.pkl",
            "regression_ood05" :"regression_regret-ood05.pkl",
            }
NAMES_OOD = {"greedy1_ood1" : "CPG", # $\Delta z=2$",
            "greedy1_ood2" : "CPG",# $\Delta z=4$",
            "greedy1_ood05" : "CPG", # $\Delta z=1$
            "greedy2_ood1" : "FPG",# $\Delta z=2$",
            "greedy2_ood2" : "FPG",# $\Delta z=4$",
            "greedy2_ood05" : "FPG", # $\Delta z=1$
            "regression_ood1" : "Reg",# $\Delta z=2$",
            "regression_ood2" : "Reg",# $\Delta z=4$",
            "regression_ood05" : "Reg", #$\Delta z=1$
            }

def barplot_ood(T=500):
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    regret_dict = {}
    banditdir = OOD
    def _unpickle_after_T(bandit):
        return np.mean(load_pickle(os.path.join(banditdir, FILENAMES_OOD[bandit]))[:,T:], axis=1)
    for bandit in OOD05:
        regret_dict[bandit] = _unpickle_after_T(bandit)
    for bandit in OOD1:
        regret_dict[bandit] = _unpickle_after_T(bandit)
    for bandit in OOD2:
        regret_dict[bandit] = _unpickle_after_T(bandit)
    list_algos = [*OOD05, *OOD1, *OOD2]
    ax.bar(x=list_algos, height=[regret_dict[b] for b in list_algos],
                palette=[OOD_PALETTE[b] for b in list_algos],
                errorbar=('se', 2))
    plt.xticks(ticks=np.arange(len(list_algos)), labels=[NAMES_OOD[b] for b in list_algos], rotation=90)
    # set axis lims and labels
    #ax.set_xlim([0, len(regret_dict[bandit].T)])
    ax.set_ylim([0, 0.15])
    # set fontsize ticks
    ax.tick_params(axis="both", which="major")
    # set legend size
    ax.set_ylabel(f"Exp Regret, T={T}")
    # Move legend in ax[0] to upper right corner
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(banditdir, 'OOD-barplot.pdf'), dpi=300)
    plt.close()


def plot_ood_onebyone():
    matplotlib.rc_file('matplotlibrc')
    banditdir = OOD
    regret_dict = {}
    for idx, ood in enumerate([OOD05, OOD1, OOD2]):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for bandit in ood:
            regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES_OOD[bandit]))
            plot(np.cumsum(regret_dict[bandit], axis=1), NAMES_OOD[bandit], ax, colour=OOD_PALETTE[bandit], errorbar=("se", 2))
        # set axis lims and labels
        ax.set_xlim([0, len(regret_dict[bandit].T)])
        ax.set_ylim([0, 200])
        # set fontsize ticks
        ax.tick_params(axis="both", which="major")
        ax.tick_params(axis="y", which="major", bottom=False)
        # set legend size
        ax.legend(loc="upper right")
        ax.set_xlabel(TIME)
        if idx == 0:
            ax.set_ylabel(REGRET)
        # Title
        #ax[0].set_title("$\Delta z=1$")
        #ax[1].set_title("$\Delta z=2$")
        #ax[2].set_title("$\Delta z=4$")
        # Move legend in ax[0] to upper right corner
        #ax[1].legend(loc="upper right")
        #ax[2].legend(loc="upper right")
        # pdf crop
        oodidx= {0:'05', 1:'1', 2:'2'}[idx]
        fig.tight_layout()
        fig.savefig(os.path.join(DIR, f'OOD-{oodidx}.pdf'), dpi=300)
        plt.close()
        plt.clf()


def plot_ood2():
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    regret_dict = {}
    banditdir = OOD
    for bandit in OOD05:
        regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES_OOD[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1), NAMES_OOD[bandit], ax[0], colour=OOD_PALETTE[bandit], errorbar=("se", 2))
    for bandit in OOD1:
        regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES_OOD[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1), NAMES_OOD[bandit], ax[1], colour=OOD_PALETTE[bandit], errorbar=("se", 2))
    for bandit in OOD2:
        regret_dict[bandit] = load_pickle(os.path.join(banditdir, FILENAMES_OOD[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1), NAMES_OOD[bandit], ax[2], colour=OOD_PALETTE[bandit], errorbar=("se", 2))
    # set axis lims and labels
    ax[0].set_xlim([0, len(regret_dict[bandit].T)])
    ax[1].set_xlim([0, len(regret_dict[bandit].T)])
    ax[2].set_xlim([0, len(regret_dict[bandit].T)])
    ax[0].set_ylim([0, 200])
    ax[1].set_ylim([0, 200])
    ax[2].set_ylim([0, 200])
    # set fontsize ticks
    ax[0].tick_params(axis="both", which="major")
    ax[1].yticks = []
    ax[2].tick_params(axis="y", which="major", bottom=False)
    # set legend size
    ax[0].legend()
    ax[1].legend().remove()
    ax[2].legend().remove()
    ax[0].set_xlabel(TIME + " - $\Delta z=1$")
    ax[1].set_xlabel(TIME + " - $\Delta z=2$")
    ax[2].set_xlabel(TIME + " - $\Delta z=4$")
    ax[0].set_ylabel(REGRET)
    # Title
    #ax[0].set_title("$\Delta z=1$")
    #ax[1].set_title("$\Delta z=2$")
    #ax[2].set_title("$\Delta z=4$")
    # Move legend in ax[0] to upper right corner
    ax[0].legend(loc="upper right")
    #ax[1].legend(loc="upper right")
    #ax[2].legend(loc="upper right")
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, 'OOD-triple.pdf'), dpi=300)
    plt.close()


#plot_ood_onebyone()
#plot_ood2()

### Plot for Multi-ARM Experiment
MARM_Path = ""
MARM_Arms = [10, 20, 50]
MARM_Bandits = ['mab', 'greedy1', 'greedy2', 'greedy1_oracle', 'greedy2_oracle']


def get_MARM_filename(bandit, arm):
    return FILENAMES[bandit].removesuffix('.pkl') + f"-{arm}.pkl"


def get_MARM_name(bandit, arm):
    return NAMES[bandit] + f" {arm} Arms"


def plot_MARM():
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    regret_dict = {}
    for idx, arm in enumerate(MARM_Arms):
        for bandit in MARM_Bandits:
            regret_dict[bandit] = load_pickle(os.path.join(MARM_Path, get_MARM_filename(bandit, arm)))
            plot(np.cumsum(regret_dict[bandit], axis=1), get_MARM_name(bandit, arm), ax[idx], colour=PALETTE[bandit], errorbar=("se", 2), zorder=Z_ORDER[bandit])
    # set axis lims and labels
    ax[0].set_xlim([0, len(regret_dict[bandit].T)])
    ax[1].set_xlim([0, len(regret_dict[bandit].T)])
    ax[2].set_xlim([0, len(regret_dict[bandit].T)])
    ax[0].set_ylim([0, 200])
    ax[1].set_ylim([0, 200])
    ax[2].set_ylim([0, 200])
    # set fontsize ticks
    ax[0].tick_params(axis="both", which="major")
    ax[1].tick_params(axis="both", which="major")
    ax[2].tick_params(axis="both", which="major")
    # set legend size
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[1].set_xlabel(TIME)
    ax[0].set_ylabel(REGRET)
    # Drop legend from ax[1]
    #ax[1].legend().remove()
    #ax[2].legend().remove()
    # Move legend in ax[0] to upper right corner
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(MARM_Path, 'Results-Multiarm.pdf'), dpi=300)
    plt.close()


def plot_MARM2(T=50, ylim=100):
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    regret_dict = {}
    bar_width = 0.25
    alpha = 0.8
    x_labels = ['10', '20', ' 50']
    n_methods = len(x_labels)

    def _unpickle_after_T(bandit):
        return np.sum(load_pickle(os.path.join(MARM_Path, get_MARM_filename(bandit, arm)))[:,T:], axis=1)

    for idx, arm in enumerate(MARM_Arms):
        for bandit in MARM_Bandits:
            regret_dict[bandit + "_" + str(arm) ] = _unpickle_after_T(bandit)
 
    # Generate x locations
    x = np.arange(len(MARM_Bandits)) * (n_methods + 1) * bar_width
    hatches = ['', '\\\\', '/',]
    #linestyles = ['-', '--', '-.']
    for i, bandit in enumerate(MARM_Bandits):
        margin = bar_width + 0# 0.05
        positions = x[i] + np.arange(n_methods) * margin
        for j in range(3):
            regret = regret_dict[bandit + "_" + str(MARM_Arms[j])]
            n = np.shape(regret)[0]
            mean = np.mean(regret)
            se = np.std(regret) / np.sqrt(n)
            ax.bar(positions[j], mean, yerr=se, capsize=5,
                color=PALETTE[bandit], edgecolor='black', width=bar_width, zorder = 3,
                hatch=hatches[j], alpha=alpha)

    # Set labels and ticks
    ax.set_ylim([0, ylim])
    ax.set_ylabel(f'$Reg_T$ After $T={T}$ Rounds')
    #if show_title:
    #    ax.set_title(f"{frac[index]*100:.1f}% of data ({int(n_samples[index])} samples)")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([NAMES[bandit] for bandit in MARM_Bandits], rotation=0, fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    style_legend = [Patch(facecolor='gray', edgecolor='black', hatch=h, label=l) for h, l in zip(hatches, x_labels)]
    style_legend[-1].set_alpha(0.4)
    lgd = plt.legend(handles=style_legend, loc='upper center', ncols=3, columnspacing=0.5, fontsize=14, markerscale=20, handlelength=2.5)
    plt.tight_layout()
    plt.savefig(os.path.join(MARM_Path, "MARM_barplot.pdf"), format="pdf", bbox_inches="tight")


################################################
# Plot for Nonlinear
################################################
NONLINEAR_DIR = ""
Nonlinear_list = ["mab", "greedy1", "greedy1_VAE", "regression"]
#cum_regret_plot(NONLINEAR_DIR, Nonlinear_list, y_lim=125, figsize=(7.5, 5), savepath=os.path.join(DIR, "Results-NonLinear.pdf"))


################################################
# Plot for Noise in Emission
################################################
EMISSION1 = ""
EMISSION05 = ""
EMISSION025 = ""
emissionDICT = {EMISSION025: '025', EMISSION1: '1', EMISSION05: '05'}

def plot_emission2(emission):
    # Plot seperately
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    regret_dict = {}
    bandit_list = ['mab', 'greedy1', 'greedy2', 'projected_thompson', 'greedy1_VAE', 'greedy2_VAE', 'greedy1_oracle', 'greedy2_oracle', 'regression', 'mab_prior']
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(emission, FILENAMES[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1), label=NAMES[bandit], ax=ax, colour=PALETTE[bandit])
    ax.set_xlim([0, len(regret_dict[bandit].T)])
    ax.set_ylim([0, 150])
    ax.tick_params(axis="both", which="major")
    ax.legend(loc="upper right", fontsize='x-small')
    ax.set_xlabel(TIME)
    if emission == EMISSION025:
        ax.set_ylabel(REGRET)
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, f'Emission-triple-{emissionDICT[emission]}.pdf'), dpi=300)
    plt.close()


def plot_emission():
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    regret_dict = {}
    bandit_list = ['mab', 'greedy1', 'greedy2', 'projected_thompson', 'greedy1_VAE', 'greedy2_VAE', 'greedy1_oracle', 'greedy2_oracle', 'regression', 'mab_prior']
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(EMISSION025, FILENAMES[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1), label=NAMES[bandit], ax=ax[0], colour=PALETTE[bandit])
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(EMISSION05, FILENAMES[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1), label=NAMES[bandit], ax=ax[1], colour=PALETTE[bandit])
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(EMISSION1, FILENAMES[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1), label=NAMES[bandit], ax=ax[2], colour=PALETTE[bandit])
    # set axis lims and labels
    ax[0].set_xlim([0, len(regret_dict[bandit].T)])
    ax[1].set_xlim([0, len(regret_dict[bandit].T)])
    ax[2].set_xlim([0, len(regret_dict[bandit].T)])
    ax[0].set_ylim([0, 150])
    ax[1].set_ylim([0, 150])
    ax[2].set_ylim([0, 150])
    # set fontsize ticks
    ax[0].tick_params(axis="both", which="major")
    ax[1].tick_params(axis="both", which="major")
    ax[2].tick_params(axis="both", which="major")
    #ax[1].set_yticks([])
    #ax[2].set_yticks([])
    # set legend size
    ax[0].legend()
    ax[1].legend().remove()
    ax[2].legend().remove()
    ax[0].set_xlabel("$\sigma=0.25$", fontsize=18)
    ax[1].set_xlabel(TIME + " - $\sigma=0.5$")
    ax[2].set_xlabel("$\sigma=1$")
    ax[0].set_ylabel(REGRET)
    # Title
    #fig.suptitle('Comparison of baselines with gradual Gaussian noise in emission function $g$:', fontsize=20)
    #ax[0].set_title("$\sigma = 0.25$")
    #ax[1].set_title("$\sigma=0.5$")
    #ax[2].set_title("$\sigma=1$")
    # Move legend in ax[0] to upper right corner
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, 'Emission-triple.pdf'), dpi=300)
    plt.close()


#plot_emission2(EMISSION025)
#plot_emission2(EMISSION05)
#plot_emission2(EMISSION1)

################################################
# Plot for ETA Exponential Family
################################################
Synthetic = ""
Laplace = ""
Uniform = ""
etaDict = {Synthetic: "Gaussian", Laplace: "Laplace", Uniform: "Uniform"}


def plot_eta_exp_family():
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    regret_dict = {}
    bandit_list = ['mab', 'greedy1', 'greedy2', 'projected_thompson', 'mab_prior']
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(Synthetic, FILENAMES[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1),  label=NAMES[bandit], ax=ax[0], colour=PALETTE[bandit])
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(Laplace, FILENAMES[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1),  label=NAMES[bandit], ax=ax[1], colour=PALETTE[bandit])
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(Uniform, FILENAMES[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1),  label=NAMES[bandit], ax=ax[2], colour=PALETTE[bandit])
    # set axis lims and labels
    ax[0].set_xlim([0, len(regret_dict[bandit].T)])
    ax[1].set_xlim([0, len(regret_dict[bandit].T)])
    ax[2].set_xlim([0, len(regret_dict[bandit].T)])
    ax[0].set_ylim([0, 200])
    ax[1].set_ylim([0, 200])
    ax[2].set_ylim([0, 200])
    # set fontsize ticks
    ax[0].tick_params(axis="both", which="major")
    ax[1].tick_params(axis="both", which="major")
    ax[2].tick_params(axis="both", which="major")
    # set legend size
    ax[0].legend()
    ax[1].legend().remove()
    ax[2].legend().remove()
    ax[0].set_xlabel("Gaussian")
    ax[1].set_xlabel(TIME + " - Laplace")
    ax[2].set_xlabel("Uniform")
    ax[0].set_ylabel(REGRET)
    # Title
    #ax[0].set_title("Gaussian")
    #ax[1].set_title("Laplace")
    #ax[2].set_title("Uniform")
    # Move legend in ax[0] to upper right corner
    ax[0].legend(loc="upper right")
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, 'Eta-Exponential-triple.pdf'), dpi=300)
    plt.close()


def plot_eta2(dist, figsize=(5,5)):
    # Plot seperately, dist Synthetic, Laplace or Uniform
    matplotlib.rc_file('matplotlibrc')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    regret_dict = {}
    bandit_list = ['mab', 'greedy1', 'greedy2', 'projected_thompson', 'mab_prior']
    for bandit in bandit_list:
        regret_dict[bandit] = load_pickle(os.path.join(dist, FILENAMES[bandit]))
        plot(np.cumsum(regret_dict[bandit], axis=1),  label=NAMES[bandit], ax=ax, colour=PALETTE[bandit])
    ax.set_xlim([0, len(regret_dict[bandit].T)])
    ax.set_ylim([0, 200])
    ax.tick_params(axis="both", which="major")
    ax.legend(loc="upper right", fontsize='x-small')
    ax.set_xlabel(TIME)
    #ax.set_ylabel(REGRET)
    # pdf crop
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, f'Dist-triple-{etaDict[dist]}.pdf'), dpi=300)
    plt.close()

#plot_eta2(Synthetic)
#plot_eta2(Laplace)
#plot_eta2(Uniform)