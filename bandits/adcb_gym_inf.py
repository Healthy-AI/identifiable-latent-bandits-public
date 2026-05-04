import numpy as np
import pandas as pd

from bandits.ADCB import config
from bandits.ADCB import data_models as dm
from bandits.ADCB import treatments as tr
from bandits.ADCB import autoregression as ar
from bandits.environment import Environment
from utils import load_pickle, read_config

ADCB_CONFIG = "configs/data_configs/adcb_config.yaml"

class ADCBGym(Environment):
    def __init__(self, data_path, reward_sigma=0.5, policy="Random", **kwargs):
        """
        Initializes the environment

        args
        data_path: str
            path to the data
        """
        self.data = pd.read_csv(data_path)
        self.reward_sigma = reward_sigma
        config = read_config(ADCB_CONFIG)
        self.outcome_columns = config.potoutcome_cols
        self.noise_col = [config.noise_col]
        self.Z_cols = config.z_cols
        self.X_cols = ["PTMARRY"] + config.x_cols # Always includes PTMARRY
        self.t_col = config.t_col
        self.id_col = [config.id_col]
        self._xdim = len(self.X_cols) + len(config.ptmarry_cols) if config.add_ptmarry else len(self.X_cols) - 1
        self.buffer_ = None
        self.policy = policy

    @property
    def all_instances(self):
        return self.data[self.id_col[0]].unique()

    @property
    def X_dim(self):
        return len(self.X_cols) + len(config.ptmarry_cols) if config.add_ptmarry else len(self.X_cols) - 1

    @property
    def Z_dim(self):
        return len(self.Z_cols)

    def refresh(self, instance_id):
        self.get_new_instance(instance_id)

    def get_new_instance(self, instance_id):
        """
        Resets the environment and returns a randomly selected patient that can be stepped through during inference
        """
        print("Resetting environment...")
        self.done = False
        self.d = None
        self.cumulative_rewards, self.cumulative_regrets, self.expected_regrets = (
            0,
            0,
            0,
        )

        self.tt = 0  # time step returned for context
        self.rid = instance_id
        self.buffer_ = self.data.groupby(self.id_col).get_group(self.rid)

    def step(self, instance_id=None):
        """
        Args:
            instance_id: Redundant, only pass None.
        Returns:
            id (int): patient id
            X (dict): patient features
            Z (dict): patient latent features
            t (dict): time step
        """
        if self.buffer_.shape[0] > 0:
            # self.reset()
            # self.done = True
            self.d = self.buffer_.iloc[0:1]
            self.buffer_ = self.buffer_.iloc[1:]
        else:
            self.d = self._one_step(self.tt, self.rid)

        assert self.d[self.t_col].values[0] == self.tt
        self.tt += 1

        return (
            self.d[self.id_col].squeeze(),
            {
                key: value[0]
                for key, value in self.d[self.X_cols].to_dict(orient="list").items()
            },
            {
                key: value[0]
                for key, value in self.d[self.Z_cols].to_dict(orient="list").items()
            },
            self.d[self.t_col].squeeze(),
        )

    def action(self, action):
        """
        Plays an action, returns a reward
        Returns:
            reward (float): reward
            regret (float): regret
            expected_rewards (float): expected rewards
            expected_regrets (float): expected regrets
            outcomes (dict): counterfactual patient outcomes
        """
        assert self.d[self.t_col].values[0] == self.tt - 1
        outcomes = self.d[self.outcome_columns]
        outcomes = np.array(
            [outcomes["Y_" + str(a)].values[0] for a in range(8)], dtype="float64"
        )

        rewards = np.array([(outcomes[a]) for a in range(8)])
        r = rewards[action]
        regret = np.max(rewards) - rewards[action]

        self.cumulative_rewards += r
        self.cumulative_regrets += regret
        self.expected_regrets = self.cumulative_regrets / (self.tt + 1)

        return {
            "reward": r,
            "regret": regret,
            "cumulative_rewards": self.cumulative_rewards,
            "cumulative_regrets": self.cumulative_regrets,
            "expected_regrets": self.expected_regrets,
            "outcomes": outcomes,
        }

    def load_cov_model(self, cov, autoreg=False):
        model = None
        model_path = config.data_path + "models/" + cov + "_model.pkl"
        if autoreg:
            model_path = config.data_path + "models/" + cov + "_autoreg_model.pkl"

        model = load_pickle(model_path)
        return model

    def _one_step(
        self,
        num_repetition,
        id,
    ):
        gen_data_bl = self.data[self.data["RID"] == id]
        bl_dict = {
            "RID": gen_data_bl.RID,
            "AGE": gen_data_bl.AGE,
            "PTETHCAT": gen_data_bl.PTETHCAT,
            "PTRACCAT": gen_data_bl.PTRACCAT,
            "PTGENDER": gen_data_bl.PTGENDER,
            "APOE4": gen_data_bl.APOE4,
            "PTEDUCAT": gen_data_bl.PTEDUCAT,
            "PTMARRY": gen_data_bl.PTMARRY,
            "VISCODE": gen_data_bl.VISCODE,
            "ABETARatio": gen_data_bl.ABETARatio,
        }
        gen_data_auto = pd.DataFrame(bl_dict)
        np.random.seed(num_repetition)

        lrTAU = self.load_cov_model("TAU", autoreg=False)
        lrPTAU = self.load_cov_model("PTAU", autoreg=False)
        lrFDG = self.load_cov_model("FDG", autoreg=False)
        lrAV45 = self.load_cov_model("AV45", autoreg=False)

        gen_data_auto["ABetaRatioEpsilon"] = np.random.normal(
            loc=gen_data_auto["ABETARatio"], scale=config.z_noise
        )
        pred_df = gen_data_auto[
            [
                "PTETHCAT",
                "PTRACCAT",
                "PTGENDER",
                "ABetaRatioEpsilon",
                "APOE4",
                "AGE",
            ]
        ].rename(columns={"ABetaRatioEpsilon": "ABETARatio"})
        TAU = lrTAU.predict(
            dm.check_categorical(
                pd.DataFrame(pred_df), config.all_pred_cols["TAU"], "TAU"
            )
        )
        TAU = ar.check_range(TAU[0])
        gen_data_auto["TAU"] = TAU

        PTAU = lrPTAU.predict(
            dm.check_categorical(
                pd.DataFrame(pred_df), config.all_pred_cols["PTAU"], "PTAU"
            )
        )

        PTAU = ar.check_range(PTAU[0])
        gen_data_auto["PTAU"] = PTAU

        pred_df = gen_data_auto[
            [
                "PTETHCAT",
                "PTRACCAT",
                "ABetaRatioEpsilon",
                "TAU",
                "PTAU",
                "APOE4",
            ]
        ].rename(columns={"ABetaRatioEpsilon": "ABETARatio"})
        FDG = lrFDG.predict(
            dm.check_categorical(
                pd.DataFrame(pred_df), config.all_pred_cols["FDG"], "FDG"
            )
        )

        FDG = ar.check_range(FDG[0])
        gen_data_auto["FDG"] = FDG

        AV45 = lrAV45.predict(
            dm.check_categorical(
                pd.DataFrame(pred_df), config.all_pred_cols["AV45"], "AV45"
            )
        )

        AV45 = ar.check_range(AV45[0])
        gen_data_auto["AV45"] = AV45

        if self.policy == "Random":
            gen_data_auto["A"] = np.random.choice(list(range(8)), 1)

        res_ = np.array(
            gen_data_auto.apply(
                lambda x: tr.assign_treatment_effect_continuous_no_noise(
                    a=x["A"],
                    z=x["ABETARatio"],
                    apoe4=x["APOE4"],
                    #ATE_noise=self.reward_sigma,
                ),
                axis=1,
            )
        )

        gen_data_auto[["Y_hat", "Delta_noise"]] = pd.DataFrame(
            res_.tolist(), index=gen_data_auto.index
        )

        gen_data_auto["Y_hat"] = gen_data_auto.apply(lambda x: 0 + x["Y_hat"], axis=1)
        # print(gen_data_auto["Y_hat"].head())
        # gen_data_auto = gen_data_auto.drop(columns=["Delta_noise"])
        YY = pd.DataFrame(
            np.array(
                list(
                    gen_data_auto.apply(
                        lambda x: ar.gen_potential_Outcomes(
                            observed_a=x["A"],
                            Y_hat=x["Y_hat"],
                            ABetaRatio=x["ABETARatio"],
                            #ATE_noise=self.reward_sigma,
                            # APOE4=x["APOE4"],
                        ),
                        axis=1,
                    )
                )
            ),
            columns=["Y_0", "Y_1", "Y_2", "Y_3", "Y_4", "Y_5", "Y_6", "Y_7"],
        )

        gen_data_auto["Y_0"] = YY["Y_0"].values
        gen_data_auto["Y_1"] = YY["Y_1"].values
        gen_data_auto["Y_2"] = YY["Y_2"].values
        gen_data_auto["Y_3"] = YY["Y_3"].values
        gen_data_auto["Y_4"] = YY["Y_4"].values
        gen_data_auto["Y_5"] = YY["Y_5"].values
        gen_data_auto["Y_6"] = YY["Y_6"].values
        gen_data_auto["Y_7"] = YY["Y_7"].values

        gen_data_auto["VISCODE"] = self.tt

        # Negate the columns
        columns_to_negate = ["Y_hat"] + self.outcome_columns
        # print("Negating columns: ", columns_to_negate)
        # print(gen_data_auto[columns_to_negate].head())
        # print(gen_data_auto["Delta_noise"].values[0])
        gen_data_auto[columns_to_negate] = gen_data_auto[
            columns_to_negate
        ] + gen_data_auto["Delta_noise"].values.reshape(-1, 1)
        gen_data_auto[columns_to_negate] = -gen_data_auto[columns_to_negate]

        return gen_data_auto
