import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind, ttest_rel
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
from scipy.stats import norm
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

class Evaluate:
    def __init__(self, exp_id_list):
        self.exp_id_list = exp_id_list
        self.result_dir = f"../result/5times5folds/exp{exp_id_list[0]}_{exp_id_list[-1]}"
        os.makedirs(self.result_dir, exist_ok=True)

    def get_prob_of_no_gnet_morfse(self):
        for exp_id in self.exp_ids:
            for fold in range(5):
                input_path = f"../result/paper/exp{exp_id}/fold_{fold}/csv/{self.model_name}_result.csv"
                output_path = f"../result/paper/exp{exp_id}/fold_{fold}/csv/no_gnet_morfse_{self.model_name}_result.csv"

                df = pd.read_csv(input_path)
                no_gnet_morfse_bc_list = (df["expert1_bc"] + df["expert2_bc"]) / 2
                no_gnet_morfse_not_bc_list = (df["expert1_not_bc"] + df["expert2_not_bc"]) / 2

                data = {
                    "basename": df["basename"],
                    "bc": ens_bc_list,
                    "not_bc": ens_non_bc_list,
                    "true": df["true"],
                    "lesion_type": df["lesion_type"]
                }
                output_df = pd.DataFrame(data)
                output_df.to_csv(output_path, index=False)

    def make_average_prediction(self, fold):
        morfse_dfs = [
            pd.read_csv(f"../result/exp{exp_id}/fold_{fold}/csv/morfse_result.csv")
            for exp_id in self.exp_id_list
        ]
        
        no_gnet_morfse_dfs = [
            pd.read_csv(f"../result/exp{exp_id}/fold_{fold}/csv/no_gnet_morfse_result.csv")
            for exp_id in self.exp_id_list
        ]
        
        baseline_dfs = [
            pd.read_csv(f"../result/jiim/exp{exp_id}/fold_{fold}/csv/baseline_result.csv")
            for exp_id in self.exp_id_list
        ]

        # Compute averages
        morfse_ave_bc_list = []
        no_gnet_morfse_ave_bc_list = []
        baseline_ave_bc_list = []
        true_list = morfse_dfs[0]['true']
        basename_list = morfse_dfs[0]['basename']
        lesion_type_list = morfse_dfs[0]['lesion_type']

        for i in range(len(morfse_dfs[0])):
            morfse_ave_bc = np.mean([df['bc'][i] for df in morfse_dfs])
            no_gnet_morfse_ave_bc = np.mean([df['bc'][i] for df in no_gnet_morfse_dfs])
            baseline_ave_bc = np.mean([df['bc'][i] for df in baseline_dfs])

            morfse_ave_bc_list.append(morfse_ave_bc)
            ens_ave_bc_list.append(ens_ave_bc)
            baseline_ave_bc_list.append(baseline_ave_bc)

        ave_df = pd.DataFrame({
            'basename': basename_list,
            'morfse_bc': morfse_ave_bc_list,
            'no_gnet_morfse_bc': no_gnet_morfse_ave_bc_list,
            'baseline_bc': baseline_ave_bc_list,
            'true': true_list,
            'lesion_type': lesion_type_list
        })

        ave_df.to_csv(f"{self.result_dir}/fold{fold}_result.csv", index=False)
        return ave_df
    
    def concat_5folds(self):
        dfs = [pd.read_csv(f"{self.result_dir}/fold{fold}_result.csv") for fold in range(5)]
        concat_df = pd.concat(dfs)
        concat_df.to_csv(f"{self.result_dir}/all_result.csv", index=False)
        return concat_df
    
    def divide_by_finding(self):
        all_df = pd.read_csv(f"{self.result_dir}/all_result.csv")
        for finding, label in zip(["mass", "calc"], [0, 1]):
            output_path = f"{self.result_dir}/all_{finding}_result.csv"
            all_df.query(f"lesion_type == {label}").to_csv(output_path, index=False)

    def process(self):
        self.get_prob_of_no_gnet_morfse
        
        for fold in range(5):
            self.make_average_prediction(fold)
        
        self.concat_5folds()
        self.divide_by_finding()


if __name__ == "__main__":
    e = Evaluate([101, 102, 103, 104, 105])
    e.process()
