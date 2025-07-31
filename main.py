import os
import pandas as pd
from torchvision.models import resnet18, ResNet18_Weights

import morfse_trainer
import baseline_trainer

def main(exp_id):
    for iter in range(5):
        gating_model = resnet18
        expert1_model = resnet18
        expert2_model = resnet18
        baseline_model = resnet18
        
        gating_weight = ResNet18_Weights.IMAGENET1K_V1
        expert1_weight = ResNet18_Weights.IMAGENET1K_V1
        expert2_weight = ResNet18_Weights.IMAGENET1K_V1
        baseline_weight = ResNet18_Weights.IMAGENET1K_V1
        
        for fold in range(5):
            print(f"exp_id: {exp_id}, fold: {fold}")
            
            morfse_trainer.train(
                network="gating", 
                num_exp_id=exp_id, 
                num_batch_size=64, 
                num_epochs=30, 
                num_test_fold=fold,
                model=gating_model,
                weights=gating_weight,
                lr=1e-5,
                weight_decay=0,
                num_gating_class=2,
                num_expert_class=3,
                output_dir="../../result"
                )
            morfse_trainer.train(
                network='expert1', 
                num_exp_id=exp_id, 
                num_batch_size=64, 
                num_epochs=30, 
                num_test_fold=fold,
                model=expert1_model,
                weights=expert1_weight,
                lr=1e-5,
                weight_decay=0,
                num_gating_class=2,
                num_expert_class=3,
                output_dir="../../result"
                )
            morfse_trainer.train(
                network='expert2', 
                num_exp_id=exp_id, 
                num_batch_size=64, 
                num_epochs=30, 
                num_test_fold=fold,
                model=expert2_model,
                weights=expert2_weight,
                lr=1e-5,
                weight_decay=0,
                num_gating_class=2,
                num_expert_class=3,
                output_dir="../../result"
                )
            morfse_trainer.test(
                num_exp_id=exp_id, 
                num_batch_size=64, 
                num_test_fold=fold,
                gating_model=gating_model,
                expert1_model=expert1_model,
                expert2_model=expert2_model,
                weights=None,
                num_gating_class=2,
                num_expert_class=3,
                output_dir="../../result"
                )
            
            baseline_trainer.train(
                num_exp_id=exp_id, 
                num_batch_size=64, 
                num_epochs=30, 
                num_test_fold=fold,
                model=baseline_model,
                weights=baseline_weight,
                lr=1e-5,
                weight_decay=0,
                num_class=3,
                output_dir="../../result"
                )
            baseline_trainer.test(
                num_exp_id=exp_id, 
                num_batch_size=64, 
                num_test_fold=fold,
                model=baseline_model,
                weights=None,
                num_class=3,
                output_dir="../../result"
                )

        exp_id += 1


if __name__ == "__main__":
    start_exp_id = 101
    main(start_exp_id)