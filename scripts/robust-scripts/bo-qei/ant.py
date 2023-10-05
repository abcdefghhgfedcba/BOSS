from design_baselines.utils import save_object, load_object
import tensorflow as tf
from design_baselines.logger import Logger
import os
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1" 
import glob

path_origin = 'results/baselines/bo-qei-ant/bo_qei_2023-08-05_01-35-11'

for omega_mu_bound in [1e-3]:
      for omega_sigma_init in [1e-3]:
            for folder in os.listdir(path_origin):
                if os.path.isdir(os.path.join(path_origin, folder)):
                    path_hacked_dat = os.path.join(os.path.join(path_origin, folder), 'data/hacked.dat')
                    os.system(f'CUDA_VISIBLE_DEVICES=0 bo-qei-robust ant --local-dir results/SIRO/bo-qei-robust/ant --cpus 5 \
                        --gpus 1 \
                        --num-parallel 1 \
                        --num-samples 1 \
                        --omega_mu_init 0. \
                        --omega_sigma_init {omega_sigma_init} \
                        --omega_mu_bound {omega_mu_bound} \
                        --omega_sigma_lower 1e-5 \
                        --omega_sigma_upper 1e-2 \
                        --lambda_ 1e-3 \
                        --n_gamma 100 \
                        --lr_omega 1e-2 \
                        --alpha 0.1 \
                        --path_hacked_task {path_hacked_dat}')
                    