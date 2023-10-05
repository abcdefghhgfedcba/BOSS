from ray import tune
import click
import ray
import os


@click.group()
def cli():
    """A group of experiments for training Conservative Score Models
    and reproducing our ICLR 2021 results.
    """


#############


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task):
    """Evaluate MINs_mins_robust on DKittyMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": "DKittyMorphology-Exact-v0",
        "task_kwargs": {"relabel": False},
        "val_size": 200,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "base_temp": 0.1,
        "noise_std": 0.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0,
        "fake_pair_frac": 0.,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 200,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def ant(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task):
    """Evaluate MINs_mins_robust on AntMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": "AntMorphology-Exact-v0",
        "task_kwargs": {"relabel": False},
        "val_size": 200,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "base_temp": 0.1,
        "noise_std": 0.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0,
        "fake_pair_frac": 0.,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 200,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task):
    """Evaluate MINs_mins_robust on HopperController-Exact-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": "HopperController-Exact-v0",
        "task_kwargs": {"relabel": False},
        "val_size": 200,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "base_temp": 0.1,
        "noise_std": 0.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0,
        "fake_pair_frac": 0.,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 500,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
@click.option('--oracle', type=str, default="RandomForest")
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task, oracle):
    """Evaluate MINs_mins_robust on Superconductor-RandomForest-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": f"Superconductor-{oracle}-v0",
        "task_kwargs": {"relabel": False},
        "val_size": 200,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "base_temp": 0.1,
        "noise_std": 0.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0,
        "fake_pair_frac": 0.,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 200,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-chembl')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
@click.option('--assay-chembl-id', type=str, default='CHEMBL3885882')
@click.option('--standard-type', type=str, default='MCHC')
def chembl(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task,
           assay_chembl_id, standard_type):
    """Evaluate MINs_mins_robust on ChEMBL-ResNet-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": f"ChEMBL_{standard_type}_{assay_chembl_id}"
                f"_MorganFingerprint-RandomForest-v0",
        "task_kwargs": {"relabel": False,
                        "dataset_kwargs": dict(
                            assay_chembl_id=assay_chembl_id,
                            standard_type=standard_type)},
        "val_size": 100,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "base_temp": 0.1,
        "keep": 0.99,
        "start_temp": 5.0,
        "final_temp": 1.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 2048,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0.,
        "fake_pair_frac": 0.0,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 200,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
@click.option('--oracle', type=str, default="Transformer")
def gfp(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task, oracle):
    """Evaluate MINs_mins_robust on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": f"GFP-{oracle}-v0",
        "task_kwargs": {"relabel": False},
        "val_size": 200,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "base_temp": 0.1,
        "keep": 0.99,
        "start_temp": 5.0,
        "final_temp": 1.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0.,
        "fake_pair_frac": 0.0,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 200,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-tf-bind-8')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def tf_bind_8(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task):
    """Evaluate MINs_mins_robust on TFBind8-Exact-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": "TFBind8-Exact-v0",
        "task_kwargs": {"relabel": False},
        "val_size": 200,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "base_temp": 0.1,
        "keep": 0.99,
        "start_temp": 5.0,
        "final_temp": 1.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0.,
        "fake_pair_frac": 0.0,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 200,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-utr')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def utr(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task):
    """Evaluate MINs_mins_robust on UTR-ResNet-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": "UTR-ResNet-v0",
        "task_kwargs": {"relabel": False},
        "val_size": 200,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "base_temp": 0.1,
        "keep": 0.99,
        "start_temp": 5.0,
        "final_temp": 1.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0.,
        "fake_pair_frac": 0.0,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 200,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-tf_bind_10')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def tf_bind_10(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task):
    """Evaluate MINs_mins_robust on TFBind10-Exact-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": "TFBind10-Exact-v0",
        "task_kwargs": {"relabel": False, "dataset_kwargs": {"max_samples": 10000}},
        "val_size": 200,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "base_temp": 0.1,
        "keep": 0.99,
        "start_temp": 5.0,
        "final_temp": 1.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0.,
        "fake_pair_frac": 0.0,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 100,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='mins_robust-nas')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def nas(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, path_hacked_task):
    """Evaluate MINs_mins_robust on CIFARNAS-Exact-v0
    """

    # Final Version

    from design_baselines.mins_robust import mins_robust
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(mins_robust, config={
        "path_hacked_task": path_hacked_task,
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "task": "CIFARNAS-Exact-v0",
        "task_kwargs": {"relabel": False},
        "val_size": 200,
        "offline": True,
        "normalize_ys": True,
        "normalize_xs": False,
        "base_temp": 0.1,
        "keep": 0.99,
        "start_temp": 5.0,
        "final_temp": 1.0,
        "method": "wasserstein",
        "use_conv": False,
        "gan_batch_size": 128,
        "hidden_size": 1024,
        "num_layers": 1,
        "bootstraps": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "oracle_lr": 0.001,
        "oracle_batch_size": 128,
        "oracle_epochs": 100,
        "latent_size": 32,
        "critic_frequency": 10,
        "flip_frac": 0.,
        "fake_pair_frac": 0.0,
        "penalty_weight": 10.,
        "generator_lr": 2e-4,
        "generator_beta_1": 0.0,
        "generator_beta_2": 0.9,
        "discriminator_lr": 2e-4,
        "discriminator_beta_1": 0.0,
        "discriminator_beta_2": 0.9,
        "initial_epochs": 200,
        "epochs_per_iteration": 0,
        "iterations": 0,
        "exploration_samples": 0,
        "exploration_rate": 0.,
        "thompson_samples": 0,
        "solver_samples": 128, "do_evaluation": False},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})

