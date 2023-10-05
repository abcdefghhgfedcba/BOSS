This is official code for paper: "SENSITIVITY-INFORMED REGULARIZATION FOR OFFLINE BLACK-BOX OPTIMIZATION"
# Abstract 
Offline optimization is an important task in numerous material engineering domains where online experimentation to collect data is too expensive and needs to be replaced by an in silico maximization of a surrogate of the black-box function. Although such a surrogate can be learned from offline data, its prediction might not be reliable outside the offline data regime, which happens when the surrogate has narrow prediction margin and is (therefore) sensitive to small perturbations of its parameterization. This raises the following questions: (1) how to regulate the sensitivity of a surrogate model; and (2) whether conditioning an offline optimizer with such less sensitive surrogate will lead to better optimization performance. To address these questions, we develop an optimizable sensitivity measurement for the surrogate model, which then inspires a sensitivity-informed regularizer that is applicable to a wide range of offline optimizers. This development is both orthogonal and synergistic to prior research on offline optimization, which is demonstrated in our extensive experiment benchmark.

This repo is builts on original repository: https://github.com/brandontrabucco/design-baselines

# Quick installation
This guide is to install the environment, original from https://github.com/kaist-silab/design-baselines-fixes.

After installing `conda`, you may run the following script to install the benchmark:

```bash
bash install.sh
```

you may also choose whether to install Mujoco or not via the script.

# Run code
Baselines: run base files in 'SIRO/scripts/baseline-scripts'

SIRO: run base files in 'SIRO/scripts/robust-scripts'

Note that: Running files in 'SIRO/scripts/baseline-scripts' will store the correspoding task in results folder. Then fill path of these tasks into "path_origin" in script files in 'SIRO/scripts/robust-scripts' to run baselins + SIRO.