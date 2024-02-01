This is official code for paper: "Boosting Offline Optimizers with Surrogate Sensitivity"
# Abstract 
Offline optimization is an important task in numerous material engineering domains where online experimentation to collect data is too expensive and needs to be replaced by an in silico maximization of a surrogate of the black-box function. Although such a surrogate can be learned from offline data, its prediction might not be reliable outside the offline data regime, which happens when the surrogate has narrow prediction margin and is (therefore) sensitive to small perturbations of its parameterization. This raises the following questions: (1) how to regulate the sensitivity of a surrogate model; and (2) whether conditioning an offline optimizer with such less sensitive surrogate will lead to better optimization performance. To address these questions, we develop an optimizable sensitivity measurement for the surrogate model, which then inspires a sensitivity-informed regularizer that is applicable to a wide range of offline optimizers. This development is both orthogonal and synergistic to prior research on offline optimization, which is demonstrated in our extensive experiment benchmark.

This repo is builts on original repository: https://github.com/brandontrabucco/design-baselines

# Quick installation

To set up the environment, you may follow the instruction from  https://github.com/kaist-silab/design-baselines-fixes.

# Run code

Baselines: run files in 'BOSS/scripts/baseline-scripts'

Please be aware that, to ensure an equitable comparison, we preserve the task during the execution of baselines and subsequently execute the baseline + BOSS on the identical task. Executing files located in 'BOSS/scripts/baseline-scripts' will archive the relevant task in the results folder. Subsequently, input the paths of these tasks into the "path_origin" parameter within the script files in 'BOSS/scripts/robust-scripts' to execute the baselines + BOSS.

Baselines + BOSS: run files in 'BOSS/scripts/robust-scripts'