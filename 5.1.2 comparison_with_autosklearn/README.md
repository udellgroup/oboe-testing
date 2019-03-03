# Usage

This folder contains scripts for reproducing our experimental results.

1. Oboe with experiment design or random initialization
Code and manual for getting oboe results is in the `oboe` folder.

2. auto-sklearn
Code and manual for getting auto-sklearn results is in the `auto-sklearn` folder.

3. plotting
After specifying configurations at the beginning of `plot_results.py`,
```
python plot_results.py <path_to_oboe_results> <path_to_autosklearn_results> <path_to_random_initialization_results>
```
will generate plots as shown in Figure 6 of our paper.


