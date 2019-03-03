# Performance Comparison with Probabilistic Matrix Factorization

This folder contains code for performance comparison between Oboe and Probabilistic Matrix Factorization (PMF) on midsize OpenML datasets, based on the error matrix we have. 

PMF is an AutoML system by Microsoft Research, with a paper [Probabilistic Matrix Factorization for Automated Machine Learning](https://papers.nips.cc/paper/7595-probabilistic-matrix-factorization-for-automated-machine-learning) at NeurIPS 2018 and a [GitHub repository](https://github.com/rsheth80/pmf-automl/) for its code.

# Usage

## Type 1: Re-collect performance data, and then draw the plots in our paper
1. Collect PMF results by running `collect_pmf_performance/run_test_see_regret.ipynb`, and get results in `collect_pmf_performance/results` folder.
2. Collect Oboe results by running the three `.ipynb` files in  `collect_oboe_performance`, and get results in `collect_oboe_performance/results` folder.
3.  Run `plot.ipynb` to get Figures 5 and 12 in our paper. 

## Type 2: Draw the plots in our paper
Step 3 above.

# Acknowledgement
The code to run PMF was revised from the code in PMF's [GitHub repository](https://github.com/rsheth80/pmf-automl/).
