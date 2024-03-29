# Comparing EM with GD in Mixture Models of Two Components
This repository contains supplementary material for the paper accepted in UAI 2019. It also contains code in the experiment section. Please cite our paper if you want to use the code. All the programs are written in python and Jupyter Notebook.
## Supplementary material:
* **33_supplement.pdf**: supplementary material; it contains full proofs and a general conjecture
## Code - Bernoulli mixture models:
* **nbm.py**: the library containing all the functions needed for learning BMMs with EM/GD
* **test.ipynb**: basic testing for EM and GD; checks if the library is working well
* **k-cluster_ratio.ipynb**: checks if EM and GD are converging to k-cluster points, and computes the probabilities, as done in Section 6 Table 1/Table 2
* **ratio_compare.ipynb**: computes the ratio of the negative log likelihood GD/EM converges to, to the ground truth loss, as done in Section 6 Table 3/Table 4
## Code - Gaussian mixture models:
* **gaussian_mixture.ipynb**: explores the escape behavior for EM and GD in GMMs
