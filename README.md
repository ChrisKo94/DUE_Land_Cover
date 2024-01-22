# Deterministic Uncertainty Estimation (DUE) for Land Cover Classification

---

This repository is linked to the GRSL special stream on "Uncertainty-Aware and Robust Machine Learning for Remote Sensing" submission "Can Land Cover Classification Models Benefit
From Distance-Aware Architectures?". The methodology is largely based on the works of [1] and [2], and the repository closely follows https://github.com/y0ast/DUE, which is the official repository of [2]. 


The individual python executable files are structured as follows:
- data: Folder for storing the data
- due: 
  - layers: Individual python files for the DUE framework
- lib: 
  - datasets.py: read-in file for the data 
  - evaluate_ood.py: evaluation file for OOD detection 
  - utils.py: other utils
- runs: Folder for storing the results
- environment.yml: conda environment file
- setup: setup file for the DUE framework
- train_due.py: main file for training the DUE framework
---

## Requirements

---

`conda create -y --name dueenv python=3.9`

`conda env create -f environment.yml`

## Training

---

All configs: `python train_due.py `

Individual configs via argument parsing, e.g.: `python train_due.py --no_spectral_conv`

---



### References

[1] J. Liu, Z. Lin, S. Padhy, D. Tran, T. Bedrax Weiss, and B. Lakshminarayanan, “Simple and principled uncertainty estimation with
deterministic deep learning via distance awareness,” NeurIPS, 2020.

[2] J. van Amersfoort, L. Smith, A. Jesson, O. Key, and Y. Gal, “On feature
collapse and deep kernel learning for single forward pass uncertainty,”
arXiv preprint arXiv:2102.11409, 2021.