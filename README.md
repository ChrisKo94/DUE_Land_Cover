# Deterministic Uncertainty Estimation (DUE) for Land Cover Classification

---

This repository is linked to the GRSL special stream on "Uncertainty-Aware and Robust Machine Learning for Remote Sensing" submission "Can Land Cover Classification Models Benefit
From Distance-Aware Architectures?". The methodology is largely based on the works of [1] and [2], and the repository closely follows https://github.com/y0ast/DUE, which is the official repository of [2]. 


The individual python executable files are structured as follows:
- Blah
---

## Requirements

---

`conda create -y --name dueenv python=3.9`

`conda env create -f environment.yml`

## Training

---

All configs: `python train.py `

---



### References

[1] J. Liu, Z. Lin, S. Padhy, D. Tran, T. Bedrax Weiss, and B. Lakshminarayanan, “Simple and principled uncertainty estimation with
deterministic deep learning via distance awareness,” NeurIPS, 2020.

[2] J. van Amersfoort, L. Smith, A. Jesson, O. Key, and Y. Gal, “On feature
collapse and deep kernel learning for single forward pass uncertainty,”
arXiv preprint arXiv:2102.11409, 2021.