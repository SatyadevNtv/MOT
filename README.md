# Overview

This repository provides the necessary factory file that implements the Optimization of OT problems using Riemannian Manifold Optimization Framework.
Details can be found in our work:

Bamdev Mishra, N. T. V. Satya Dev, Hiroyuki Kasai, Pratik Jawanpuria. [Manifold Optimization for Optimal Transport.](https://arxiv.org/abs/2103.00902)

# Usage

The factory file is in compatible with the Pymanopt toolbox and can be easily integrated by placing it in the `pymanopt/manifolds/` folder (As of the Pymanopt's release `7f7a022 origin/dev` ). A helper file has been provided in the `helpers` directory explaining the usage of factory file.

For MATLAB implementation checkout `matlab` branch.

## NOTE

For GPU based acceleration (cupy), set the environment variable `USE_GPU_OPT` to `1`
