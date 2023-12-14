# Sparse Subspace Clustering (SSC) ADMM Python Implementation with CUDA support

This is a reimplementation of the paper [Sparse Subspace Clustering: Algorithm, Theory, and Applications](https://ieeexplore.ieee.org/document/6482137) with CUDA support.

Some of the implementation is based on the Matlab SSC-Basis Pursuit ADMM code from the [author's website](http://vision.jhu.edu/code/).

The SSC implementation in `ssc.ssc.sparse_subspace_clustering`.

The corresponding ADMM implementation in `ssc/admm.py`(CPU) and `ssc/admm_gpu.py`(CUDA).

The experiment from the original paper is in `face_clustering.py` and `motion_segmenation.py`

## Install

Please install follow the instructions in [CuPY install guide](https://docs.cupy.dev/en/stable/install.html).
According to your CUDA version, you will use `pip install cupy-cuda12x` or `pip install cupy-cuda11x`.

Other requirements are in `requirements.txt`.


## Dataset

The following datasets should be placed in the `./data` folder to reproduce the experiments.

- [`Hopkins155`](http://www.vision.jhu.edu/data/hopkins155/)
- `YaleBCrop025.mat` is in the original Matlab code.
