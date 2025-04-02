# DiffMamba: Generative Diffusion Model for Dynamic MRI Reconstruction with Temporal State Space Representation (Accepted at ISMRM 2025)

## Implementational Details
This code extends the [DiffCMR](https://github.com/xmed-lab/DiffCMR) and [Mamba](https://github.com/state-spaces/mamba) codebase. Data can be found at [CMRxRecon2023](https://cmrxrecon.github.io/Challenge.html).

## Sample Usagep
For single-coil data training and inference, simply using `python train_DiffMamba.py` and `python inference.py`,remember to set your parameters and paths in this file. For multi-coil data, using `train_DiffMamba_multisens.py` and  `inference_multicoil.py` respectively. Please ensure you have had the sensitivity map of the data.
