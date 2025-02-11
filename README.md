# Arousal dynamics codebase
Code supporting all analyses in the preprint ["**Arousal as a universal embedding for spatiotemporal brain dynamics**"](https://doi.org/10.1101/2023.11.06.565918). Jupyter notebooks for reproducing all paper analyses and figures are available in the notebooks directory.

## Overview
* The main pupil-widefield modeling pipeline is contained in **pupil_modeling_group.ipynb**
* After running this notebook, **r2_plots.ipynb** computes various $R^2$ plots.
  * Principal components are computed in the **descriptive_analyses.ipynb** notebook, which can be run first.
* **GMM**, **HMM**, and **DMD** analyses can be performed on the pupil-reconstructed data by running the respective notebooks after running the **pupil_modeling_group.ipynb** notebook.
* See **pupil_modeling_indv.ipynb** for the invididual, within-mouse modeling pipeline.
* See **lorenz_example.ipynb** for a toy demonstration of the VAE+delay embedding approach on a stochastic Lorenz simulation.
* For Allen analysis, run in sequence **allen_compute_observables.ipynb**, **allen_embedding.ipynb**, and **allen_WF_transferVAE.ipynb**
