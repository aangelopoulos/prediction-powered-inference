<h1 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Prediction-Powered Inference</h1>
<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">confidence intervals and p-values powered by machine learning algorithms</h3>
<p align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, Tijana Zrnic</p>

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
    <a style="text-decoration:none !important;" href="https://twitter.com/ml_angelopoulos?ref_src=twsrc%5Etfw" alt="package management"><img src="https://img.shields.io/twitter/follow/ml_angelopoulos?style=social" /></a>
    <a style="text-decoration:none !important;" href="https://twitter.com/stats_stephen" alt="package management"><img src="https://img.shields.io/twitter/follow/stats_stephen?style=social" /></a>
    <a style="text-decoration:none !important;" href="https://twitter.com/stats_stephen" alt="package management"><img src="https://img.shields.io/twitter/follow/seafann?style=social" /></a>
</p>

<p>
This repository contains code for prediction-powered inference --- a framework for constructing confidence intervals and p-values when using predictions from a machine learning model.
The main algorithms are in `ppi.py`.  
Each subfolder prediction-powered inference to a real prediction problem in proteomics, genomics, electronic voting, remote sensing, census analysis, and ecology.
</p>

<p align="center"> <b>The notebooks are easy to run!</b></p>
<p>
You can test and develop prediction-powered inference strategies entirely in this sandbox, locally on your laptop. Open a notebook to see the expected output. You can use these notebooks to experiment with existing methods or as templates to develop your own. 
</p>
<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Example notebooks</h3>
<ul>
    <li><a href="https://github.com/aangelopoulos/prediction-powered-inference/blob/main/alphafold/odds-ratio.ipynb"><code>alphafold/odds-ratio.ipynb</code></a>: Measuring the assotiation between phosphorylation and intrinsically disordered regions using Alphafold.</li>
    <li><a href="https://github.com/aangelopoulos/prediction-powered-inference/blob/main/ballots/ballots.ipynb"><code>ballots/ballots.ipynb</code></a>: Calling the 2022 San Francisco Special Election between Matt Haney and David Campos using an optical voting system.</li>
    <li><a href="https://github.com/aangelopoulos/prediction-powered-inference/blob/main/census/ols.ipynb"><code>census/ols.ipynb</code></a>: Quantifying the relationship between age, sex, and income using XGBoost, census data, and linear regression.</li>
    <li><a href="https://github.com/aangelopoulos/prediction-powered-inference/blob/main/census/logistic.ipynb"><code>census/logistic.ipynb</code></a>: Quantifying the relationship between income and private health insurance using XGBoost, census data, and logistic regression.</li>
    <li><a href="https://github.com/aangelopoulos/prediction-powered-inference/blob/main/forest/deforestation.ipynb"><code>forest/deforestation.ipynb</code></a>: Gauging deforestation levels in the Amazon Rainforest using satellite imagery and computer vision. </li>
    <li><a href="https://github.com/aangelopoulos/prediction-powered-inference/blob/main/gene-expression/gene-expression-quantiles.ipynb"><code>gene-expression/gene-expression-quantiles.ipynb</code></a>: Analyzing the effect of promoters on gene expression using a transformer model. </li>
    <li><a href="https://github.com/aangelopoulos/prediction-powered-inference/blob/main/plankton/plankton.ipynb"><code>plankton/plankton.ipyn</code></a>: Ecological species counting for the number of plankton seen by a submersible flow cytometry system in Woods Hole oceanographic institute using a ResNet. </li>
</ul>

<p>
    To run these notebooks locally, you just need to have the correct dependencies installed and press <code>run all cells</code>! Cloning the GitHub and running the notebooks will automatically download all required data and model outputs. Code for generating the precomputed data from the raw datasets is available in each individual subfolder. There is one for each dataset. To create a <code>conda</code> environment with the correct dependencies, run <code>conda env create -f environment.yml</code>. If you still get a dependency error, make sure to activate the <code>ppi</code> environment within the Jupyter notebook.
</p>

This repository is meant to accompany our paper, the <a href="https://arxiv.org/abs/">Prediction-Powered Inference</a>.
In that paper is a detailed explanation of each example and attributions.
If you find this repository useful, in addition to the relevant methods and datasets, please cite:
</p>
<pre><code>@article{angelopoulos2023prediction,
  title={Prediction-Powered Inference},
  author={Angelopoulos, Anastasios N and Bates, Stephen and Fannjiang, Clara and Jordan, Michael I. and Zrnic, Tijana},
  journal={arXiv preprint arXiv:},
  year={2023}
}</code></pre>
