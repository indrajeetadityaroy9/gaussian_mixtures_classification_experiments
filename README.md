# Classification Experiments with Gaussian Mixtures

An end-to-end workflow for experimenting with 3 different classification strategies applied to a 4-dimensional random vector X drawn from a mixture of 2 Gaussian distributions to illustrate and compare performance under different modeling assumptions and methods.

A binary classification problem where the data X comes from one of two classes, each defined by its own Gaussian distribution with known or estimated parameters. The overall data distribution is:

$$
p(x) = p(x \mid L=0)P(L=0) + p(x \mid L=1)P(L=1),
$$
with class priors \( P(L=0) = 0.35 \) and \( P(L=1) = 0.65 \).

## Classification Scenarios
- **Part A:** Minimum Expected Risk (Bayes) Classification with full knowledge of true class conditional distributions (means and covariance matrices) and class priors.
- **Part B:** Naive Bayesian Classification with incorrect model assumptions consisting of class means and priors but incorrectly assumed independence among features, using only diagonal covariances.
- **Part C:** Fisher Linear Discriminant Analysis (LDA)-based Classification using estimated parameters.

## Overview

1. **Data Generation:**  
   Generates 10,000 samples from the defined mixture distribution.

2. **Classification Strategies:**
   - **Bayes Classifier (Minimum Expected Risk):** Uses the fully known parameters to classify each sample.
   - **Naive Bayesian Classifier:** Assumes a diagonal covariance matrix, ignoring the off-diagonal terms of the covariance matrices.
   - **Fisher LDA Classifier:** Estimates parameters from the data and projects onto a one-dimensional space for classification.

3. **Evaluation:**
   - Produces **ROC curves** for each classification method.
   - Estimates the **minimum probability of error** and compares performance across the three methods.