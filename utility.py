import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.linalg import eigh


def generate_samples(mean_vectors, cov_matrices, n=10000, priors=None):
    if priors is None:
        priors = [1 / len(mean_vectors)] * len(mean_vectors)

    d = mean_vectors[0].shape[0]
    labels = np.random.choice(len(mean_vectors), size=n, p=priors)
    all_samples = np.zeros((n, d))

    rng = np.random.default_rng()
    for i in range(len(mean_vectors)):
        class_indices = np.where(labels == i)[0]
        class_samples = rng.multivariate_normal(mean_vectors[i], cov_matrices[i], size=len(class_indices))
        all_samples[class_indices] = class_samples

    df = pd.DataFrame(all_samples, columns=[f'feature_{j + 1}' for j in range(d)])
    df['True Class Label'] = labels
    return df


def discriminant_scores(features, mean_vectors, cov_matrices):
    p0 = multivariate_normal.pdf(features, mean_vectors[0], cov_matrices[0])
    p1 = multivariate_normal.pdf(features, mean_vectors[1], cov_matrices[1])

    p0 = np.clip(p0, a_min=1e-300, a_max=None)
    ratio = p1 / p0
    return ratio


def naive_bayes_discriminant_scores(features, mean_vectors, diag_covs=None):
    n, d = features.shape
    if diag_covs is None:
        diag_covs = [np.eye(d), np.eye(d)]
    else:
        diag_covs = [np.diag(np.diag(c)) for c in diag_covs]

    p0 = multivariate_normal.pdf(features, mean_vectors[0], diag_covs[0])
    p1 = multivariate_normal.pdf(features, mean_vectors[1], diag_covs[1])

    p0 = np.clip(p0, a_min=1e-300, a_max=None)
    ratio = p1 / p0
    return ratio


def lda_discriminant_scores(features, w_LDA):
    return np.dot(features, w_LDA)


def estimate_parameters(data):
    classes = np.unique(data['True Class Label'])
    estimated_means = []
    estimated_covs = []
    for c in classes:
        class_data = data[data['True Class Label'] == c].iloc[:, :-1]
        estimated_means.append(class_data.mean().values)
        estimated_covs.append(class_data.cov().values)
    return estimated_means, estimated_covs


def fisher_lda(estimated_means, estimated_covs):
    m0, m1 = estimated_means
    sb = np.outer(m1 - m0, m1 - m0)
    sw = estimated_covs[0] + estimated_covs[1]

    vals, vecs = eigh(sb, sw)
    w_LDA = vecs[:, -1]
    return w_LDA
