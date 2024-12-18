import numpy as np
from evaluation import empirical_min_error_threshold, roc_curve_and_probabilities
from utility import discriminant_scores, generate_samples, naive_bayes_discriminant_scores
from utility import estimate_parameters, fisher_lda, lda_discriminant_scores
from visualization import plot_combined_roc_2


def compute_roc_and_threshold(scores, labels, priors):
    fpr, tpr, P_D0_L1, P_D1_L0 = roc_curve_and_probabilities(scores, labels)
    empirical_gamma, empirical_min_error = empirical_min_error_threshold(scores, priors, P_D0_L1, P_D1_L0)
    return fpr, tpr, empirical_gamma, empirical_min_error


def main():
    # Define class means and covariances for two classes.
    means = [
        np.array([-1, -1, -1, -1]),
        np.array([1, 1, 1, 1])
    ]
    covariances = [
        np.array([[2, -0.5, 0.3, 0],
                  [-0.5, 1, -0.5, 0],
                  [0.3, -0.5, 1, 0],
                  [0, 0, 0, 2]]),
        np.array([[1, 0.3, -0.2, 0],
                  [0.3, 2, 0.3, 0],
                  [-0.2, 0.3, 1, 0],
                  [0, 0, 0, 3]])
    ]
    priors = [0.35, 0.65]
    # Generate data samples
    data = generate_samples(means, covariances, n=10000, priors=priors)
    features = data.drop(columns=['True Class Label']).to_numpy()
    labels = data['True Class Label'].values
    # Original classifier (correct model) scores
    scores_original = discriminant_scores(features, means, covariances)
    fpr_original, tpr_original, empirical_gamma_original, empirical_min_error_original = compute_roc_and_threshold(scores_original, labels, priors)
    # Naive Bayesian classifier scores
    scores_naive = naive_bayes_discriminant_scores(features, means)
    fpr_naive, tpr_naive, empirical_gamma_naive, empirical_min_error_naive = compute_roc_and_threshold(scores_naive, labels, priors)
    # LDA-based classifier
    estimated_means, estimated_covs = estimate_parameters(data)
    w_LDA = fisher_lda(estimated_means, estimated_covs)
    scores_lda = lda_discriminant_scores(features, w_LDA)
    fpr_lda, tpr_lda, empirical_gamma_lda, empirical_min_error_lda = compute_roc_and_threshold(scores_lda, labels, priors)
    # Plot combined ROC for original, naive, and LDA classifiers
    plot_combined_roc_2(
        fpr_original, tpr_original,
        fpr_naive, tpr_naive,
        fpr_lda, tpr_lda,
        scores_original, scores_naive, scores_lda,
        labels,
        empirical_gamma_original, empirical_min_error_original,
        empirical_gamma_naive, empirical_min_error_naive,
        empirical_gamma_lda, empirical_min_error_lda
    )


if __name__ == "__main__":
    main()
