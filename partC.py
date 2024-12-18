import numpy as np
from evaluation import roc_curve_and_probabilities, optimal_threshold, theoretical_min_error
from utility import generate_samples, estimate_parameters, fisher_lda, lda_discriminant_scores
from visualization import plot_lda_roc


def main():
    # Define class means for the two classes.
    mean_0 = np.array([-1, -1, -1, -1])
    mean_1 = np.array([1, 1, 1, 1])

    # Define class covariances for the two classes.
    cov_0 = np.array([[2, -0.5, 0.3, 0],
                      [-0.5, 1, -0.5, 0],
                      [0.3, -0.5, 1, 0],
                      [0, 0, 0, 2]])
    cov_1 = np.array([[1, 0.3, -0.2, 0],
                      [0.3, 2, 0.3, 0],
                      [-0.2, 0.3, 1, 0],
                      [0, 0, 0, 3]])
    mean_vectors = [mean_0, mean_1]
    covariance_matrices = [cov_0, cov_1]

    # Class priors
    prior_probabilities = [0.35, 0.65]

    # Misclassification costs for 0-1 loss
    lambda_10 = lambda_01 = 1
    lambda_11 = lambda_00 = 0

    # Compute a theoretical decision threshold using the given priors and costs.
    theoretical_gamma = (prior_probabilities[0] / prior_probabilities[1]) * ((lambda_10 - lambda_11) / (lambda_01 - lambda_00))
    # Generate data samples
    data = generate_samples(mean_vectors, covariance_matrices, n=10000, priors=prior_probabilities)
    # Estimate parameters from data
    estimated_means, estimated_covs = estimate_parameters(data)
    # Compute Fisher LDA projection vector
    w_LDA = fisher_lda(estimated_means, estimated_covs)
    # Compute LDA discriminant scores
    features = data.drop(columns=['True Class Label']).to_numpy()
    labels = data['True Class Label'].values
    lda_scores = lda_discriminant_scores(features, w_LDA)
    # Compute ROC metrics
    fpr, tpr, P_D0_L1, P_D1_L0 = roc_curve_and_probabilities(lda_scores, labels)
    # Determine the optimal empirical threshold and error based on LDA scores
    empirical_gamma, empirical_error = optimal_threshold(lda_scores, labels, prior_probabilities)
    # Compute the theoretical minimum error for the given theoretical gamma
    theoretical_min_error_value = theoretical_min_error(lda_scores, labels, prior_probabilities, theoretical_gamma)
    # Plot the ROC for LDA classifier
    plot_lda_roc(
        fpr, tpr,
        lda_scores,
        labels,
        theoretical_gamma,
        theoretical_min_error_value,
        empirical_gamma,
        empirical_error
    )


if __name__ == "__main__":
    main()
