import numpy as np
from evaluation import roc_curve_and_probabilities, empirical_min_error_threshold, theoretical_min_error
from utility import generate_samples, discriminant_scores, naive_bayes_discriminant_scores
from visualization import plot_combined_roc


def main():
    # Define class means for the two classes.
    mean_0 = np.array([-1, -1, -1, -1])
    mean_1 = np.array([1, 1, 1, 1])

    # Define class covariances (original scenario used correlated covariances; here assume diagonal).
    # This is the naive assumption where it is incorrectly believed features are independent.
    cov_0 = np.array([[2, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 2]])
    cov_1 = np.array([[1, 0, 0, 0],
                      [0, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 3]])
    mean_vectors = [mean_0, mean_1]
    covariance_matrices = [cov_0, cov_1]
    priors = [0.35, 0.65]
    # Misclassification costs for 0-1 loss
    lambda_10 = lambda_01 = 1
    lambda_11 = lambda_00 = 0
    # Compute the theoretical decision threshold (gamma)
    theoretical_gamma = (priors[0] / priors[1]) * ((lambda_10 - lambda_11) / (lambda_01 - lambda_00))
    # Generate dataset
    data = generate_samples(mean_vectors, covariance_matrices, n=10000, priors=priors)
    features = data.drop(columns=['True Class Label']).to_numpy()
    labels = data['True Class Label'].values
    # Compute discriminant scores using the true class-conditional distributions
    scores_original = discriminant_scores(features, mean_vectors, covariance_matrices)
    # Compute ROC metrics for the original (correct) model
    FPR_original, TPR_original, P_D0_L1_original, P_D1_L0_original = roc_curve_and_probabilities(scores_original, labels)
    empirical_gamma_original, empirical_min_error_original = empirical_min_error_threshold(scores_original, priors, P_D0_L1_original, P_D1_L0_original)
    theoretical_min_error_original = theoretical_min_error(scores_original, labels, priors, theoretical_gamma)
    # Compute discriminant scores using Naive Bayes assumption (incorrect covariance)
    naive_scores = naive_bayes_discriminant_scores(features, mean_vectors)
    # Compute ROC metrics for the Naive Bayesian classifier
    naive_FPR, naive_TPR, naive_P_D0_L1, naive_P_D1_L0 = roc_curve_and_probabilities(naive_scores, labels)
    naive_empirical_gamma, naive_empirical_min_error = empirical_min_error_threshold(naive_scores, priors, naive_P_D0_L1, naive_P_D1_L0)

    plot_combined_roc(
        FPR_original, TPR_original,
        naive_FPR, naive_TPR,
        scores_original, naive_scores,
        labels,
        theoretical_gamma,
        theoretical_min_error_original,
        empirical_gamma_original, empirical_min_error_original,
        naive_empirical_gamma, naive_empirical_min_error
    )


if __name__ == "__main__":
    main()
