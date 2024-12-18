import numpy as np


def binary_metrics(predictions, true_classes):
    predictions = np.asarray(predictions)
    true_classes = np.asarray(true_classes)

    tp = np.sum((predictions == 1) & (true_classes == 1))
    tn = np.sum((predictions == 0) & (true_classes == 0))
    fp = np.sum((predictions == 1) & (true_classes == 0))
    fn = np.sum((predictions == 0) & (true_classes == 1))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {'TPR': tpr, 'FPR': fpr}


def roc_curve_and_probabilities(discriminant, labels):
    sorted_scores = np.sort(discriminant)
    thresholds = np.concatenate(([sorted_scores[0] - 1e-9], sorted_scores, [sorted_scores[-1] + 1e-9]))

    fpr, tpr, P_D0_L1, P_D1_L0 = [], [], [], []

    for t in thresholds:
        preds = (discriminant > t).astype(int)
        metrics = binary_metrics(preds, labels)

        fpr.append(metrics['FPR'])
        tpr.append(metrics['TPR'])
        P_D0_L1.append(1 - metrics['TPR'])
        P_D1_L0.append(metrics['FPR'])

    return fpr, tpr, P_D0_L1, P_D1_L0


def empirical_min_error_threshold(discriminant, priors, P_D0_L1, P_D1_L0):
    P_L0, P_L1 = priors
    sorted_scores = np.sort(discriminant)
    thresholds = np.concatenate(([sorted_scores[0] - 1e-9], sorted_scores, [sorted_scores[-1] + 1e-9]))

    min_error = 1.0
    empirical_gamma = thresholds[0]

    for i, t in enumerate(thresholds):
        error = P_D1_L0[i] * P_L0 + P_D0_L1[i] * P_L1
        if error < min_error:
            min_error = error
            empirical_gamma = t

    return empirical_gamma, min_error


def error_rate(threshold, discriminant, labels, priors):
    P_L0, P_L1 = priors
    preds = (discriminant > threshold).astype(int)
    metrics = binary_metrics(preds, labels)
    return metrics['FPR'] * P_L0 + (1 - metrics['TPR']) * P_L1


def optimal_threshold(discriminant, labels, priors):
    sorted_scores = np.sort(discriminant)
    thresholds = np.concatenate(([sorted_scores[0] - 1e-9], sorted_scores, [sorted_scores[-1] + 1e-9]))

    errors = [error_rate(t, discriminant, labels, priors) for t in thresholds]
    min_idx = np.argmin(errors)
    return thresholds[min_idx], errors[min_idx]


def theoretical_min_error(discriminant, labels, priors, gamma):
    return error_rate(gamma, discriminant, labels, priors)
