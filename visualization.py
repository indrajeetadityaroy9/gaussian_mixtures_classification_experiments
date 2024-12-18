import matplotlib.pyplot as plt
from evaluation import binary_metrics


def plot_3d(df):
    fig, ax = create_plot(figsize=(10, 10), projection='3d')
    color_map = {0: 'r', 1: 'b'}

    for label in df['True Class Label'].unique():
        data = df[df['True Class Label'] == label]
        ax.scatter(
            data['feature_1'],
            data['feature_2'],
            data['feature_3'],
            c=color_map.get(label, 'g'),
            label=f'Class {label}'
        )

    ax.set_xlabel('Dimension 1 X')
    ax.set_ylabel('Dimension 2 Y')
    ax.set_zlabel('Dimension 3 Z')
    ax.legend()
    plt.show()


def create_plot(figsize=(10, 8), projection=None):
    fig = plt.figure(figsize=figsize)
    if projection:
        ax = fig.add_subplot(111, projection=projection)
    else:
        ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    fig.patch.set_facecolor('white')
    return fig, ax


def plot_roc_curve(ax, fpr, tpr, color='blue', label="ROC Curve"):
    ax.plot(fpr, tpr, label=label, color=color)


def plot_point(ax, discriminant, gamma, labels, color, marker, label_format, error, error_label):
    metrics = binary_metrics(discriminant > gamma, labels)
    ax.scatter(metrics['FPR'], metrics['TPR'], color=color, marker=marker,
               label=label_format.format(error=error, gamma=gamma))


def plot_roc_base(fpr, tpr, discriminant, labels, t_gamma, t_min_error, e_gamma, e_min_error, label_format_o, label_format_x):
    fig, ax = create_plot()
    plot_roc_curve(ax, fpr, tpr)
    plot_point(ax, discriminant, e_gamma, labels, 'red', 'o', label_format_o, e_min_error, 'Empirical')
    plot_point(ax, discriminant, t_gamma, labels, 'green', 'x', label_format_x, t_min_error, 'Theoretical')
    finalize_roc_plot(ax)
    plt.show()


def finalize_roc_plot(ax):
    ax.set_xlabel("False Positive Probability P(D=1|L=0)")
    ax.set_ylabel("True Positive Probability P(D=1|L=1)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.6)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right')


def plot_simple_roc(fpr, tpr):
    fig, ax = create_plot()
    plot_roc_curve(ax, fpr, tpr)
    finalize_roc_plot(ax)
    plt.show()


def plot_roc(fpr, tpr, discriminant, labels, t_gamma, t_min_error_value, e_gamma, e_min_error):
    label_format_o = "Empirical Min P(error)={error:.3f}, Empirical gamma={gamma:.3f}"
    label_format_x = "Theoretical Min P(error)={error:.3f}, Theoretical gamma={gamma:.3f}"
    plot_roc_base(fpr, tpr, discriminant, labels, t_gamma, t_min_error_value, e_gamma, e_min_error,
                  label_format_o, label_format_x)


def plot_combined_roc(fpr_orig, tpr_orig, fpr_naive, tpr_naive,
                      discriminant_orig, discriminant_naive, labels,
                      t_gamma, t_min_error_orig, e_gamma_orig, e_min_error_orig,
                      e_gamma_naive, e_min_error_naive):
    fig, ax = create_plot(figsize=(12, 10))
    label_format_1 = "Original Empirical Min P(error) ={error:.3f}, Empirical Gamma={gamma:.3f}"
    label_format_2 = "Theoretical Min P(error) ={error:.3f}, Theoretical Gamma={gamma:.3f}"
    label_format_3 = "Naive Empirical Min P(error) ={error:.3f}, Naive Gamma={gamma:.3f}"

    plot_roc_curve(ax, fpr_orig, tpr_orig, color='blue', label="Original Classifier")
    plot_roc_curve(ax, fpr_naive, tpr_naive, color='green', label="Naive Bayesian Classifier")

    plot_point(ax, discriminant_orig, e_gamma_orig, labels, 'blue', 'o', label_format_1,
               e_min_error_orig, 'Empirical')
    plot_point(ax, discriminant_orig, t_gamma, labels, 'green', 'x', label_format_2,
               t_min_error_orig, 'Theoretical')
    plot_point(ax, discriminant_naive, e_gamma_naive, labels, 'cyan', 'o', label_format_3,
               e_min_error_naive, 'Empirical')

    finalize_roc_plot(ax)
    plt.show()


def plot_lda_roc(fpr, tpr, discriminant, labels, t_gamma, t_min_error, e_gamma, e_min_error):
    plot_roc_base(
        fpr, tpr, discriminant, labels, t_gamma, t_min_error, e_gamma, e_min_error,
        label_format_o="LDA Empirical Min P(error)={error:.3f}, Empirical gamma={gamma:.3f}",
        label_format_x="LDA Theoretical Min P(error)={error:.3f}, Theoretical gamma={gamma:.3f}"
    )


def plot_combined_roc_2(fpr_orig, tpr_orig, fpr_naive, tpr_naive, fpr_lda, tpr_lda,
                        discriminant_orig, discriminant_naive, lda_scores, labels,
                        e_gamma_orig, e_min_error_orig, e_gamma_naive, e_min_error_naive,
                        e_gamma_lda, e_min_error_lda):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    ax.set_facecolor("white")

    def plot_roc_and_point(fpr, tpr, discriminant, gamma, e_min_error, label, color):
        plt.plot(fpr, tpr, label=label, color=color)
        metrics = binary_metrics(discriminant > gamma, labels)
        plt.scatter(metrics['FPR'], metrics['TPR'], color=color, marker='o',
                    label=f"{label} Empirical Min P(error) ={e_min_error:.3f}, Gamma={gamma:.3f}")

    plot_roc_and_point(fpr_orig, tpr_orig, discriminant_orig, e_gamma_orig,
                       e_min_error_orig, "Original Classifier", 'blue')
    plot_roc_and_point(fpr_naive, tpr_naive, discriminant_naive, e_gamma_naive, e_min_error_naive,
                       "Naive Bayesian Classifier", 'green')
    plot_roc_and_point(fpr_lda, tpr_lda, lda_scores, e_gamma_lda, e_min_error_lda, "LDA Classifier",
                       'orange')

    plt.xlabel("False Positive Probability P(D=1|L=0)")
    plt.ylabel("True Positive Probability P(D=1|L=1)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.6)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc='lower right')
    plt.title("Combined ROC for Original Classifier, Naive Bayes, and LDA")
    plt.show()
