import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# Plot original class distribution
def plot_class_distribution(df):
    class_counts = df['Class'].value_counts()
    labels = ['Legit', 'Fraud']
    colors = ['green', 'red']

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Class', palette=colors, ax=ax)

    for i, count in enumerate(class_counts):
        percent = (count / len(df)) * 100
        ax.text(i, count, f'{count:,}\n({percent:.2f}%)', ha='center', va='bottom', fontsize=10)

    ax.set_xticklabels(labels)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Transaction Count")
    ax.set_xlabel("Transaction Type")
    return fig

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    return fig

# Plot evaluation metrics
def plot_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.barplot(x=["Accuracy", "Precision", "Recall", "F1 Score"],
                y=[accuracy, precision, recall, f1], ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Evaluation Metrics")
    return fig

# Compare class distribution before and after SMOTE
def plot_class_distribution_before_after_smote(y_before, y_after):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    sns.countplot(x=y_before, ax=axs[0], palette="Blues")
    axs[0].set_title("Before SMOTE")
    axs[0].set_xlabel("Class")
    axs[0].set_ylabel("Count")

    sns.countplot(x=y_after, ax=axs[1], palette="Greens")
    axs[1].set_title("After SMOTE")
    axs[1].set_xlabel("Class")
    axs[1].set_ylabel("Count")

    plt.tight_layout()
    return fig
