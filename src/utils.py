import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def init_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def log(message, level=logging.INFO):
    logging.log(level, message)


def plot_feature_importances(importances, feature_names):
    sns.barplot(x=importances, y=feature_names)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.show()


def plot_learning_curve(train_losses, val_losses, title='Learning Curve'):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_performance(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['mse'], label='MSE')
    plt.plot(history['mae'], label='MAE')
    plt.title('Model Performance')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()
