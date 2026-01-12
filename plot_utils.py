import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score

def plot_images(images, labels, class_indices, num_images=5):
    
    color = ['dodgerblue', 'crimson']
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))
    axes = axes.flatten()
    for img, lbl, ax in zip(images[:num_images], labels[:num_images], axes):
        ax.imshow(img.squeeze(), cmap='gray')
        label = list(class_indices.keys())[list(class_indices.values()).index(np.argmax(lbl))]
        ax.set_title(label, color = color[class_indices[label]])
        ax.axis('off')
    
    plt.suptitle('Chest X-Ray images post augmentation')
    plt.tight_layout()
    plt.show()
    
    
def plot_training_history_all(history, metrics=['precision', 'recall', 'accuracy', 'loss']):

  fig, ax = plt.subplots(1, len(metrics), figsize=(20, 8))
  ax = ax.ravel()

  for i, metric in enumerate(metrics):
    ax[i].plot(history.history[metric], label='Training ' + metric.capitalize())
    ax[i].plot(history.history['val_' + metric], label='Validation ' + metric.capitalize())
    ax[i].set_title('Model ' + metric.capitalize())
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel(metric.capitalize())
    ax[i].legend()

  plt.show()
  
  
def plot_training_history(history):
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot training & validation loss values
    axs[0].plot(history.history['loss'], label='Train')
    axs[0].plot(history.history['val_loss'], label='Validation')
    axs[0].set_title('Model Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot training & validation accuracy values
    axs[1].plot(history.history['accuracy'], label='Train')
    axs[1].plot(history.history['val_accuracy'], label='Validation')
    axs[1].set_title('Model Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=12)
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} \n({cm[i,j] / cm.sum() * 100:.2f}%)',
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    
def plot_predictions(test_generator, y_true, y_pred, num_images=5):
    
    test_image_paths = test_generator.filepaths
    random_indices = random.sample(range(len(test_image_paths)), num_images)
    
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 3*num_rows))
    for i, idx in enumerate(random_indices):
        img = cv2.imread(test_image_paths[idx])
        img = cv2.resize(img, (224, 224))
    
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(img,  cmap='gray')
        true_label = y_true[idx]
        predicted_label = y_pred[idx]
        title_color = 'seagreen' if true_label == predicted_label else 'crimson'
        plt.title(f'True: {true_label}, Pred: {predicted_label}', color=title_color)
        plt.axis('off')
        
    plt.suptitle('Predictions')
    plt.show()
    

def print_evaluation_scores(y_true, y_pred):
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f'* Accuracy: {accuracy:.4f}')
    print(f'* Precision: {precision:.4f}')
    print(f'* Recall: {recall:.4f}')
    print(f'* F1-Score: {f1:.4f}')
    
    
def plot_precision_recall_curve(y_true, y_probs):
    
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_probs)
        
    plt.figure()
    plt.plot(recalls, precisions)
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(alpha=0.3)
    plt.show()


def evaluate_classification_performance(predictions, test_generator):

    # Generate predictions
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Classification report
    print('\nClassification Report:')
    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(len(test_generator.class_indices))])
    print(report)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'* Accuracy: {accuracy:.4f}')
    print(f'* Precision: {precision:.4f}')
    print(f'* Recall: {recall:.4f}')
    print(f'* F1-Score: {f1:.4f}')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = list(test_generator.class_indices.keys())
    plot_confusion_matrix(cm, classes)
    
    # Precision-recall curve
    y_probs = predictions[:, 1] 
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_probs)
    
    plt.figure()
    plt.plot(recalls, precisions)
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(alpha=0.3)
    plt.show()
