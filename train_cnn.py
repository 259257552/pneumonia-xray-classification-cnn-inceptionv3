import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import itertools
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, metrics
from keras.models import load_model

from tensorflow.keras.applications import InceptionV3

from plot_utils import *

df_all = pd.read_csv('Dataset/processed/chest_xray_images_properties.csv')
df_all.head()
train_val_df = df_all[(df_all['dataset'] == 'train') | (df_all['dataset'] == 'val')]
valid_size = 0.2
train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['label'], random_state=42)
train_df['dataset'] = 'train'
val_df['dataset'] = 'val'

test_df = df_all[df_all['dataset'] == 'test']

print("Before:")
print("Train:", len(df_all[df_all['dataset'] == 'train']))
print("Val:", len(df_all[df_all['dataset'] == 'val']))
print("------\nAfter:")
print("Train:", len(train_df))
print("Val:", len(val_df))

print("\nTest:", len(test_df))
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=5, # Rotate images by a maximum of 5 degrees
    width_shift_range=0.1, # Shift images horizontally by 10% of the width
    height_shift_range=0.1, # Shift images horizontally by 10% of the height
    zoom_range=0.2, # Zoom in/out by 20%
    fill_mode='nearest',
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

test_datagen = ImageDataGenerator(rescale=1.0/255)

img_size = (224, 224)
batch_size = 64 # 32, 64, 128

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle = True,
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle = False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

images, labels = next(train_generator)
class_indices = train_generator.class_indices

plot_images(images, labels, class_indices, num_images=10)
model = models.Sequential([
    layers.Input(shape=(224, 224, 1)),

    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model = models.Sequential([
    layers.Input(shape=(224, 224, 1)),

    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
count_normal = len(train_df[train_df['label'] == 'NORMAL'])
count_pneumonia = len(train_df[train_df['label'] == 'PNEUMONIA'])
count_total = len(train_df)

# Scaling by total/2 helps keep the loss to a similar magnitude - the sum of the weights of all examples stays the same.
weight_for_0 = (1 / count_normal) * (count_total / 2.0)
weight_for_1 = (1 / count_pneumonia) * (count_total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
checkpoint_path = "Models/CNN_callback/best_model.keras"

checkpoint_callback = ModelCheckpoint(checkpoint_path, 
                                      monitor='val_loss', 
                                      save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', 
                                        patience=5)
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

model.save(f'Models/CNN/cnn_model.keras')
print('Model saved!')

with open('Models/CNN/training_history.pkl', 'wb') as file:
    pickle.dump(history, file)
model = tf.keras.models.load_model('Models/CNN/model1/cnn_model.keras')

with open('Models/CNN/model1/training_history.pkl', 'rb') as file:
    history = pickle.load(file)    

plot_training_history(history)
test_loss, test_accuracy = model.evaluate(test_generator)
predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
y_probs = predictions[:, 1]

print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
plot_predictions(test_generator, y_true, y_pred, num_images=10)
print('\nClassification Report:')
report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(len(test_generator.class_indices))])
print(report)

print('\nEvaluation scores:')
print_evaluation_scores(y_true, y_pred)

print('\nConfusion matrix:')
cm = confusion_matrix(y_true, y_pred)
classes = list(test_generator.class_indices.keys())
plot_confusion_matrix(cm, classes)

