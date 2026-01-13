import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
df = pd.read_csv('Dataset/processed/chest_xray_images_properties.csv')

train_val = df[df['dataset'].isin(['train','val'])]
test_df = df[df['dataset'] == 'test']

train_df, val_df = train_test_split(
    train_val,
    test_size=0.2,
    stratify=train_val['label'],
    random_state=42
)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))
img_size = (224,224)
batch_size = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_loader = train_gen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size
)

val_loader = val_gen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

test_loader = test_gen.flow_from_dataframe(
    test_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
))

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=20,
    callbacks=[early_stop]
)
test_loss, test_acc = model.evaluate(test_loader)
print("Test accuracy:", test_acc)

pred = model.predict(test_loader)
y_pred = np.argmax(pred, axis=1)
y_true = test_loader.classes

print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()

classes = list(test_loader.class_indices.keys())
plt.xticks([0,1], classes)
plt.yticks([0,1], classes)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center')

plt.show()
