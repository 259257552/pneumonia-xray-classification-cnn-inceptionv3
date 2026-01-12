import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cv2
base_dir = 'Dataset/chest_xray'
train_path = os.path.join(base_dir, 'train')
val_path = os.path.join(base_dir, 'val')
test_path = os.path.join(base_dir, 'test')
train_neg = glob(os.path.join(train_path, 'NORMAL/*.jpeg'))
train_pos = glob(os.path.join(train_path, 'PNEUMONIA/*.jpeg'))
val_neg = glob(os.path.join(val_path, 'NORMAL/*.jpeg'))
val_pos = glob(os.path.join(val_path, 'PNEUMONIA/*.jpeg'))
test_neg = glob(os.path.join(test_path, 'NORMAL/*.jpeg'))
test_pos = glob(os.path.join(test_path, 'PNEUMONIA/*.jpeg'))
def extract_image_properties(image_paths):
    data = []
    for img_path in image_paths: 
        img = cv2.imread(img_path)
        properties = {
            'path': img_path,
            'label': img_path.split('/')[-2],
            'dataset': img_path.split('/')[-3],
            'max': img.max(),
            'min': img.min(),
            'mean': img.mean(),
            'std': img.std(),
            'height': img.shape[0],
            'width': img.shape[1]
        }
        data.append(properties)

    df_img = pd.DataFrame(data)
    
    return df_img
    df_train_neg = extract_image_properties(train_neg)
df_train_pos = extract_image_properties(train_pos)
df_val_neg = extract_image_properties(val_neg)
df_val_pos = extract_image_properties(val_pos)
df_test_neg = extract_image_properties(test_neg)
df_test_pos = extract_image_properties(test_pos)

df_all = pd.concat([df_train_neg, df_train_pos, df_val_neg, df_val_pos, df_test_neg, df_test_pos])
df_all.to_csv('Dataset/processed/chest_xray_images_properties.csv', index=False)
# df_all = pd.read_csv('Dataset/processed/chest_xray_images_properties.csv')
df_all = pd.read_csv('Dataset/processed/chest_xray_images_properties.csv')
df_all.head()

df_all.describe()
fig, ax = plt.subplots(1,3, figsize=(20,5))

# Normal
sns.histplot(df_all[df_all['label'] == 'NORMAL'], x='mean', bins=20, kde=True, label='Normal', color='dodgerblue', ax = ax[0])
ax[0].legend()
ax[0].set_title('Distribution of Mean Pixel Intensities for NORMAL')

# Pneumonia
sns.histplot(df_all[df_all['label'] == 'PNEUMONIA'], x='mean', bins=20, kde=True, label = 'Pneumonia', color='crimson', ax = ax[1])
ax[1].legend()
ax[1].set_title('Distribution of Mean Pixel Intensities for PNEUMONIA')

# Both
sns.histplot(df_all[df_all['label'] == 'NORMAL'], x='mean', bins=20, kde=True, label='Normal', color='dodgerblue', ax = ax[2])
sns.histplot(df_all[df_all['label'] == 'PNEUMONIA'], x='mean', bins=20, kde=True, label = 'Pneumonia', color='crimson', ax = ax[2])
ax[2].legend()
ax[2].set_title('Combined Distribution of Mean Pixel Intensities for NORMAL and PNEUMONIA')

for axis in ax[1:]:
    axis.set_xlim(0, 255)

plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# Normal
sns.histplot(df_all[df_all['label'] == 'NORMAL'], x='std', bins=20, kde=True, label='Normal', color='dodgerblue', ax=ax[0])
ax[0].legend()
ax[0].set_title('Distribution of Std Pixel Intensities for NORMAL')

# Pneumonia
sns.histplot(df_all[df_all['label'] == 'PNEUMONIA'], x='std', bins=20, kde=True, label='Pneumonia', color='crimson', ax=ax[1])
ax[1].legend()
ax[1].set_title('Distribution of Std Pixel Intensities for PNEUMONIA')

# Both
sns.histplot(df_all[df_all['label'] == 'NORMAL'], x='std', bins=20, kde=True, label='Normal', color='dodgerblue', ax=ax[2])
sns.histplot(df_all[df_all['label'] == 'PNEUMONIA'], x='std', bins=20, kde=True, label='Pneumonia', color='crimson', ax=ax[2])
ax[2].legend()
ax[2].set_title('Combined Distribution of Std Pixel Intensities for NORMAL and PNEUMONIA')

for axis in ax:
    axis.set_xlim(0, 100)

plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.histplot(df_all, x='height', bins=20, kde=True, color='seagreen', ax=ax[0])
sns.histplot(df_all, x='width', bins=20, kde=True, color='orange', ax=ax[1])
ax[0].set_title('Image height distribution')
ax[1].set_title('Image width distribution')

sns.histplot(df_all[df_all['dataset'] == 'NORMAL'], x='mean', bins=20, kde=True, label='Normal', color='dodgerblue', ax = ax[0])
    
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(2,2, figsize=(10, 8))

sns.histplot(df_all[df_all['dataset'] == 'train'], x='height', bins=20, kde=True, color='seagreen', ax = ax[0, 0])
ax[0, 0].set_title('Image height distribution - Train')

sns.histplot(df_all[df_all['dataset'] == 'train'], x='width', bins=20, kde=True, color='orange', ax = ax[0, 1])
ax[0, 1].set_title('Image width distribution - Train')

sns.histplot(df_all[df_all['dataset'] == 'test'], x='height', bins=20, kde=True, color='seagreen', ax = ax[1, 0])
ax[1, 0].set_title('Image height distribution - Test')

sns.histplot(df_all[df_all['dataset'] == 'test'], x='width', bins=20, kde=True, color='orange', ax = ax[1, 1])
ax[1, 1].set_title('Image width distribution - Test')

plt.tight_layout()
plt.show()
df_distribution = df_all.groupby(['label'])['dataset'].value_counts().reset_index(name='counts')

plt.figure(figsize=(15, 5))
barplot = sns.barplot(data=df_distribution, x='dataset', y='counts', hue='label', palette=['dodgerblue', 'crimson'])
for p, total, count in zip(barplot.patches, df_distribution.groupby('dataset')['counts'].transform('sum'), df_distribution['counts']):
    percentage = 100 * count / total
    annotation_text = f"{count}\n({percentage:.1f}%)"
    plt.annotate(annotation_text, (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.title('Label Distribution across Datasets')
plt.xlabel('Dataset')
plt.ylabel('Counts')
plt.ylim([0, 4500])
plt.legend(title='Label')
plt.show()
normal_samples = df_all[df_all['label'] == 'NORMAL'].sample(n=5, random_state=None)
pneumonia_samples = df_all[df_all['label'] == 'PNEUMONIA'].sample(n=5, random_state=None)

fig, axes = plt.subplots(2, 5, figsize=(15, 5))

for ax, sample in zip(axes.ravel(), normal_samples['path']):
    img = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap='gray')
    ax.set_title('NORMAL', color='dodgerblue')
    ax.axis('off')

for ax, sample in zip(axes.ravel()[5:], pneumonia_samples['path']):
    img = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap='gray')
    ax.set_title('PNEUMONIA', color='crimson')
    ax.axis('off')
