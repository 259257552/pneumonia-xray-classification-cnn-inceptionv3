import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cv2
print("="*70)
print("第一部分：数据加载")
print("="*70)
base_dir = 'Dataset/chest_xray'
train_path = os.path.join(base_dir, 'train')
val_path = os.path.join(base_dir, 'val')
test_path = os.path.join(base_dir, 'test')
print(f"\n数据集根目录：{base_dir}")
print(f"训练集路径：{train_path}")
print(f"验证集路径：{val_path}")
print(f"测试集路径：{test_path}")
print("\n1.2 获取图像文件路径...")
train_neg = glob(os.path.join(train_path, 'NORMAL/*.jpeg'))
train_pos = glob(os.path.join(train_path, 'PNEUMONIA/*.jpeg'))
val_neg = glob(os.path.join(val_path, 'NORMAL/*.jpeg'))
val_pos = glob(os.path.join(val_path, 'PNEUMONIA/*.jpeg'))
test_neg = glob(os.path.join(test_path, 'NORMAL/*.jpeg'))
test_pos = glob(os.path.join(test_path, 'PNEUMONIA/*.jpeg'))
print(f"训练集 - 正常图像：{len(train_neg)} 张")
print(f"训练集 - 肺炎图像：{len(train_pos)} 张")
print(f"验证集 - 正常图像：{len(val_neg)} 张")
print(f"验证集 - 肺炎图像：{len(val_pos)} 张")
print(f"测试集 - 正常图像：{len(test_neg)} 张")
print(f"测试集 - 肺炎图像：{len(test_pos)} 张")
print("\n" + "="*70)
print("第二部分：图像特征提取")
print("="*70)
 def extract_image_properties(image_paths, dataset_name=''):
    data = [] 
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {img_path}")
            continue
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
        if (idx + 1) % 500 == 0:
            print(f"  已处理 {dataset_name} 中的 {idx + 1}/{len(image_paths)} 张图像")
    df_img = pd.DataFrame(data)
    return df_img
print("\n2.1 提取训练集特征...")
df_train_neg = extract_image_properties(train_neg, "训练集-正常")
df_train_pos = extract_image_properties(train_pos, "训练集-肺炎")
print("\n2.2 提取验证集特征...")
df_val_neg = extract_image_properties(val_neg, "验证集-正常")
df_val_pos = extract_image_properties(val_pos, "验证集-肺炎")
print("\n2.3 提取测试集特征...")
df_test_neg = extract_image_properties(test_neg, "测试集-正常")
df_test_pos = extract_image_properties(test_pos, "测试集-肺炎")
print("\n2.4 合并所有数据...")
df_all = pd.concat([df_train_neg, df_train_pos, df_val_neg, df_val_pos, 
                     df_test_neg, df_test_pos], 
                    ignore_index=True)
print(f"总样本数：{len(df_all)}")
print(f"\n数据集分布：")
print(df_all['dataset'].value_counts())
print(f"\n标签分布：")
print(df_all['label'].value_counts())
output_path = 'Dataset/processed/chest_xray_images_properties.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_all.to_csv(output_path, index=False)
print(f"\n数据已保存到：{output_path}")
print("\n" + "="*70)
print("第三部分：数据统计分析")
print("="*70)
print("\n3.1 基本统计信息：")
print(df_all[['max', 'min', 'mean', 'std', 'height', 'width']].describe())
print("\n3.2 按标签分组的统计信息：")
print("\n正常图像统计：")
print(df_all[df_all['label'] == 'NORMAL'][['mean', 'std']].describe())
print("\n肺炎图像统计：")
print(df_all[df_all['label'] == 'PNEUMONIA'][['mean', 'std']].describe())
print("\n" + "="*70)
print("第四部分：数据可视化")
print("="*70)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)
print("\n4.1 绘制平均像素值分布...")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df_all[df_all['label'] == 'NORMAL'], x='mean', bins=20, 
             kde=True, label='正常', ax=ax[0], color='dodgerblue')
ax[0].set_title('正常图像的平均像素值分布', fontsize=12, fontweight='bold')
ax[0].set_xlabel('平均像素值')
ax[0].set_ylabel('频数')
ax[0].legend()
sns.histplot(df_all[df_all['label'] == 'PNEUMONIA'], x='mean', bins=20, 
             kde=True, label='肺炎', ax=ax[1], color='crimson')
ax[1].set_title('肺炎图像的平均像素值分布', fontsize=12, fontweight='bold')
ax[1].set_xlabel('平均像素值')
ax[1].set_ylabel('频数')
ax[1].legend()
sns.histplot(df_all[df_all['label'] == 'NORMAL'], x='mean', bins=20, 
             kde=True, label='正常', ax=ax[2], color='dodgerblue', alpha=0.6)
sns.histplot(df_all[df_all['label'] == 'PNEUMONIA'], x='mean', bins=20, 
             kde=True, label='肺炎', ax=ax[2], color='crimson', alpha=0.6)
ax[2].set_title('正常与肺炎图像平均像素值对比', fontsize=12, fontweight='bold')
ax[2].set_xlabel('平均像素值')
ax[2].set_ylabel('频数')
ax[2].set_xlim(0, 255)
ax[2].legend()
plt.tight_layout()
plt.savefig('Results/01_mean_pixel_distribution.png', dpi=300, bbox_inches='tight')
print("已保存：Results/01_mean_pixel_distribution.png")
plt.show()
print("\n4.2 绘制像素值标准差分布...")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df_all[df_all['label'] == 'NORMAL'], x='std', bins=20, 
             kde=True, label='正常', ax=ax[0], color='dodgerblue')
ax[0].set_title('正常图像的像素值标准差分布', fontsize=12, fontweight='bold')
ax[0].set_xlabel('标准差')
ax[0].set_ylabel('频数')
ax[0].legend()
sns.histplot(df_all[df_all['label'] == 'PNEUMONIA'], x='std', bins=20, 
             kde=True, label='肺炎', ax=ax[1], color='crimson')
ax[1].set_title('肺炎图像的像素值标准差分布', fontsize=12, fontweight='bold')
ax[1].set_xlabel('标准差')
ax[1].set_ylabel('频数')
ax[1].legend()
sns.histplot(df_all[df_all['label'] == 'NORMAL'], x='std', bins=20, 
             kde=True, label='正常', ax=ax[2], color='dodgerblue', alpha=0.6)
sns.histplot(df_all[df_all['label'] == 'PNEUMONIA'], x='std', bins=20, 
             kde=True, label='肺炎', ax=ax[2], color='crimson', alpha=0.6)
ax[2].set_title('正常与肺炎图像标准差对比', fontsize=12, fontweight='bold')
ax[2].set_xlabel('标准差')
ax[2].set_ylabel('频数')
ax[2].set_xlim(0, 100)
ax[2].legend()
plt.tight_layout()
plt.savefig('Results/02_std_pixel_distribution.png', dpi=300, bbox_inches='tight')
print("已保存：Results/02_std_pixel_distribution.png")
plt.show()
print("\n4.3 绘制图像尺寸分布...")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_all, x='height', bins=20, kde=True, ax=ax[0], color='steelblue')
ax[0].set_title('图像高度分布', fontsize=12, fontweight='bold')
ax[0].set_xlabel('高度（像素）')
ax[0].set_ylabel('频数')
sns.histplot(df_all, x='width', bins=20, kde=True, ax=ax[1], color='steelblue')
ax[1].set_title('图像宽度分布', fontsize=12, fontweight='bold')
ax[1].set_xlabel('宽度（像素）')
ax[1].set_ylabel('频数')
plt.tight_layout()
plt.savefig('Results/03_image_size_distribution.png', dpi=300, bbox_inches='tight')
print("已保存：Results/03_image_size_distribution.png")
plt.show()
print("\n4.4 绘制不同数据集的图像尺寸分布...")
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df_all[df_all['dataset'] == 'train'], x='height', bins=20, 
             kde=True, ax=ax[0, 0], color='steelblue')
ax[0, 0].set_title('训练集 - 图像高度分布', fontsize=12, fontweight='bold')
ax[0, 0].set_xlabel('高度（像素）')
sns.histplot(df_all[df_all['dataset'] == 'train'], x='width', bins=20, 
             kde=True, ax=ax[0, 1], color='steelblue')
ax[0, 1].set_title('训练集 - 图像宽度分布', fontsize=12, fontweight='bold')
ax[0, 1].set_xlabel('宽度（像素）')
sns.histplot(df_all[df_all['dataset'] == 'test'], x='height', bins=20, 
             kde=True, ax=ax[1, 0], color='coral')
ax[1, 0].set_title('测试集 - 图像高度分布', fontsize=12, fontweight='bold')
ax[1, 0].set_xlabel('高度（像素）')
sns.histplot(df_all[df_all['dataset'] == 'test'], x='width', bins=20, 
             kde=True, ax=ax[1, 1], color='coral')
ax[1, 1].set_title('测试集 - 图像宽度分布', fontsize=12, fontweight='bold')
ax[1, 1].set_xlabel('宽度（像素）')
plt.tight_layout()
plt.savefig('Results/04_dataset_size_distribution.png', dpi=300, bbox_inches='tight')
print("已保存：Results/04_dataset_size_distribution.png")
plt.show()
print("\n4.5 绘制数据集标签分布...")
df_distribution = df_all.groupby(['dataset', 'label']).size().reset_index(name='counts')
fig, ax = plt.subplots(figsize=(12, 6))
barplot = sns.barplot(data=df_distribution, x='dataset', y='counts', hue='label', 
                      palette=['dodgerblue', 'crimson'], ax=ax)
for p in barplot.patches:
    height = p.get_height()
    barplot.annotate(f'{int(height)}',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
ax.set_title('各数据集中正常和肺炎图像的分布', fontsize=14, fontweight='bold')
ax.set_xlabel('数据集', fontsize=12)
ax.set_ylabel('样本数', fontsize=12)
ax.legend(title='标签', labels=['正常', '肺炎'], fontsize=11)
ax.set_ylim([0, max(df_distribution['counts']) * 1.1])

plt.tight_layout()
plt.savefig('Results/05_label_distribution.png', dpi=300, bbox_inches='tight')
print("已保存：Results/05_label_distribution.png")
plt.show()
print("\n" + "="*70)
print("第五部分：样本图像展示")
print("="*70)
print("\n5.1 展示正常和肺炎图像样本...")
np.random.seed(42)
normal_samples = df_all[df_all['label'] == 'NORMAL'].sample(n=5, random_state=42)
pneumonia_samples = df_all[df_all['label'] == 'PNEUMONIA'].sample(n=5, random_state=42)
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for idx, (ax, sample_path) in enumerate(zip(axes[0], normal_samples['path'])):
    img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap='gray')
    ax.set_title('正常', color='dodgerblue', fontsize=12, fontweight='bold')
    ax.axis('off')
for idx, (ax, sample_path) in enumerate(zip(axes[1], pneumonia_samples['path'])):
    img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap='gray')
    ax.set_title('肺炎', color='crimson', fontsize=12, fontweight='bold')
    ax.axis('off')
fig.suptitle('胸部X光图像样本展示', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('Results/06_sample_images.png', dpi=300, bbox_inches='tight')
print("已保存：Results/06_sample_images.png")
plt.show()
print("\n" + "="*70)
print("第六部分：数据分析总结")
print("="*70)
print("\n关键发现：")
print(f"1. 总样本数：{len(df_all)}")
print(f"   - 训练集：{len(df_all[df_all['dataset'] == 'train'])} 张")
print(f"   - 验证集：{len(df_all[df_all['dataset'] == 'val'])} 张")
print(f"   - 测试集：{len(df_all[df_all['dataset'] == 'test'])} 张")
print(f"\n2. 标签分布：")
print(f"   - 正常图像：{len(df_all[df_all['label'] == 'NORMAL'])} 张 ({len(df_all[df_all['label'] == 'NORMAL'])/len(df_all)*100:.1f}%)")
print(f"   - 肺炎图像：{len(df_all[df_all['label'] == 'PNEUMONIA'])} 张 ({len(df_all[df_all['label'] == 'PNEUMONIA'])/len(df_all)*100:.1f}%)")
print(f"\n3. 图像尺寸：")
print(f"   - 高度范围：{df_all['height'].min()}-{df_all['height'].max()} 像素")
print(f"   - 宽度范围：{df_all['width'].min()}-{df_all['width'].max()} 像素")
print(f"\n4. 像素值特征：")
normal_mean = df_all[df_all['label'] == 'NORMAL']['mean'].mean()
pneumonia_mean = df_all[df_all['label'] == 'PNEUMONIA']['mean'].mean()
print(f"   - 正常图像平均像素值：{normal_mean:.2f}")
print(f"   - 肺炎图像平均像素值：{pneumonia_mean:.2f}")
print(f"   - 差异：{abs(normal_mean - pneumonia_mean):.2f}")
print("\n数据探索分析完成！")
print("="*70)
