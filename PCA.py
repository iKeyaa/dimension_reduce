import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import time
import os
import tracemalloc

data = pd.read_csv('..')

# 选取数值特征和目标列
target = data['label']
data = data.drop(columns=['label', 'id'])
numeric_features = data.select_dtypes(include=[np.number]).columns
categorical_features = data.select_dtypes(include=['object']).columns


# 数据预处理：标准化
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse=False), categorical_features)
    ])

data_scaled = preprocessor.fit_transform(data)

# 2. 应用PCA并绘制信息含量图
pca = PCA().fit(data_scaled)

# 累计解释的方差比例
cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)

# 计算百分比维度并绘图
dimension_percentages = np.arange(10, 110, 10)
explained_variances = [cumulative_var_ratio[int(len(cumulative_var_ratio)*p/100)-1] for p in dimension_percentages]

plt.figure(figsize=(10, 6))
plt.plot(dimension_percentages, explained_variances, marker='o')
plt.xlabel('Percentage of Dimensions (%)')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance by Dimension Percentage')
plt.grid(True)
plt.show()

# 3. 不同维度下的准确率、运行时间和内存使用
results_pca = []

for p in dimension_percentages:
    n_components = int(len(cumulative_var_ratio) * p / 100)
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(data_pca, target, test_size=0.3, random_state=32)
    
    # 训练模型并计时、内存使用
    start_time = time.time()
    tracemalloc.start()
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    end_time = time.time()
    
    results_pca.append({
        'Percentage of Dimensions': p,
        'Accuracy': accuracy,
        'Time (s)': end_time - start_time,
        'Memory Usage (Bytes)': peak
    })

# 输出结果
for result in results_pca:
    print(f"Dimensions: {result['Percentage of Dimensions']}%, Accuracy: {result['Accuracy']}, Time: {result['Time (s)']:.2f}s, Memory Usage: {result['Memory Usage (Bytes)']} Bytes")