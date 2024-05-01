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
data = data.drop(columns=[ 'id'])
numeric_features = data.select_dtypes(include=[np.number]).columns


# 数据预处理：标准化
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
    ])

data_processed = preprocessor.fit_transform(data[numeric_features])
data_processed = pd.DataFrame(data_processed)  # 将处理后的数据转换为DataFrame

# 添加标签列回数据集
data_processed['label'] = target.values

# 计算相关性
correlation_matrix = data_processed.corr()

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

percentages = np.arange(10, 110, 10)  # 可以根据需求调整
results_corr = []

for percentage in percentages:
    # 计算要选择的特征数
    num_features = int(len(correlation_matrix.columns) * percentage / 100)
    # 选取相关性最强的特征
    strongest_features = correlation_matrix['label'].abs().sort_values(ascending=False)[1:num_features+1].index.tolist()

    # 去掉自己
    strongest_features.remove('label')
    
    # 分割数据集
    X = np.array(data_processed[strongest_features])
    y = np.array(data_processed['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)
    
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
    
    results_corr.append({
        'Percentage of Features': percentage,
        'Accuracy': accuracy,
        'Time (s)': end_time - start_time,
        'Memory Usage (Bytes)': peak
    })

for result in results_corr:
    print(f"Features: {result['Percentage of Features']}%, Accuracy: {result['Accuracy']}, Time: {result['Time (s)']:.2f}s, Memory Usage: {result['Memory Usage (Bytes)']} Bytes")