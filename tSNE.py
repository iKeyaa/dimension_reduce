import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
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

# 应用 t-SNE 并评估不同维度下的准确率、运行时间和内存使用
perplexities = np.arange(10, 110, 10)
results_tsne = []

for perplexity in perplexities:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    data_tsne = tsne.fit_transform(data_scaled)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(data_tsne, target, test_size=0.3, random_state=32)
    
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
    
    results_tsne.append({
        'Perplexity': perplexity,
        'Accuracy': accuracy,
        'Time (s)': end_time - start_time,
        'Memory Usage (Bytes)': peak
    })

# 输出结果
for result in results_tsne:
    print(f"Perplexity: {result['Perplexity']}, Accuracy: {result['Accuracy']}, Time: {result['Time (s)']:.2f}s, Memory Usage: {result['Memory Usage (Bytes)']} Bytes")