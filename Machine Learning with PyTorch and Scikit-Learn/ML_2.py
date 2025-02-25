#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 匯入所需的函式庫
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# 加載葡萄酒數據集並分割數據
# 葡萄酒數據集包含178個樣本，分為三個類別，共13個特徵
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values

# 將數據集分為訓練集和測試集（70%訓練，30%測試）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# 標準化特徵
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 使用 SBS（序列特徵選擇）選擇前 5 個特徵
knn = KNeighborsClassifier(n_neighbors=5)
sfs = SequentialFeatureSelector(knn, n_features_to_select=5, direction='backward', scoring='accuracy', cv=5)
sfs.fit(X_train_std, y_train)
selected_features_sbs = sfs.get_support(indices=True)

# 使用隨機森林選擇特徵
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
selected_features_rf = indices[:2]

# 使用 SBS 和 RF 選擇的特徵進行比較
selected_features = {
    'SBS': selected_features_sbs[:2],
    'RF': selected_features_rf
}

# 定義 plot_decision_regions 函數，用於繪製決策區域
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 設定標記生成器和顏色映射
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 繪製決策邊界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 繪製類別樣本
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

    # 突顯測試樣本
    if test_idx is not None:
        # 繪製所有測試樣本
        X_test, y_test = X[test_idx, :], y_test[test_idx]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='Test set')

# 訓練不同模型並計算準確率以進行比較
classifiers = {
    "Perceptron": Perceptron(max_iter=100, eta0=0.05, random_state=1),
    "Logistic Regression": LogisticRegression(C=0.5, solver='lbfgs', multi_class='ovr', random_state=1),
    "Support Vector Machine": SVC(kernel='rbf', C=1.0, gamma=0.1, random_state=1),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=1),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski')
}

results = []

plt.figure(figsize=(15, 10))  # 設置總體圖形大小

# 訓練並評估每個分類器
for idx, (name, classifier) in enumerate(classifiers.items()):
    for method, features in selected_features.items():
        # 在選擇的特徵上擬合模型
        classifier.fit(X_train_std[:, features], y_train)
        # 預測訓練和測試數據的標籤
        y_train_pred = classifier.predict(X_train_std[:, features])
        y_test_pred = classifier.predict(X_test_std[:, features])

        # 計算訓練和測試的準確率
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # 儲存結果以便後續比較
        results.append([name, method, train_acc, test_acc])

        # 繪製每個分類器的決策區域
        plt.subplot(4, 3, idx * 2 + (1 if method == 'SBS' else 2))  # 在一個畫布上畫多個子圖
        plot_decision_regions(X_train_std[:, features], y_train, classifier=classifier)
        plt.xlabel('Feature 1 [standardized]')
        plt.ylabel('Feature 2 [standardized]')
        plt.legend(loc='upper left')
        plt.title(f'{name} ({method}) Decision Region')

plt.tight_layout()
plt.show()


# In[2]:


# 以表格形式顯示結果
# 打印表格頭部
print("{:<25} {:<15}     {:<15} {:<15}".format("Algorithm", "Feature Selection", "訓練資料正確率", "測試資料正確率"))
print("-" * 82)

# 打印結果
for name, method, train_acc, test_acc in results:
    print("{:<25}     {:<15}       {:<15.2f}       {:<15.2f}".format(name, method, train_acc, test_acc))


# In[ ]:




