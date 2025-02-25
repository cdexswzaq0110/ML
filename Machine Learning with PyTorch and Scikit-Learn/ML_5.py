#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. 讀取葡萄酒數據集
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                   'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# 過濾數據，只使用類別 2 和 3
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

# 2. 劃分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# 3. 定義參數組合
learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5]
n_estimators_list = [100, 500, 1000]

# 儲存結果
results = []

# 4. 測試不同參數組合
for lr in learning_rates:
    for n_estimators in n_estimators_list:
        ada = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators,
            learning_rate=lr,
            random_state=1
        )
        ada.fit(X_train, y_train)
        
        # 計算正確率
        train_acc = accuracy_score(y_train, ada.predict(X_train))
        test_acc = accuracy_score(y_test, ada.predict(X_test))
        
        # 儲存結果
        results.append([lr, n_estimators, train_acc, test_acc])

# 5. 輸出結果為 DataFrame
columns = ['learning_rate', 'n_estimators', 'train_accuracy', 'test_accuracy']
results_df = pd.DataFrame(results, columns=columns)

# 6. 將結果轉為所需的表格格式
pivot_table = results_df.pivot(index='learning_rate', columns='n_estimators', 
                               values=['train_accuracy', 'test_accuracy'])

# 調整列寬與數字格式
pd.set_option("display.max_columns", None)  # 顯示所有列
pd.set_option("display.width", 1000)  # 調整寬度
pd.set_option("display.float_format", lambda x: f"{x:.4f}")  # 格式化數字至小數點後 4 位
print(pivot_table)
pivot_table.to_csv("adaboost_results.csv")  # 儲存成 CSV 文件


# In[ ]:




