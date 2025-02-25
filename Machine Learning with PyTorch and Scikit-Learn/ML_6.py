#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Ensure NLTK resources are downloaded
nltk.download('stopwords', quiet=True)

# Data cleaning and preprocessing
porter = PorterStemmer()

def preprocessor(text):
    """
    Clean the text by removing HTML tags and non-word characters,
    convert to lowercase
    """
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)
    # Remove non-word characters and convert to lowercase
    text = re.sub('[\W]+', ' ', text.lower())
    return text

def tokenizer_porter(text):
    """
    Tokenize and stem the text using Porter Stemmer
    """
    return [porter.stem(word) for word in text.split()]

# Get English stopwords
stop = stopwords.words('english')

# Read the IMDb dataset
df = pd.read_csv(r'C:\Users\HUANG\Desktop\movie_data.csv')

# Preprocess the reviews
df['review'] = df['review'].apply(preprocessor)

# Separate features and target
X = df['review']
y = df['sentiment']

# Split the dataset (50/50 train/test with stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

# Define TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    strip_accents=None,
    lowercase=False,
    preprocessor=None,
    tokenizer=tokenizer_porter,
    stop_words=stop
)

# Define models and their parameter grids
models = {
    "Perceptron": (
        Perceptron(), 
        {
            'clf__penalty': [None, 'l1', 'l2'], 
            'clf__alpha': [0.0001, 0.001, 0.01]
        }
    ),
    "Logistic Regression": (
        LogisticRegression(solver='liblinear'), 
        {
            'clf__penalty': ['l1', 'l2'], 
            'clf__C': [1.0, 10.0, 100.0]
        }
    ),
    "Support Vector Machine": (
        SVC(), 
        {
            'clf__kernel': ['linear', 'rbf'], 
            'clf__C': [1.0, 10.0, 100.0]
        }
    ),
    "Decision Tree": (
        DecisionTreeClassifier(), 
        {
            'clf__criterion': ['gini', 'entropy'], 
            'clf__max_depth': [None, 10, 50]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(), 
        {
            'clf__n_estimators': [10, 50, 100], 
            'clf__max_depth': [None, 10, 50]
        }
    ),
    "K-Nearest Neighbors": (
        KNeighborsClassifier(), 
        {
            'clf__n_neighbors': [3, 5, 7], 
            'clf__p': [1, 2]
        }
    )
}

# Store results
results = []

# Evaluate each model
for model_name, (model, param_grid) in models.items():
    print(f"Processing: {model_name}")
    
    # Create pipeline
    pipeline = Pipeline([('vect', tfidf), ('clf', model)])
    
    # Perform Grid Search
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        scoring='accuracy', 
        cv=5, 
        n_jobs=-1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    
    # Calculate accuracies
    train_acc = grid_search.best_score_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    # Print detailed classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    # Store results
    results.append([
        model_name, 
        train_acc, 
        test_acc, 
        best_params
    ])
    
    print(f"{model_name} completed!")

# Convert results to DataFrame
results_df = pd.DataFrame(
    results, 
    columns=['Algorithm', 'Train Accuracy', 'Test Accuracy', 'Best Parameters']
)

# Display results
print("\nFinal Results:")
print(results_df)


# In[ ]:




