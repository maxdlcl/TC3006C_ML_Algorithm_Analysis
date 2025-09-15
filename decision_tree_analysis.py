from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np


def load_data():
  """Loads the Palmer Penguins dataset"""
  try:
    data = sns.load_dataset('penguins')
  except Exception:
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')


  X = data.drop(columns=['species'])
  y = data['species']
  class_names = sorted(y.unique().tolist())
  return X, y, class_names


def plot_data(y, class_names):
  """Plots class distribution as a pie chart.
  
  Args:
      y (Series): label vector.
      class_names (list): ordered class names to display.
  """
  class_counts = y.value_counts().reindex(class_names, fill_value=0)
  plt.figure(figsize=(6, 6))
  plt.pie(
      class_counts,
      labels=class_names,
      autopct="%1.1f%%",
      startangle=140,
      colors=sns.color_palette("pastel"),
  )
  plt.title("Class Distribution")


def evaluate_model(y_true, y_pred, class_names, title="Confusion Matrix"):
  """Prints standard classification metrics and plots a confusion matrix.
  
  Args:
      y_true (array-like): ground-truth labels.
      y_pred (array-like): predicted labels.
      class_names (list): label order to align rows/cols in the confusion matrix.
      title (str): title for the heatmap figure.
  """
  cm = confusion_matrix(y_true, y_pred, labels=class_names)
  print("Confusion Matrix:\n", cm)
  print("\nClassification Report:")
  print(classification_report(y_true, y_pred, target_names=class_names))
  print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
  # Heatmap for quick visual inspection of errors by class.
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d',
              xticklabels=class_names, yticklabels=class_names, cmap='Blues')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title(title)


def collect_split_accuracies(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
  """Returns a tuple with accuracies to compare bias/variance quickly."""
  return {
        "model": name,
        "train": model.score(X_train, y_train),
        "val":   model.score(X_val, y_val),
        "test":  model.score(X_test, y_test)
    }


def plot_split_comparison(rows, title="Train vs Val vs Test Accuracy"):
  """Grouped bar chart to compare accuracies across splits and models."""
  df = pd.DataFrame(rows)
  plt.figure(figsize=(8,5))
  idx = np.arange(len(df))
  width = 0.25
  plt.bar(idx - width, df["train"], width, label="Train")
  plt.bar(idx,         df["val"],   width, label="Validation")
  plt.bar(idx + width, df["test"],  width, label="Test")
  plt.xticks(idx, df["model"], rotation=15)
  plt.ylim(0, 1.05)
  plt.ylabel("Accuracy")
  plt.title(title)
  plt.grid(True, axis="y", alpha=0.3)
  plt.legend()

def preprocess_data(X):
  """Impute numeric (median) + categorical (most_frequent) and OHE on whole X before splitting."""
  # Split columns
  num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
  cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

  # Numeric imputer
  num_imputer = SimpleImputer(strategy='median')
  X[num_cols] = num_imputer.fit_transform(X[num_cols])

  # Categorical imputer + OHE
  cat_imputer = SimpleImputer(strategy='most_frequent')
  X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
  ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  X_ohe = ohe.fit_transform(X[cat_cols])
  ohe_cols = ohe.get_feature_names_out(cat_cols)
  X_ohe_df = pd.DataFrame(X_ohe, columns=ohe_cols, index=X.index)
  
  # Combine numeric and OHE categorical
  X_processed = pd.concat([X[num_cols], X_ohe_df], axis=1)
  return X_processed

if __name__ == "__main__":
  # Load data
  X_raw, y, class_names = load_data()

  # Plot class distribution
  plot_data(y, class_names)

  X = preprocess_data(X_raw)

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

  print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

  rows = []

  print("Evaluating Decision Tree Classifier with default parameters:")

  # Create the full pipeline with a Decision Tree classifier with best found hyperparameters, which are the default ones:
  m1 = DecisionTreeClassifier(ccp_alpha=0.0, criterion='gini', max_depth=None, 
                                            min_samples_leaf=1, min_samples_split=2 , random_state=42)
  
  # Train the model 
  m1.fit(X_train, y_train) 
  
  # Evaluate on validation set 
  print("\nValidation Set Evaluation:") 
  y_val_pred = m1.predict(X_val) 
  evaluate_model(y_val, y_val_pred, class_names, title="Confusion Matrix (Validation) Model 1")

  # Evaluate on test set 
  print("Test Set Evaluation:")
  y_test_pred = m1.predict(X_test) 
  evaluate_model(y_test, y_test_pred, class_names, title="Confusion Matrix (Test) Model 1")
  
  rows.append(collect_split_accuracies("DT default", m1, X_train, y_train, X_val, y_val, X_test, y_test))


  print("\nEvaluating Decision Tree Classifier with min_impurity_decrease=0.2:")

  m2 = DecisionTreeClassifier(min_impurity_decrease=0.2, random_state=42)

  # Train the model 
  m2.fit(X_train, y_train) 
  
  # Evaluate on validation set 
  print("\nValidation Set Evaluation:") 
  y_val_pred = m2.predict(X_val) 
  evaluate_model(y_val, y_val_pred, class_names, title="Confusion Matrix (Validation) Model 2")

  # Evaluate on test set 
  print("Test Set Evaluation:")
  y_test_pred = m2.predict(X_test) 
  evaluate_model(y_test, y_test_pred, class_names, title="Confusion Matrix (Test) Model 2")

  rows.append(collect_split_accuracies("DT imp_dec=0.2", m2, X_train, y_train, X_val, y_val, X_test, y_test))


  print("\nEvaluating Decision Tree with an example of an unregularized tree:")

  m3 = DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)

  # Train the model 
  m3.fit(X_train, y_train)

  # Evaluate on validation set 
  print("\nValidation Set Evaluation:") 
  y_val_pred = m3.predict(X_val) 
  evaluate_model(y_val, y_val_pred, class_names, title="Confusion Matrix (Validation) Model 3")

  # Evaluate on test set 
  print("Test Set Evaluation:")
  y_test_pred = m3.predict(X_test) 
  evaluate_model(y_test, y_test_pred, class_names, title="Confusion Matrix (Test) Model 3")

  rows.append(collect_split_accuracies("DT unreg", m3, X_train, y_train, X_val, y_val, X_test, y_test))


  print("Evaluating Decision Tree with regularization for overfitting mitigation:")

  # Create the full pipeline with a Bagging Classifier wrapping a Decision Tree classifier
  
  base = DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=4, random_state=42)
  m4 = BaggingClassifier(estimator=base, n_estimators=50, random_state=42, n_jobs=-1)

  # Train the model 
  m4.fit(X_train, y_train)

  # Evaluate on validation set 
  print("\nValidation Set Evaluation:") 
  y_val_pred = m4.predict(X_val) 
  evaluate_model(y_val, y_val_pred, class_names, title="Confusion Matrix (Validation) Model 4")

  # Evaluate on test set 
  print("Test Set Evaluation:")
  y_test_pred = m4.predict(X_test) 
  evaluate_model(y_test, y_test_pred, class_names, title="Confusion Matrix (Test) Model 4")
  
  rows.append(collect_split_accuracies("DT bagging", m4, X_train, y_train, X_val, y_val, X_test, y_test))


  plot_split_comparison(rows, title="Decision Trees / Bagging — Train vs Validation vs Test Accuracy")  


  plt.show()
