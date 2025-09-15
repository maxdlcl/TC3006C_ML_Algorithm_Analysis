# Performance Analysis of a Decision Tree Classifier and Bagging with Scikit-learn

This repository reports the performance of a [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and an ensemble model with [Bagging Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) from the Scikit-learn library, evaluated on the [Palmer Penguins Dataset](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris).

The dataset includes three classes (Adelie, Chinstrap, Gentoo) with numerical features (bill length, bill depth, flipper length, body mass) and categorical features (island, sex), for a total of 344 samples.
The class distribution is moderately imbalanced: Adelie (44.2 %), Gentoo (36.0 %), Chinstrap (19.8 %), which makes metrics such as macro F1-score more suitable than accuracy to evaluate performance.

The implementation explores multiple models with regular and extreme hyperparameter values, showing examples of models with underfitting and overfitting to compare them with the best model:

1. Default Decision Tree (baseline, shows strong performance but risk of overfitting).
2. Tree with high impurity threshold (example of underfitting with high bias).
3. Deep Tree (example of overfitting with high variance).
3. Bagging Classifier with regularized Decision Trees (ensemble that achieves the best generalization, balancing bias and variance).

After running the file, the following plots are displayed:

1. A pie chart showing the class distribution of the dataset.
2. Confusion matrices with the model’s performance on the validation and test sets.
3. A comparison bar chart of train/validation/test accuracy across the four models.

## Running the implementation

```
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
```

Then, clone the repository and execute the script:

```
git clone https://github.com/maxdlcl/TC3006C_ML_Algorithm_Analysis
cd TC3006C_ML_Algorithm_Analysis
python3 decision_tree_analysis.py
```

## Submission info

* Maximiliano De La Cruz Lima
* A01798048
* Submission date: September 14th, 2025
