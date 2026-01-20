# Program: Load different kinds of datasets using scikit-learn library

from sklearn.datasets import load_iris, load_digits, load_diabetes, load_breast_cancer

# 1) Load Iris dataset (classification dataset)
iris = load_iris()
print("1) IRIS DATASET (Classification)")
print("Data shape:", iris.data.shape)
print("Target shape:", iris.target.shape)
print("First 3 target values:", iris.target[:3])
print("-" * 50)

# 2) Load Digits dataset (image classification dataset)
digits = load_digits()
print("2) DIGITS DATASET (Image Classification)")
print("Data shape:", digits.data.shape)
print("Target shape:", digits.target.shape)
print("First 3 target values:", digits.target[:3])
print("-" * 50)

# 3) Load Diabetes dataset (regression dataset)
diabetes = load_diabetes()
print("3) DIABETES DATASET (Regression)")
print("Data shape:", diabetes.data.shape)
print("Target shape:", diabetes.target.shape)
print("First 3 target values:", diabetes.target[:3])
print("-" * 50)

# 4) Load Breast Cancer dataset (classification dataset)
cancer = load_breast_cancer()
print("4) BREAST CANCER DATASET (Classification)")
print("Data shape:", cancer.data.shape)
print("Target shape:", cancer.target.shape)
print("First 3 target values:", cancer.target[:3])
print("-" * 50)
