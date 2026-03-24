import sklearn
from sklearn import datasets
from sklearn import svm

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

XTrain, XTest, YTrain, YTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']