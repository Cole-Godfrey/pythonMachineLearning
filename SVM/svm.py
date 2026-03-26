import sklearn
from sklearn import datasets
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

XTrain, XTest, YTrain, YTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(XTrain, YTrain)

YPred = clf.predict(XTest)

acc = metrics.accuracy_score(YPred, YTest)
print(f"KNN: {acc}")

clf = svm.SVC(kernel='linear', C=1)
clf.fit(XTrain, YTrain)

YPred = clf.predict(XTest)

acc = metrics.accuracy_score(YPred, YTest)

print(f"SVM: {acc}")

