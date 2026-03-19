import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
X = np.array(data.drop(columns=[predict]))
y = np.array(data[predict])
XTrain, XTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
linear = linear_model.LinearRegression()
linear.fit(XTrain, yTrain)
acc = linear.score(XTest, yTest)

print(acc)
print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(XTest)

for x in range(len(predictions)):
    print(predictions[x], XTest[x], yTest[x])



