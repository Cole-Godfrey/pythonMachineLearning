import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

# load in data
data = pd.read_csv("student-mat.csv", sep=";")
# select only the columns we want
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# prediction is final grade
predict = "G3"
X = np.array(data.drop(columns=[predict]))
y = np.array(data[predict])
XTrain, XTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best = 0
for _ in range(30):
    XTrain, XTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(XTrain, yTrain)
    acc = linear.score(XTest, yTest)
    print(acc)

    if acc > best:
        best = acc
        # save the model
        with open("student-model.pkl", "wb") as f:
            pickle.dump(linear, f)

# load the model
pickleIn = open("student-model.pkl", "rb")
linear = pickle.load(pickleIn)

print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# make predictions
predictions = linear.predict(XTest)

# display predicted and actual value
for x in range(len(predictions)):
    print(predictions[x], XTest[x], yTest[x])

p = "absences"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()





