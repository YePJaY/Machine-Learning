# training a logistic regression classifier to predict whether the flower is virginica or not
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['data'])
# print(iris.data.shape)#it will tell the shape of the array
# print(iris.target)

x = iris["data"][:,3:]# [row, column] # i need all index rows but index 3 column (0,1,2,3)
#y = (iris["target"]==2) # if the value is == 2 then it will return true else false
y = (iris["target"]==2).astype(np.int64) # now it will give true false in the form of 0 and 1
# print(y)

#train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x,y)

print(clf.predict([[2.6]])) # it is gonna tell 0 means no virginica otherwise 1 for yes

# using matplotlib to plot it
# (-1,1) will show data only in row while (1,-1) will show data in row and column
x_new = np.linspace(0,3,1000).reshape(-1,1)# its gonna divide the value  between 0 to 3 in 1000 parts
# print(x_new)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1], "g-",label="virginica")
plt.show()
