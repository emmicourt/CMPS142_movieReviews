import numpy as np 
import matplotlib.pyplot as plt 

from sklearn import svm, linear_model, metrics, KFold, cross_val_score, datasets
from sklearn.neighbors import KNeighborsClassifier

#scikit initializers 
log = linear_model.LogisticRegression(solver='lbfgs', C=1e5, multi_class='multinomial')
svc = svm.SVC(kernel='linear') 


# fake datasets I am using until we have a good boii
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# KNN on iris data
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# sckikit initializer 
# hey emily figure out what these mean thanks! 
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2,
           weights='uniform')

# training on traingin data 
knn.fit(iris_X_train, iris_y_train)
print(knn.fit(iris_X_train, iris_y_train).score(iris_X_train, iris_y_train))

#prediction
a = knn.predict(iris_X_test)
print(a)
print(iris_y_test)


