import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm, linear_model, metrics, KFold, cross_val_score, datasets
from sklearn.neighbors import KNeighborsClassifier

## how to load a dataset from csv file
# f = open("filename.txt")
## skip the header: 
# f.readline()  
# data = np.loadtxt(f)

# scikit initializers
# Use parameters to control 
log = linear_model.LogisticRegression(solver='lbfgs', C=1e5, multi_class='multinomial')
svc = svm.SVC(kernel='linear') 

## how to load data from sklearn 
## These are fake datasets I am using until we have a good boii
iris = datasets.load_iris()
iris_X = iris.data 		# this is how you load data 
iris_y = iris.target	# this is how you load lables


#####################
# KNN on iris data
#####################

# this preps the training and test data indicies so the algo knows which ones to 
# train/test on
np.random.seed(0)	#randomly splits training data 
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# KKN initializer 
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


