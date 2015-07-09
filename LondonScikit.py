import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

# load data
x_train = np.loadtxt(open("train.csv","rb"), delimiter=",", skiprows=0)
y_train = np.loadtxt(open("trainLabels.csv","rb"), delimiter=",", skiprows=0)
x_test = np.loadtxt(open("test.csv","rb"), delimiter=",", skiprows=0)

# set parameters to be tuned
tuned_parameters = {'C': 10.**np.arange(-5,5)}

# conduct exhaustive gridsearch
clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=5,verbose=3)
clf.fit(x_train, y_train)

# print report
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

# make predictions
predictions = clf.predict(x_test)
out = np.vstack((np.arange(len(predictions)).astype(int)+1, predictions.astype(int))).T

# wirte out data
np.savetxt("LinearSVMresult.csv", out, header="Id,Solution", fmt="%d", comments="", delimiter=",")
