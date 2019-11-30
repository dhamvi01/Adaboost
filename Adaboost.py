from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = AdaBoostClassifier(n_estimators=50, learning_rate=1)

model.fit(x_train,y_train)

pred = model.predict(x_test)

print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
print(metrics.confusion_matrix(y_test,pred))


########################################################################################

from sklearn.svm import SVC

sv = SVC(probability=True, kernel='linear')

model = AdaBoostClassifier(n_estimators=50, learning_rate=1, base_estimator=sv)

model.fit(x_train,y_train)

pred = model.predict(x_test)

print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
print(metrics.confusion_matrix(y_test,pred))














