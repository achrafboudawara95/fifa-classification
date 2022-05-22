import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as xgb

data = pd.read_csv("./FIFA 22 MLS PLAYER RATINGS.csv")

data.info()

data.describe()

features = data[["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]]

labels = data["POS"]

labels.value_counts().plot(kind='bar')
plt.show()

le = LabelEncoder()
encodedLabels = le.fit_transform(labels)
encodedLabels = encodedLabels.reshape(-1,1)

xTrain, xTest, yTrain, yTest = train_test_split(features, encodedLabels, test_size=0.33, random_state=0)

modelxgb = xgb(booster='gbtree',
            objective='multi:softprob', max_depth=3,
            learning_rate=0.1, n_estimators=100)

modelxgb.fit(xTrain, yTrain)

predxgb = modelxgb.predict(xTest)

# Model Evaluation metrics 
from sklearn.metrics import accuracy_score

print('Accuracy Score (xgboost) : ' + str(accuracy_score(yTest, predxgb)))

from sklearn.neighbors import KNeighborsClassifier
modelknn = KNeighborsClassifier(n_neighbors=6)

modelknn.fit(xTrain,yTrain)
predknn= modelknn.predict(xTest)

print('Accuracy Score (knn) : ' + str(accuracy_score(yTest, predknn)))

from sklearn.ensemble import RandomForestClassifier

modelRforest = RandomForestClassifier(random_state=0,n_estimators=100,max_depth=10)

modelRforest.fit(xTrain, yTrain)

predRforest = modelRforest.predict(xTest)

print('Accuracy Score (random forest) : ' + str(accuracy_score(yTest, predRforest)))

from sklearn import svm

modelSVC = svm.SVC(kernel='poly')
    
modelSVC.fit(xTrain, yTrain)
    
predSVC = modelSVC.predict(xTest)

print('Accuracy Score (svm) : ' + str(accuracy_score(yTest, predSVC)))