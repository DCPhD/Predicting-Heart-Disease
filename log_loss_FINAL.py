import pandas as pd
import numpy as np
import scipy as sp



# load csv files
testdf = pd.read_csv("train_predictions.csv")
testlabelsdf = pd.read_csv("train_labels.csv")
predictdf = pd.read_csv("export_dataframe.csv")

#change output settings
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)

# transform str to int
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
testdf["ID"] = le.fit_transform(testdf["ID"])
predictdf["ID"] = le.fit_transform(predictdf["ID"])
testdf["THAL"] = le.fit_transform(testdf["THAL"])
predictdf["THAL"] = le.fit_transform(predictdf["THAL"])

# ID is not relevant to model, HEART DZ will be our target
# train dataset
cols = [col for col in testdf.columns if col not in ["ID"]]
data = testdf[cols]
target = testlabelsdf["HEART DZ"]
# predict dataset
colspred = [col for col in predictdf.columns if col not in ["ID"]]
predictdf = predictdf[colspred]

from sklearn.model_selection import train_test_split
# split dataset
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=10) #test_size=0.30,random_state=10)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gnb = GaussianNB()
gnb.fit(data_train, target_train)
target_pred = gnb.predict(data_test)
ac = accuracy_score(target_test, target_pred, normalize=True)

# apply .predict function to predict dataset
yNew = gnb.predict(predictdf)

# determine probabilities for each prediction
prob = gnb.predict_proba(predictdf)

# filtered predictions to only include YES values
probabilityYes = pd.read_csv("probability_YES.csv")

# convert from df to array
py = np.asarray(probabilityYes)


def logloss(yNew, py, eps=1e-15):
    p = np.clip(py, eps, 1 - eps)
    if yNew == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)


#targets = np.array([yNew])
#predictions = np.array([py])

ll = [logloss(x,y) for (x,y) in zip(yNew, py)]
print(ll)

#ll_df = pd.DataFrame(ll)

#ll_df.to_csv("logloss.csv", index=False)




