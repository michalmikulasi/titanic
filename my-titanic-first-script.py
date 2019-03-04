# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
x = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 0].values

#importing test.csv. sadly, i did not know how to do that simply. I am really new to this and i will be thankful for suggestions
dataset2 = pd.read_csv('test.csv')
testovaci = dataset2.iloc[:, :].values


#taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median')
imputer = imputer.fit(x[:, 2:3])
x[: , 2:3] = imputer.transform(x[:, 2:3])


#encoding categorical variable Gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 1] = labelencoder_x.fit_transform(x[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_x.fit_transform(y)



#this is real test.csv
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median')
imputer = imputer.fit(testovaci[:, 2:3])
testovaci[: , 2:3] = imputer.transform(testovaci[:, 2:3])


#again encoding the same variable, but this time in test.csv file. 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencodertestovaci = LabelEncoder()
testovaci[:, 1] = labelencodertestovaci.fit_transform(testovaci[:, 1])
onehotencoder1 = OneHotEncoder(categorical_features = [1])
testovaci = onehotencoder1.fit_transform(testovaci).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)

#actual random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(xtrain, ytrain)


ypred = classifier.predict(testovaci)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.








