import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
pd.options.mode.chained_assignment = None

# Type: DataFrame
train_set = pd.read_csv(SCRIPT_PATH + "/train.csv")
test_set = pd.read_csv(SCRIPT_PATH + "/test.csv")

# Output the number of passengers
# Output all features and the number of NaN value
print(len(train_set))
print(train_set.isnull().sum())
print(len(test_set))
print(test_set.isnull().sum())

# Get statistical information of the data
print(train_set.describe())
print(test_set.describe())
# We found that there are too many missing data in Cabin column
# of both train_set and test_set. So I decided to exclude Cabin column.

# Output the value and number of Embarked column
print(train_set["Embarked"].value_counts())

# Fill in NaN Age column with mean value
train_set["Age"].fillna(train_set["Age"].mean(), inplace=True)
test_set["Age"].fillna(test_set["Age"].mean(), inplace=True)
train_set["Embarked"].fillna(train_set["Embarked"].value_counts().index[0], inplace=True)

# Convert age into labels
age_bins = [0, 20, 30, 40, 50, 999]
age_labels = ['20-', '20-30', '30-40', '40-50', '50+']
train_set['AgeRange'] = pd.cut(train_set['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
test_set['AgeRange'] = pd.cut(test_set['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

# Drop useless column which do not have an impact on survivability
X_train = train_set.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Fare', 'Age', 'Cabin'], axis=1)
y_train = train_set['Survived']
X_test = test_set.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Age', 'Cabin'], axis=1)

# Convert some data columns into dummy variables
X_train = pd.get_dummies(X_train, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'AgeRange'])
X_test = pd.get_dummies(X_test, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'AgeRange'])

# Remove duplicated data
# e.g. Sex_male = 1 if MALE else FEMAIL
#   so Sex_Female is duplicated
X_train.drop(['Pclass_1', 'Sex_female', 'SibSp_0', 'Parch_0', 'Embarked_C', 'AgeRange_20-'], axis=1, inplace=True)
X_test.drop(['Pclass_1', 'Sex_female', 'SibSp_0', 'Parch_0', 'Embarked_C', 'AgeRange_20-'], axis=1, inplace=True)

"""
While I were fitting classifier to the dataset, 
I found that train_set has 1 less column than test_set 
and that missing column is 'Parch_9'. 
This is because train_set does not have any data with Parch = 9.
So I inserted column of zeros as Parch_9 dummy variable into train_set.
"""
zeros_array = np.zeros((X_train.shape[0],), dtype=np.int)
X_train.insert(loc=15, column='Parch_9', value=zeros_array)

# Use ExtraTreesClassifier to predict which features are important
clf = ExtraTreesClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)

# Use tree-based feature selection to minimize the numbers of features.
X_train_opt = model.transform(X_train)
# Which an element is True iff its corresponding feature is selected for retention.
feature_idx_train = model.get_support()
feature_name_train = X_train.columns[feature_idx_train]
print(feature_name_train)  # ['Pclass_3', 'Sex_male', 'Parch_1']

X_test_opt = model.transform(X_test)
feature_idx_test = model.get_support()
feature_name_test = X_test.columns[feature_idx_test]
print(feature_name_test)  # ['Pclass_3', 'Sex_male', 'Parch_1']

# Classify and get the prediction of testing data -> y_pred
classifier = GaussianNB()
classifier.fit(X_train_opt, y_train)
y_pred = classifier.predict(X_test_opt)
passengerId = test_set['PassengerId']
submission = pd.DataFrame({'PassengerId': passengerId, 'Survived': y_pred})
submission.to_csv('submission.csv', index=False)

X_train.to_csv("new_train.csv")
