import pandas as pd
from sklearn import preprocessing
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle

# Load du lieu
df = pd.read_csv('https://raw.githubusercontent.com/CanhHo1004/dataset/main/divorce/divorce.csv',';')
# print(df)

# Kiem tra gia tri null
total =  df.isnull().sum()
# print(total)

X = df.drop('Class', axis=1)
y = df['Class']

def score_dt(X, y):
  kf = KFold(n_splits= 10)
  total = 0
  for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    total += accuracy_score(y_test, y_pred)
  pickle.dump(model, open('model.pkl','wb'))
  return total/10

test = score_dt(X, y) # < 0.95 tai k-Fold = 15
print(test)