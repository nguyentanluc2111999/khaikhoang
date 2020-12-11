# Khai báo thư viện cần thiết
import pandas as pd
from sklearn import preprocessing
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

# Nạp dữ liệu vào
df = pd.read_csv('https://raw.githubusercontent.com/CanhHo1004/dataset/main/divorce/divorce.csv',';')
# print(df)

# Kiểm tra giá trị null trong tập dữ liệu
total =  df.isnull().sum()
# print(total)

# Lấy các cột thuôc tính
X = df.drop('Class', axis=1)
# Lấy cột nhãn
y = df['Class']

# Hàm xây dựng mô hình cây quyết định sử dụng 170-Fold
def score_dt(X, y):
  kf = KFold(n_splits= 170)                   # Chọn số fold = 170
  total = 0
  for train_index, test_index in kf.split(X): # Tiến hành phân chia
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] # Lấy X_train, X_test theo từng phần được chia của X
    y_train, y_test = y.iloc[train_index], y.iloc[test_index] # Lấy y_train, y_test theo từng phần được chia của y
    # Cây quyết định
    model = DecisionTreeClassifier()          # Khai báo mô hình sử dụng
    model.fit(X_train, y_train)               # Tiến hành huấn luyện
    y_pred = model.predict(X_test)            # Tiến hành dự đoán
    total += accuracy_score(y_test, y_pred)   # Tổng độ chính xác qua các phần
    pickle.dump(model, open('model.pkl','wb'))  # Xuất ra file model.pkl
  return total/170                             # Trả về độ chính xác trung bình

tong = 0
print("Cây quyết định")
for i in range(0,10):
  test = score_dt(X, y)                         # Chạy thử hàm score_dt với X, y là 2 giá trị được lấy ở phía trên
  tong += test
  print("Lần " + str(i+1) + ": " + str(test))
print("Giá trị trung bình 10 lần chạy:", tong/10)
print("#########")


# So sánh với các giải thuật khác
print("So sánh với các giải thuật khác")
# Bayes thơ ngây
def sosanh(X, y):
  tong0 = 0
  tong1 = 0
  tong2 = 0
  for i in range(0, 10):
    kf = KFold(n_splits = 170)
    total0 = 0
    total1 = 0
    total2 = 0
    for train_index, test_index in kf.split(X):
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      # Cây quyết định
      model0 = DecisionTreeClassifier()
      model0.fit(X_train, y_train)
      # Bayes thơ ngây
      model1 = GaussianNB()
      model1.fit(X_train, y_train)
      # SVM
      model2 = svm.SVC(kernel='rbf')
      model2.fit(X_train, y_train)
      
      y_pred0 = model0.predict(X_test)
      total0 += accuracy_score(y_test, y_pred0)
      y_pred1 = model1.predict(X_test)
      total1 += accuracy_score(y_test, y_pred1)
      y_pred2 = model2.predict(X_test)
      total2 += accuracy_score(y_test, y_pred2)
    tong0 += total0/170
    tong1 += total1/170
    tong2 += total2/170
    print("Lần " + str(i+1) + ":")
    print("Cây quyết định : ", total0/170)
    print("Bayes thơ ngây : ", total1/170)
    print("SVM            : ", total2/170)
  print("Trung bình 10 lần chạy:")
  print("Cây quyết định : ", tong0/10)
  print("Bayes thơ ngây : ", tong1/10)
  print("SVM            : ", tong2/10)
sosanh(X, y)
