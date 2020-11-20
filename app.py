# Import các thư viện cần thiết
from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import pandas as pd
import pickle
import csv
import os

# Khai báo Flask
app = Flask(__name__)

# Tiến hành đọc mô hình đã huấn luyện trước từ file model.py
model = pickle.load(open('model.pkl', 'rb'))

# Gọi trang index.html
@app.route('/', methods = ["GET", "POST"])
def index():
    return render_template('index.html')

# Tải file csv lên
@app.route('/upload', methods = ["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"] # Nhận file csv được tải lên
        file.save(os.path.join("static/upload", file.filename)) # Lưu file csv vào thư mục static/upload
        return render_template('data.html', data = file.filename) # Gọi trang data.html kèm theo giá trị data là tên file csv
    return render_template('uploadcsv.html') # Gọi trang uploadcsv.html

# Dự đoán kết quả do người dùng chọn từ trang index.html
@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()] # Lấy tất cả 54 thuộc tính do người dùng chọn
    final_features = [np.array(int_features)] # Gán dữ liệu vào mảng
    prediction = model.predict(final_features) # Tiến hành dự đoán dựa vào mô hình đã huấn luyện
    output = round(prediction[0], 2) # 2 trạng thái đầu ra 0,1
    result = ""
    # Hiển thị kết quả dự đoán có hoặc không
    if(output==1):
        result = "Có"
    if(output==0):
        result = "Không"
    return render_template('index.html', prediction_text='Có ly hôn hay không: {}'.format(result)) # Hiển thị kết quả

# Dự đoán từ file csv
@app.route('/result', methods=['GET','POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile'] # Lấy tên file csv cần dự đoán
        path = "static/upload/" + f # Đường dẫn đến file csv 
        data = []
        with open(path) as file:
            csvfile = csv.reader(file) # Đọc file
            next(file) # Bỏ qua dòng tiêu đề

            # Dự đoán cho từng dòng dữ liệu
            for row in csvfile:
                final_features = [np.array(row)]
                prediction = model.predict(final_features) # Dự đoán 1 dòng
                output = round(prediction[0], 2)
                if(output==1):
                    row.append('1') # Nối kết quả dự đoán vào cột cuối (Class)
                    data.append(row) # Nối dòng đã dự đoán vào mảng data
                if(output==0):
                    row.append('0')
                    data.append(row)

        # Chèn tiêu đề cho bảng kết quả
        data.insert(0,['Atr1','Atr2','Atr3','Atr4','Atr5','Atr6','Atr7','Atr8','Atr9','Atr10',
                        'Atr11','Atr12','Atr13','Atr14','Atr15','Atr16','Atr17','Atr18','Atr19','Atr20',
                        'Atr21','Atr22','Atr23','Atr24','Atr25','Atr26','Atr27','Atr28','Atr29','Atr30',
                        'Atr31','Atr32','Atr33','Atr34','Atr35','Atr36','Atr37','Atr38','Atr39','Atr40',
                        'Atr41','Atr42','Atr43','Atr44','Atr45','Atr46','Atr47','Atr48','Atr49','Atr50',
                        'Atr51','Atr52','Atr53','Atr54','Class'])
        data = pd.DataFrame(data)
        data.to_csv('./static/output/result.csv', header=False, index=False) # Xuất ra file csv và lưu vào thư mục static/output
        return render_template('result.html', prediction_text_csv=data.to_html(header=False)) # Hiển thị bảng đã được dự đoán

# Tải file csv xuống
@app.route('/download', methods=['POST'])
def download():
    file = open("./static/output/result.csv", "r")
    return Response(file, mimetype="text/csv",headers={"Content-disposition":"attachment; filename=result.csv"})

# Hàm main để chạy chương trình
if __name__ == "__main__":
    app.run(debug=True)