from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import csv
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods = ["GET", "POST"])
def index():
    return render_template('index.html')

@app.route('/upload', methods = ["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        file.save(os.path.join("static/upload", file.filename))
        return render_template('data.html', data = file.filename)
    return render_template('uploadcsv.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    result = ""
    if(output==1):
        result = "Có"
    if(output==0):
        result = "Không"
    return render_template('index.html', prediction_text='Có ly hôn hay không: {}'.format(result))

@app.route('/result', methods=['GET','POST'])
def data():
    # Hiển thị bảng CSV chưa có cột dự đoán
    # if request.method == 'POST':
    #     f = request.form['csvfile']
    #     path = "static/upload/" + f
    #     data = []
    #     with open(path) as file:
    #         csvfile = csv.reader(file)
    #         for row in csvfile:
    #             data.append(row)
    #     data = pd.DataFrame(data)
    #     return render_template('result.html', data=data.to_html(header=False))

    # Hiển thị bảng CSV đã được dự đoán
    if request.method == 'POST':
        f = request.form['csvfile']
        path = "static/upload/" + f
        data = []
        with open(path) as file:
            csvfile = csv.reader(file)
            next(file)
            for row in csvfile:
                final_features = [np.array(row)]
                prediction = model.predict(final_features)
                output = round(prediction[0], 2)
                if(output==1):
                    row.append('1')
                    data.append(row)
                if(output==0):
                    row.append('0')
                    data.append(row)
        data.insert(0,['Atr1','Atr2','Atr3','Atr4','Atr5','Atr6','Atr7','Atr8','Atr9','Atr10',
                        'Atr11','Atr12','Atr13','Atr14','Atr15','Atr16','Atr17','Atr18','Atr19','Atr20',
                        'Atr21','Atr22','Atr23','Atr24','Atr25','Atr26','Atr27','Atr28','Atr29','Atr30',
                        'Atr31','Atr32','Atr33','Atr34','Atr35','Atr36','Atr37','Atr38','Atr39','Atr40',
                        'Atr41','Atr42','Atr43','Atr44','Atr45','Atr46','Atr47','Atr48','Atr49','Atr50',
                        'Atr51','Atr52','Atr53','Atr54','Class'])
        data = pd.DataFrame(data)
        return render_template('result.html', prediction_text_csv=data.to_html(header=False))

if __name__ == "__main__":
    app.run(debug=True)