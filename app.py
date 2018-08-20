from flask import Flask,render_template,request
import os

from digit_recognizer import digit

app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def index():
    return render_template('digit.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    try:
        imgData = request.get_data()
        out = digit(imgData)
        li=str(out)
        return li
    except Exception as e:
        return (str(e))


if __name__ == '__main__':
    app.run(debug=False)
