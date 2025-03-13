from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
weights = np.loadtxt("Weights.csv")

def sigmoid(z):
    result = 1/(1+np.exp(-z))
    return (result >= 0.5).astype(int)

@app.route('/test')
def test():
    data = request.form
    print(data)

    patient_data = np.ones((1,6))
    patient_data[0, 1] = float(data["box1"])
    patient_data[0, 2] = float(data["box2"])
    patient_data[0, 3] = int(data["box3"])
    patient_data[0, 4] = float(data["box4"])
    patient_data[0, 5] = int(data["box5"])

    prediction = sigmoid(np.dot(patient_data, weights))

    return render_template("index.html", result=prediction)

if __name__=="__main__":
    app.run(debug=True)