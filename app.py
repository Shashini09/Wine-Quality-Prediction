import joblib
from flask import Flask, render_template, request
import numpy as np

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    # Get input data from the form
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    
    # Convert input data to a 2D array
    arr = np.array([[float(data1), float(data2), float(data3), float(data4), float(data5)]])
    
    # Scale the input data using the loaded scaler
    scaled_arr = scaler.transform(arr)

    # Make prediction
    pred = model.predict(scaled_arr)

    pred = int(pred[0])

    # Convert prediction to readable format
    if pred == 1:  # Assuming 1 represents "Pass"
        result = "Pass"
    else:
        result = "Fail"

    # Render the result to after.html
    return render_template('after.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
