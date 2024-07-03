# app.py

from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
# Import necessary functions or classes from project.py
from project import predict_patient_status

# Initialize Flask app
app = Flask(__name__)

# Define route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Define route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    age = int(request.form["age"])
    gender = request.form["gender"]
    protein1 = float(request.form["protein1"])
    protein2 = float(request.form["protein2"])
    protein3 = float(request.form["protein3"])
    protein4 = float(request.form["protein4"])
    tumour_stage = request.form["tumour_stage"]
    histology = request.form["histology"]
    er_status = request.form["er_status"]
    pr_status = request.form["pr_status"]
    her2_status = request.form["her2_status"]
    surgery_type = request.form["surgery_type"]
    # Additional parameters can be added here
    
    # Make prediction using the predict_patient_status function from project.py
    prediction = predict_patient_status({
        'Age': age,
        'Gender': gender,
        'Protein1': protein1,
        'Protein2': protein2,
        'Protein3': protein3,
        'Protein4': protein4,
        'Tumour_Stage': tumour_stage,
        'Histology': histology,
        'ER status': er_status,
        'PR status': pr_status,
        'HER2 status': her2_status,
        'Surgery_type': surgery_type
    })
    
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
