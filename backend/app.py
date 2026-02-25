from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load ML models
burnout_model = joblib.load("../models/burnout_model.pkl")
happiness_model = joblib.load("../models/happiness_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/burnout")
def burnout_page():
    return render_template("burnout.html")

@app.route("/happiness")
def happiness_page():
    return render_template("happiness.html")

# ---------------------- BASIC HAPPINESS PREDICTOR ----------------------
@app.route("/predict_happiness_basic", methods=["POST"])
def predict_happiness_basic():
    data = [
        float(request.form["work_life_balance"]),
        float(request.form["job_satisfaction"]),
        float(request.form["sleep_hours"]),
        float(request.form["stress_level"]),
        float(request.form["mental_health_score"]),
    ]

    prediction = happiness_model.predict([data])[0]
    return render_template("result.html", title="Happiness Result", prediction=prediction)

# ---------------------- ADVANCED HAPPINESS PREDICTOR ----------------------
@app.route("/predict_happiness_advanced", methods=["POST"])
def predict_happiness_advanced():
    data = [
        float(request.form["income"]),
        float(request.form["sleep_hours"]),
        float(request.form["social_support"]),
        float(request.form["gratitude_level"]),
        float(request.form["exercise_minutes"]),
        float(request.form["stress_level"]),
    ]

    prediction = happiness_model.predict([data])[0]
    return render_template("result.html", title="Happiness Result", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
