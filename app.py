"""
Digital Data Analysis - Exam Project
Flask backend serving performance predictions.

Run with:
    python3 app.py

Then open http://127.0.0.1:5000 in a browser.

Requires model.json, produced by running analysis.py first.
"""

import json
import os

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.json")

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

with open(MODEL_PATH) as f:
    MODEL = json.load(f)


def predict_target(target_name, age, sleep_hours, energy_drinks, genre):
    coefs = MODEL["targets"][target_name]["coefficients"]
    value = coefs["Intercept"]
    value += coefs["Age"] * age
    value += coefs["Sleep_Hours"] * sleep_hours
    value += coefs["Energy_Drinks"] * energy_drinks

    baseline_genre = MODEL["baseline_genre"]
    if genre != baseline_genre:
        value += coefs.get(genre, 0.0)

    return value


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/model-info")
def model_info():
    return jsonify(MODEL)


@app.route("/api/predict")
def predict():
    try:
        age = float(request.args.get("age", 25))
        sleep_hours = float(request.args.get("sleep_hours", 7))
        energy_drinks = float(request.args.get("energy_drinks", 1))
        genre = request.args.get("genre", MODEL["baseline_genre"])
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    if genre not in MODEL["genres"]:
        return jsonify({"error": f"Unknown genre: {genre}"}), 400

    apm = predict_target("APM", age, sleep_hours, energy_drinks, genre)
    reaction = predict_target("Reaction_Time_ms", age, sleep_hours, energy_drinks, genre)

    apm_r2 = MODEL["targets"]["APM"]["test_r2"]
    reaction_r2 = MODEL["targets"]["Reaction_Time_ms"]["test_r2"]
    apm_residual_std = MODEL["targets"]["APM"]["residual_std"]
    reaction_residual_std = MODEL["targets"]["Reaction_Time_ms"]["residual_std"]

    return jsonify({
        "apm": {
            "value": round(apm, 1),
            "low": round(apm - apm_residual_std, 1),
            "high": round(apm + apm_residual_std, 1),
            "r2": apm_r2,
        },
        "reaction_time_ms": {
            "value": round(reaction, 1),
            "low": round(reaction - reaction_residual_std, 1),
            "high": round(reaction + reaction_residual_std, 1),
            "r2": reaction_r2,
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
