"""
Digital Data Analysis - Exam Project
Flask backend serving performance predictions.

Run with:
    python3 app.py

Then open http://127.0.0.1:5000 in a browser.

Requires model.json, produced by running analysis.py first.

----------------------------------------------------------------------------
IN PLAIN LANGUAGE: what this file does
----------------------------------------------------------------------------
This is the "backend" - the part of the website that runs on the server and
does the actual math. The "frontend" (index.html) is what the user sees and
clicks on in their browser; whenever a slider moves, the frontend sends a
request here asking "given these inputs, what's the prediction?", and this
file sends back an answer as JSON (a simple text format for structured data).

Flask is a small Python library for building exactly this kind of web app:
you write a function, decorate it with @app.route("/some/url"), and Flask
makes that function run whenever a browser visits that URL.

This file never re-does the statistics itself - it just loads the already-
computed formula (model.json, produced by analysis.py) once when it starts,
and then plugs numbers into that formula every time someone asks for a
prediction. That's why starting the site requires running analysis.py first.

There are three "routes" (URLs) a browser can ask for:
  /                    -> sends back the website itself (index.html)
  /api/predict         -> used by the "Population Explorer" tab: given age,
                           sleep, energy drinks and genre, predicts APM and
                           reaction time (the population-average view)
  /api/personal-zone   -> used by the "Personal Zone" tab: given age, energy
                           drinks and genre, predicts heart rate AND says
                           which "zone" (Normal / Elevated / Challenge /
                           Risk) that falls into for players that age
----------------------------------------------------------------------------
"""

import json
import os

from flask import Flask, jsonify, request, send_from_directory

# Figure out the folder this file lives in, so we can find model.json and
# index.html next to it no matter where the script is run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.json")

# Set up the Flask app. static_folder=BASE_DIR means Flask is allowed to
# serve any file in this folder directly (that's how index.html gets sent
# to the browser when someone visits "/").
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

# Load the pre-computed model (coefficients, R² scores, age-group tables)
# ONE time when the server starts, and keep it in memory as MODEL. This
# means every prediction request is basically instant - it's just plugging
# numbers into a formula we already have, not re-reading the 250,000-row
# CSV or re-fitting anything.
with open(MODEL_PATH) as f:
    MODEL = json.load(f)


def predict_target(target_name, genre, **feature_values):
    """Plug a player's numbers into one of our regression formulas and
    return the predicted value (e.g. predicted APM, or predicted heart
    rate).

    Reminder of what a regression formula looks like:
        prediction = Intercept + (coefficient_1 * input_1)
                                + (coefficient_2 * input_2) + ...
                                + genre_adjustment

    - target_name is which formula to use: "APM", "Reaction_Time_ms" or
      "Heart_Rate_BPM".
    - genre is the player's chosen game genre (e.g. "MOBA").
    - feature_values are the actual slider numbers, passed in as e.g.
      Age=24, Sleep_Hours=7, Energy_Drinks=2. We only use the ones this
      particular target actually needs - the list of which features a
      target uses is stored in model.json as "features_used". That's why
      Heart_Rate_BPM happily ignores a Sleep_Hours value even if one is
      passed in: it was trained without that feature (see analysis.py for
      why), so it has no coefficient for it to multiply by.
    - The genre adjustment: one genre (Battle Royale) is the "baseline"
      with no adjustment. Every other genre has its own coefficient that
      gets added on top, representing "how much higher/lower than Battle
      Royale players in this genre tend to be".
    """
    target = MODEL["targets"][target_name]
    coefs = target["coefficients"]
    value = coefs["Intercept"]
    for feat in target["features_used"]:
        value += coefs[feat] * feature_values[feat]

    baseline_genre = MODEL["baseline_genre"]
    if genre != baseline_genre:
        value += coefs.get(genre, 0.0)

    return value


def find_age_bucket(age):
    """Look up which 5-year age group (e.g. "26-30") a given age falls
    into, using the age_bucket tables analysis.py already calculated.

    The dataset only has players aged 16-40. If someone enters an age
    outside that (like the professor's "I'm 45" example), we don't have
    real data for that exact age, so we "clamp" them to the closest group
    we DO have data for (here, the oldest group, 36-40) rather than
    guessing wildly or crashing.
    """
    buckets = MODEL["heart_rate_age_buckets"]
    for b in buckets:
        if b["age_min"] <= age <= b["age_max"]:
            return b
    if age < buckets[0]["age_min"]:
        return buckets[0]
    return buckets[-1]


def classify_zone(hr_value, bucket):
    """Decide which "zone" a predicted heart rate falls into, by comparing
    it to the real percentile values for that age group (p50 = median,
    p75, p90, p95 - see analysis.py for how these were calculated).

    In plain terms:
      - below the age group's median (p50)       -> "Below Average"
      - between median and 75th percentile        -> "Normal"
      - between 75th and 90th percentile           -> "Elevated"
      - between 90th and 95th percentile           -> "Challenge Zone"
      - above the 95th percentile                   -> "High Intensity Zone"
    So "Challenge Zone" means: higher than 90% of real players that age in
    this dataset, but not yet in the most extreme 5%.

    Returns two things: a short "key" (used by the website to pick a
    color) and a human-readable "label" (the text shown to the user).
    """
    if hr_value < bucket["p50"]:
        return "resting", "Below Average"
    if hr_value < bucket["p75"]:
        return "normal", "Normal"
    if hr_value < bucket["p90"]:
        return "elevated", "Elevated"
    if hr_value < bucket["p95"]:
        return "challenge", "Challenge Zone"
    return "risk", "High Intensity Zone"


def drinks_to_reach_threshold(age, genre, threshold):
    """Answer the question "how many energy drinks until I reach this
    heart-rate threshold?" by simply trying every possible whole number of
    drinks (0, 1, 2, 3, 4 - the only values actually observed in the
    dataset) and checking the predicted heart rate for each one.

    This is brute force rather than solving the equation algebraically,
    but with only 5 possible values to check it's instant and much easier
    to read than the algebra would be.

    Returns the first (smallest) number of drinks where the prediction
    reaches the threshold, or None if even 4 drinks isn't enough to get
    there (meaning: within the range of data we have, this profile never
    reaches that zone).
    """
    max_observed_drinks = 4
    for drinks in range(0, max_observed_drinks + 1):
        hr = predict_target("Heart_Rate_BPM", genre, Age=age, Energy_Drinks=drinks)
        if hr >= threshold:
            return drinks
    return None  # not reached within the observed 0-4 drink range


@app.route("/")
def index():
    """When someone opens the website's homepage, just send them the
    index.html file as-is. All the actual slider/chart logic lives in that
    file's own <script> tag and runs in the browser, not here."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/model-info")
def model_info():
    """Hand the whole model.json file to the browser. The frontend uses
    this on page load just to show "Model trained on 250,000 rows" - it's
    not strictly needed for predictions, just a nice bit of transparency."""
    return jsonify(MODEL)


@app.route("/api/predict")
def predict():
    """This is the endpoint the "Population Explorer" tab calls every time
    a slider moves. It reads the four inputs from the URL (Flask puts
    anything after "?" in request.args), predicts both APM and reaction
    time, and sends the results back as JSON.

    Example of what the browser actually requests:
        /api/predict?age=24&sleep_hours=7&energy_drinks=2&genre=FPS
    """
    try:
        # request.args.get("age", 25) means "read the 'age' value from the
        # URL, or use 25 if it's missing". float(...) converts the text
        # from the URL into an actual number we can do math with.
        age = float(request.args.get("age", 25))
        sleep_hours = float(request.args.get("sleep_hours", 7))
        energy_drinks = float(request.args.get("energy_drinks", 1))
        genre = request.args.get("genre", MODEL["baseline_genre"])
    except (TypeError, ValueError):
        # If someone sends something that isn't a valid number (e.g. the
        # text "abc" instead of a number), float() throws an error - we
        # catch it here and reply with a clear error message instead of
        # crashing the whole server.
        return jsonify({"error": "Invalid input"}), 400

    if genre not in MODEL["genres"]:
        return jsonify({"error": f"Unknown genre: {genre}"}), 400

    # Run both formulas with the same inputs.
    apm = predict_target("APM", genre, Age=age, Sleep_Hours=sleep_hours, Energy_Drinks=energy_drinks)
    reaction = predict_target("Reaction_Time_ms", genre, Age=age, Sleep_Hours=sleep_hours, Energy_Drinks=energy_drinks)

    # Pull out each model's accuracy (R²) and typical error size
    # (residual_std), which analysis.py already calculated and saved.
    apm_r2 = MODEL["targets"]["APM"]["test_r2"]
    reaction_r2 = MODEL["targets"]["Reaction_Time_ms"]["test_r2"]
    apm_residual_std = MODEL["targets"]["APM"]["residual_std"]
    reaction_residual_std = MODEL["targets"]["Reaction_Time_ms"]["residual_std"]

    # Send everything back as JSON. "low"/"high" give the frontend a
    # "give or take this much" range to display (prediction ± typical
    # error), rather than showing a single number as if it were exact.
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


@app.route("/api/personal-zone")
def personal_zone():
    """This is the endpoint the "Personal Zone" tab calls every time its
    sliders move. Unlike /api/predict above, this one is about ONE
    individual: it predicts their heart rate, then explains what that
    means by comparing it to real players their age (see classify_zone
    and find_age_bucket above), and works out how many energy drinks it
    would take to cross into the Challenge Zone or the Risk Zone.

    Example request:
        /api/personal-zone?age=45&energy_drinks=2&genre=MOBA
    (Note: no sleep_hours here - see analysis.py for why Heart_Rate_BPM's
    formula doesn't use sleep at all.)
    """
    try:
        age = float(request.args.get("age", 25))
        energy_drinks = float(request.args.get("energy_drinks", 1))
        genre = request.args.get("genre", MODEL["baseline_genre"])
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    if genre not in MODEL["genres"]:
        return jsonify({"error": f"Unknown genre: {genre}"}), 400

    # 1. Which age group's percentile table do we compare this person to?
    bucket = find_age_bucket(age)

    # 2. What heart rate does our formula predict for them right now?
    hr_value = predict_target("Heart_Rate_BPM", genre, Age=age, Energy_Drinks=energy_drinks)

    # 3. Which zone does that predicted heart rate fall into?
    zone_key, zone_label = classify_zone(hr_value, bucket)

    # 4. At their current age and genre, how many energy drinks would it
    #    take to reach the Challenge Zone (90th percentile) or the Risk
    #    Zone (95th percentile)? This directly answers the "how many
    #    energy drinks before I enter the challenge zone?" style question
    #    from the course feedback.
    challenge_drinks = drinks_to_reach_threshold(age, genre, bucket["p90"])
    risk_drinks = drinks_to_reach_threshold(age, genre, bucket["p95"])

    return jsonify({
        "heart_rate_bpm": round(hr_value, 1),
        "zone_key": zone_key,
        "zone_label": zone_label,
        "age_bucket": f"{bucket['age_min']}-{bucket['age_max']}",
        "age_bucket_n": bucket["n"],  # how many real players in the dataset are in this age group
        "percentiles": {
            "p50": bucket["p50"], "p75": bucket["p75"],
            "p90": bucket["p90"], "p95": bucket["p95"],
        },
        "challenge_zone_drinks": challenge_drinks,
        "risk_zone_drinks": risk_drinks,
        "r2": MODEL["targets"]["Heart_Rate_BPM"]["test_r2"],
    })


if __name__ == "__main__":
    # This starts Flask's built-in development web server on port 5000, so
    # the site becomes available at http://127.0.0.1:5000 in a browser.
    # debug=True makes Flask auto-restart when you edit this file and show
    # more detailed error pages - handy for building the site, but this
    # setting should be turned off (debug=False) if it were ever deployed
    # somewhere other people could reach it.
    app.run(debug=True, port=5000)
