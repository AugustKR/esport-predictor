# Esports Performance Predictor

Two tabs, two different analytical framings of the same dataset:

<<<<<<< HEAD
- **Personal Zone** (default tab): centers the analysis on a single individual.
  Enter your age, energy drinks consumed, and game genre, and see your
  estimated heart rate and which "zone" it falls in, compared to real players
  your age in the dataset. Answers questions like the course feedback's
  example: *"I'm 45, I play MOBA, how many energy drinks before I enter the
  Challenge Zone?"* Sleep_Hours is deliberately excluded here: it was tested
  and has essentially zero relationship with Heart_Rate_BPM in this dataset
  (correlation -0.0004, flat ~84 bpm average across every sleep-hour group),
  so a sleep slider would move but have no real effect on the output, which
  would be misleading in a tool framed around one person's real risk.
- **Population Explorer**: the original exploratory view, predicting **APM**
  and **Reaction Time** as population-average relationships across all 250,000
  players. Useful for showing the aggregate patterns, but intentionally kept
  separate from the individual-centered Personal Zone tab.

Both are powered by linear regressions fit on `esports_gaming_biometrics_250k.csv`
in `analysis.py`.

## Course feedback that shaped this design

Initial feedback was that an exploratory, population-level analysis (e.g. "on
average, more energy drinks -> higher APM") doesn't center the data around an
individual, which this course is about: the datafication of an individual's
own actions, not aggregate statistics. The Personal Zone tab is the response
to that: the model's prediction is anchored to one person's age/sleep/genre,
and framed against where that prediction falls relative to other players their
age (age-bucketed percentiles: p50/p75/p90/p95 of Heart_Rate_BPM), rather than
reporting an average effect across everyone.

This also surfaces, directly in the UI, the observed-vs-not-observed point
from the feedback: the model only knows age, energy drinks and genre. It has
no idea about this specific person's caffeine tolerance, actual resting heart
rate, medication, anxiety, sleep, or prior habits, so "your zone" really means
"someone with your observed traits", not a personalized medical reading. That
gap is stated explicitly in the Personal Zone tab, not just in this README.

## Why we checked and removed Sleep_Hours from the Personal Zone model

We asked the same "does this seem realistic" question of our own feature
choices. Sleep_Hours is a strong predictor of Reaction_Time_ms (correlation
-0.39) but its correlation with Heart_Rate_BPM is -0.0004, essentially zero,
confirmed by grouping average heart rate by rounded sleep hours (flat at
~84 bpm from 2 to 11 hours of sleep) and by its regression coefficient
(-0.007 bpm/hour, Heart_Rate_BPM model R² unchanged with or without it:
0.2898 vs 0.2911). We also checked PlayTime_Hours, Screen_Brightness,
Blue_Light_Filter and HRV_Score as possible additions to the Heart_Rate_BPM
model: all showed negligible or near-zero relationships (|correlation| <
0.12, R² gains under 0.002). Age, Energy_Drinks and Game_Genre remain the
only three variables in this dataset with a real, usable relationship to
heart rate, so the Personal Zone tab is built on exactly those three.

## Why APM / Reaction Time instead of Win/Loss (Population Explorer tab)

Match_Outcome (Win/Loss) is essentially unpredictable from these inputs: a
logistic regression combining Age, Sleep_Hours, Energy_Drinks and Game_Genre
only reaches an AUC of ~0.53 (0.50 = random chance). This matches how
competitive matchmaking is designed to push win rates toward 50% regardless
of a single player's biometrics.

APM and Reaction_Time_ms, in contrast, have real, explainable relationships
with these inputs:

- **APM** — test R² = 0.68. Driven mainly by Game_Genre (Fighting/MOBA much
  higher than Battle Royale) and Energy_Drinks (+10 APM per drink).
- **Reaction_Time_ms** — test R² = 0.25. Driven mainly by Sleep_Hours
  (less sleep = slower reactions) and Age (older = slightly slower).

See `analysis.py` for the full model-fitting code and coefficients.

=======
>>>>>>> b41f92ab96c75b63d2abbad3c36e9b282000f350
## How to run

```bash
pip install -r requirements.txt
python3 analysis.py   # trains the model, writes model.json
python3 app.py         # starts the web app
```

Then open http://127.0.0.1:5000 in a browser. Move the sliders / change the
genre to see predicted APM and reaction time update live.

## Files

- `analysis.py` — loads the dataset, fits the regression models (train/test
  split, R², residual std) for APM, Reaction_Time_ms and Heart_Rate_BPM, and
  computes the age-bucketed percentile tables used by the Personal Zone tab.
  Writes `model.json`.
- `app.py` — Flask backend exposing `/api/predict` (Population Explorer),
  `/api/personal-zone` (Personal Zone), and serving the frontend.
- `index.html` — tabbed frontend: Personal Zone (sliders + zone gauge) and
  Population Explorer (sliders + live gauges for APM and reaction time).
- `model.json` — generated by `analysis.py` (coefficients, R², age-bucket
  percentiles, feature ranges).
- `data/esports_gaming_biometrics_250k.csv` — source dataset.

<<<<<<< HEAD
## Known limitations (worth mentioning in the report)

- Reaction_Time_ms R² (0.25) is moderate, not strong — the Population
  Explorer tab shows a shaded uncertainty range (±1 residual std) rather than
  a single confident number.
- Heart_Rate_BPM R² (0.29) is also moderate; the Personal Zone tab's "zone"
  should be read as an estimate for someone with these observed traits, not a
  precise individual prediction (see the in-app disclaimer).
- The 36-40 age bucket has only 127 players in the dataset (vs 50,000+ for
  the younger buckets), so its percentile thresholds are noisier. Worth
  flagging if your demo uses an older age, like the "I'm 45" example (clamped
  to the 36-40 bucket since the dataset only covers ages 16-40).
- The dataset is synthetic (confirmed with the course lecturer), so absolute
  coefficient values should be discussed as patterns in this dataset, not
  claims about real human physiology.
- The model is a simple linear regression with no interaction terms (e.g.
  Energy_Drinks × Game_Genre); that's a natural "future work" extension.
=======
>>>>>>> b41f92ab96c75b63d2abbad3c36e9b282000f350
