# CIA-3 — ML for Social Good: Breast Cancer Risk Screening

**Course:** MCA 521-4 Machine Learning | 
**Author:** Angel Blessy | **Guide:** Dr. Helen K Joy
**Institution:** CHRIST (Deemed to be University), Department of Computer Science
**Mission track:** Mission Health

## What this is

This is the CIA-3 submission for MCA 521-4. The brief was to pick a real-world "Mission" and build an ensemble-based ML solution for it — not just train a model and report accuracy, but frame the problem, compare against a baseline honestly, explain the model's predictions, and actually think through the ethics instead of bolting on a token paragraph at the end.

I went with Mission Health and built a model that screens breast biopsy measurements for malignancy risk. There are three pieces to it:

- `CIA_3_ML_breast_cancer.ipynb` — the actual notebook, and the main thing being graded. It goes from problem framing to a working, explainable model.
- `app.py` — a small Streamlit app so the model isn't just numbers in a notebook. You can type in measurements and get a live prediction with an explanation.
- `pipeline.py` — the training part of the notebook (data prep, model building, saving the artifacts) pulled out into a plain `.py` script, so it can be run from the terminal instead of clicking through notebook cells.

## The problem I was solving

When a patient gets a breast lump checked, a common first step is a fine-needle aspirate (FNA) biopsy — a small sample of cells gets pulled out and looked at under a microscope. A pathologist measures things like how big the cell nuclei are, how irregular their shape is, how rough the texture looks, and from that decides malignant or benign. A skilled pathologist is accurate at this, but it takes time and experience, and not every clinic has one readily available.

So the idea is pretty simple: train a model on those same measurements so it can give a fast second opinion. It flags likely-malignant cases for priority review — it doesn't replace the pathologist, it just helps triage. I kept coming back to that distinction throughout the project because it matters a lot ethically (more on that further down).

No single measurement tells you malignant vs. benign on its own — it's the combination of radius, texture, concavity, and so on that matters, and that interaction is exactly what makes this a reasonable ML problem instead of a simple threshold rule. And plain ensembles genuinely pull their weight here: once I made the data realistically noisy instead of the too-clean version scikit-learn ships, a simple Logistic Regression was still fine, but Random Forest, XGBoost, and a stacked combination of the two beat it on every metric I measured, not just accuracy.

## Dataset

I used the Breast Cancer Wisconsin (Diagnostic) dataset — the classic one, originally from Wolberg, Street, and Mangasarian (1995), pulled in via `sklearn.datasets.load_breast_cancer` (same data as what's on UCI) and exported to `breast_cancer.csv`.
Citation: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

It's 569 rows, 30 numeric features plus the diagnosis label. No missing values, no duplicates, no PII of any kind — just numbers derived from imaging. The target is a bit imbalanced but not badly: 357 benign (62.7%) vs 212 malignant (37.3%), close enough that I didn't need to do anything special like class weighting.

The 30 features aren't 30 different things — they're 10 actual measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each reported three ways: the mean value, the standard error, and the "worst" (largest) value seen across the cells in that sample.

## How I actually built this, step by step

**Loading and checking the data.** Straightforward — read the CSV, confirm the shape (569 × 31), check for missing values and duplicates (there were none), and look at how balanced the target is.

**Preprocessing.** First I turned the text `diagnosis` column into a plain 0/1 target. Then the more interesting part: the raw dataset is unrealistically clean, because it all came from one lab with one piece of imaging equipment back in 1995. If you train on it as-is, every model — even a basic one — scores 98–100%, which isn't a very honest test of anything. So I added calibrated Gaussian noise to every feature (scaled to 0.6× that feature's own standard deviation, clipped at zero since these are physical measurements that obviously can't go negative) before doing the train/test split. This is a normal technique for simulating the kind of variation you'd actually see between different equipment and different institutions, and it's what turns this into a problem where model choice and tuning actually matter, instead of everything just scoring near-perfect.

**Exploratory analysis.** After adding the noise, I checked which features still correlated with the target. The size and shape features — `worst radius`, `worst concave points`, and similar — stayed the strongest predictors, just weaker than before (correlation dropped from roughly 0.7–0.8 down to 0.5–0.6). I also plotted the target distribution and a correlation heatmap of the more important features, just to sanity-check everything visually before moving on.

**Feature engineering.** I added two new features on top of the original 30. `radius_growth_ratio` is just `worst radius` divided by `mean radius` — basically, how much bigger the biggest cell got compared to the average cell in that sample. It turned out to actually matter — it shows up fairly high in XGBoost's SHAP importance later, so it's not just a feature I added for the sake of ticking a box. I also bucketed `mean radius` into `size_category` (Small/Medium/Large) using tertiles from the training data, mainly to satisfy the categorical-feature requirement, but it also happens to show a very clean risk gradient across the three buckets.

**Train/test split.** I used a 60/40 split instead of the usual 80/20, stratified on the target, with `random_state=42` fixed so it's reproducible. I went bigger on the test set (228 rows) on purpose — I was comparing four models closely, and a small test set makes Precision/Recall/F1 noisy enough that small differences between models stop being meaningful.

**Scaling and encoding.** I built one `ColumnTransformer` — `StandardScaler` on the numeric features, `OneHotEncoder` on `size_category` — but instead of applying it once up front, I wrapped it inside every model's own scikit-learn `Pipeline`. That way, during cross-validation and hyperparameter search, the scaler gets refit separately on each fold's training data, so there's no leakage from the validation fold sneaking into the training statistics.

**Building and tuning the models.** Four models total, all evaluated on the exact same held-out test set so the comparison is fair:

- A Logistic Regression baseline, kept simple (just `max_iter=1000` bumped up so it actually converges). I picked Logistic Regression over a Decision Tree for the baseline because the Decision Tree turned out to be pretty unstable once the noise was added.
- A Random Forest, tuned with `RandomizedSearchCV` (40 iterations, 5-fold CV, optimizing ROC-AUC) over the usual suspects — number of trees, max depth, leaf size, feature sampling, split criterion.
- An XGBoost model, tuned the same way but with 15 iterations, over learning rate, number of trees, max depth, and the subsampling parameters.
- A Stacking ensemble that combines the fitted Logistic Regression, Random Forest, and XGBoost, with another Logistic Regression on top as the meta-learner (using internal 5-fold CV so the meta-learner isn't just learning to trust whichever base model memorized the training data best).

One thing I had to deal with: a model having a higher ROC-AUC doesn't automatically mean its default 0.5 probability cutoff gives good accuracy or recall. So for each ensemble I picked a decision threshold off its own ROC/PR curve instead of leaving it at 0.5 — Random Forest ended up at 0.38, XGBoost at 0.20, Stacking at 0.25. I checked each of these against nearby threshold values too, just to make sure I hadn't landed on some lucky knife-edge number.

**Comparing the models.** Once everything was trained and thresholded, I put all four side by side — same metrics, same test set, plus confusion matrices for each. Results are below.

**Explaining the predictions.** I used SHAP on the tuned XGBoost model to understand what it's actually paying attention to — both overall (which features matter most across the whole test set) and for individual predictions (why did this one specific sample get scored as high risk).

**Ethics.** I wrote this as its own real section, not an afterthought — see further down.

**Saving everything and building the app.** Last step was saving the four fitted models plus a small metadata file (per-feature min/max/median, and the calibrated thresholds) so the Streamlit app could load them directly instead of retraining every time someone opens it.

## Results

All four models scored on the same 228-row test set:

| Model | ROC-AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Baseline (Logistic Regression) | 0.9880 | 0.9474 | 0.9286 | 0.9398 | 0.9176 |
| Bagging (Random Forest) | 0.9929 | 0.9693 | 0.9586 | 0.9643 | 0.9529 |
| Boosting (XGBoost) | 0.9896 | 0.9649 | 0.9524 | 0.9639 | 0.9412 |
| Stacking Ensemble | 0.9917 | 0.9693 | 0.9586 | 0.9643 | 0.9529 |

The main takeaway is that all three ensembles beat the plain Logistic Regression on every metric, not just accuracy — that was actually the point of using ensembles here rather than just picking the simplest thing that works. Random Forest edges out everything on ROC-AUC, but it's basically tied with Stacking overall (both hit 96.9% accuracy). Nothing scores 100% anywhere, which is expected given the noise added earlier — a perfect score would honestly be a red flag, not a good sign.

## Why the model predicts what it predicts

I used SHAP on the tuned XGBoost model to answer the "why" question, at two levels.

Globally, across the whole test set, the features the model leans on most are `worst radius`, `worst perimeter`, `worst texture`, and `worst concave points` — which lines up with what a pathologist would actually look at. The engineered `radius_growth_ratio` feature shows up too, just lower down the ranking than it would've been before I added the noise.

For any individual prediction, I can also generate a waterfall plot showing exactly which measurements pushed that specific patient's risk score up or down — for example, an unusually large `worst radius` pushing risk up, or small, regular measurements pulling it down. This isn't just for the notebook — the app generates one of these live for every prediction someone makes, so it's never a black-box number with nothing behind it.

## Ethics, fairness, and where this falls short

I tried to be honest about this rather than gloss over it:

A false negative here — a malignant tumor scored as low risk — is worse than a false positive, because it delays real treatment. A false positive causes anxiety and an unnecessary follow-up biopsy, which isn't great but isn't dangerous. With the thresholds I picked, recall sits around 92–95% across the models, meaning even the best model still misses a few real malignant cases in this test set. That's exactly why this is meant to support a pathologist's decision, not replace it.

There's also no way to check for fairness across patient groups here, because the dataset simply doesn't have any demographic fields — no age, no ethnicity, nothing. I want to be upfront that this is a real gap, not something I'm glossing over as a strength. If this were ever actually deployed, that audit would need to happen first.

Privacy isn't really a concern — everything here is de-identified numeric measurements, no personal information at all.

The bigger limitation is just that this dataset is small, old, and from a single institution — 569 samples, one hospital, 1995-era equipment. The noise I added simulates some of that variability, but it's not a substitute for testing on real, current, diverse clinical data.

And to repeat the point from earlier: this is a screening aid, not a diagnosis. Nobody should be treated based on this model's output alone — a pathologist needs to confirm it through an actual biopsy review.

## The app

`app.py` runs on Streamlit and is called "BreastScan AI" in the UI. It doesn't train anything itself — it just loads whatever the notebook (or `pipeline.py`) saved and serves predictions from that.

Opening it, you get a collapsible panel showing the same model comparison and confusion matrices as the notebook, so you can see how trustworthy the thing is before you even use it. Below that is a form for entering the 10 mean-value measurements — the error and worst-value measurements are optional, and if you skip them the app just fills in realistic median values instead of forcing you through all 30 fields. The input ranges on each field aren't hardcoded either — they're pulled from the training data's actual min/max (with a bit of padding), so they can't silently go stale if the underlying data ever changes.

Once you submit, you get a prediction card (lower risk / high risk, the actual probability, and a confidence score, using the Stacking model's calibrated 0.25 threshold rather than a plain 0.5 cutoff), a live SHAP waterfall plot explaining that specific prediction, and the same ethics disclaimers from the notebook — shown every time, not just buried somewhere you'd only read once.

## Files in this project

| File | What it is |
|---|---|
| `CIA_3_ML_breast_cancer.ipynb` | The main notebook — everything from problem framing to ethics |
| `pipeline.py` | The training pipeline as a plain script, for running outside Jupyter |
| `breast_cancer.csv` | The raw dataset |
| `app.py` | The Streamlit demo app |
| `bc_model_stacking.pkl` | The fitted Stacking model (what the app actually predicts with) |
| `bc_model_rf.pkl` | The fitted, tuned Random Forest |
| `bc_model_xgb_for_shap.pkl` | The fitted, tuned XGBoost (also used for the app's SHAP explanations) |
| `bc_model_logreg.pkl` | The fitted Logistic Regression baseline |
| `bc_feature_metadata.json` | Per-feature min/max/median plus each model's calibrated threshold |
| `.streamlit/config.toml` | Forces a light theme for the app |
| `README.md` | This file |

## Running it yourself

Install what you need:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn streamlit altair joblib
```

To go through the full analysis the way it was built, open the notebook:

```bash
jupyter notebook CIA_3_ML_breast_cancer.ipynb
```

and run it top to bottom. Everything's seeded (`random_state=42`), so you'll get the exact same numbers I got. Running it to the end also re-saves the model files the app uses.

If you'd rather skip Jupyter and just regenerate the same models from the terminal:

```bash
python pipeline.py
```

This runs the same preprocessing, tuning, and stacking logic, prints the metrics as it goes, and re-saves the same artifact files.

To try the app itself:

```bash
python -m streamlit run app.py
```

It opens at `http://localhost:8501`. Fill in some measurements (or just leave the pre-filled median values) and hit **Predict Malignancy Risk**.

> If `streamlit run app.py` doesn't work because it's not on your PATH, use `python -m streamlit run app.py` instead — that always works.

## What I used

Pandas and numpy for data handling; scikit-learn for the pipelines, preprocessing, Logistic Regression, Random Forest, Stacking, and hyperparameter search; XGBoost for the boosting model; SHAP for explainability; matplotlib and seaborn for the notebook's plots; Altair for the app's live chart; Streamlit for the app itself; joblib and json for saving everything.

## A note on academic integrity

The dataset is public and cited above, and contains no personal information. All the modeling code here was written by me for this assignment — the libraries used (scikit-learn, XGBoost, SHAP, Streamlit, Altair) are all standard, documented, open-source tools used the way their own docs describe.
