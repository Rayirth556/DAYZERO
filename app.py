from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)
DATA_FILE = "data.csv"

model = joblib.load("model/saved/stock_predictor.pkl")
preprocessor = joblib.load("model/saved/preprocessor.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    # Save data to CSV only
    entry = {
        "name": request.form["name"],
        "age": int(request.form["age"]),
        "salary": float(request.form["salary"]),
        "position": request.form["position"],
        "yoe": int(request.form["yoe"]),
        "bonus_percent": float(request.form["bonus_percent"]),
        "investment_expert": request.form["investment_expert"]
    }

    # Append to CSV
    df = pd.DataFrame([entry])
    if os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(DATA_FILE, index=False)

    return render_template("index.html", prediction=None)

@app.route("/analyze", methods=["POST"])
def analyze():
    # Load all data
    df = pd.read_csv(DATA_FILE)

    # Preprocess and predict
    X = preprocessor.transform(df.drop(columns=["name"]))
    preds = model.predict(X)

    inc = (preds == "increase").sum()
    dec = (preds == "decrease").sum()
    count = len(preds)

    summary = f"Suggest increasing stock for {inc} out of {count} entries."

    return render_template("index.html", prediction=summary, inc=inc, dec=dec, count=count)

if __name__ == "__main__":
    app.run(debug=True)
