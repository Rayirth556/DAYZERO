print("🚀 Starting Flask app...")

from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
from model.data_preprocessing import train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score

app = Flask(__name__)

DATA_FILE = "data.csv"
MODEL_FILE = "model/saved/stock_predictor.pkl"
PREPROCESSOR_FILE = "model/saved/preprocessor.pkl"

def get_table_html():
    df = pd.read_csv(DATA_FILE)
    return df.to_html(classes="data", index=False, border=0)

@app.route("/")
def index():
    return render_template("index.html", table_html=get_table_html())

@app.route("/submit", methods=["POST"])
def submit():
    try:
        new_data = {
            "name": request.form["name"],
            "salary": float(request.form["salary"]),
            "position": request.form["position"],
            "yoe": int(request.form["yoe"]),
            "bonus_percent": float(request.form["bonus_percent"]),
            "investment_expert": request.form["investment_expert"],
            "label": "unknown"
        }
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        return ("", 204)
    except Exception as e:
        return f"❌ Error in /submit: {str(e)}", 400

@app.route("/table")
def table():
    return get_table_html()

@app.route("/analyze")
def analyze():
    try:
        train_model()
        model = joblib.load(MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        df = pd.read_csv(DATA_FILE)

        # Clean labels
        df = df[~df["label"].astype(str).str.contains("unknown", na=False)]
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        if not df.empty:
            X = df.drop(columns=["name", "label"])
            y = df["label"]

            X_preprocessed = preprocessor.transform(X)
            preds = model.predict(X_preprocessed)

            accuracy = accuracy_score(y, preds)
            precision = precision_score(y, preds, average="macro")
            recall = recall_score(y, preds, average="macro")

            summary = f"""
            <h3>📊 Stock Insight Summary</h3>
            <ul>
                <li><strong>Accuracy:</strong> {accuracy:.2f}</li>
                <li><strong>Precision:</strong> {precision:.2f}</li>
                <li><strong>Recall:</strong> {recall:.2f}</li>
            </ul>
            """
        else:
            summary = "<p>Not enough labeled data to analyze.</p>"

    except Exception as e:
        summary = f"<p>❌ Error during analysis: {str(e)}</p>"

    return render_template("analyze.html", summary=summary)

@app.route("/fetch_person")
def fetch_person():
    try:
        name = request.args.get("name")
        df = pd.read_csv(DATA_FILE)
        person = df[df["name"] == name].to_dict(orient="records")
        return person[0] if person else {}
    except Exception as e:
        return {"error": str(e)}

@app.route("/download_csv")
def download_csv():
    try:
        return send_file(DATA_FILE, as_attachment=True)
    except Exception as e:
        return f"❌ Error downloading CSV: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
