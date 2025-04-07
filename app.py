#mera naam murgo mera kaam chudno, dude rayirth please make your shit work man
from flask import Flask, request, render_template, jsonify, redirect, flash, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.secret_key = 'qwerty12345'  # 🔐 Use a long, random string in production

DATA_FILE = "data.csv"

# Load model and preprocessor
model = joblib.load("model/saved/stock_predictor.pkl")
preprocessor = joblib.load("model/saved/preprocessor.pkl")

# Helper: Generate HTML for current data table
def get_table_html():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        return df.to_html(classes="styled-table", index=False)
    return "<p>No data yet.</p>"

# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", table=get_table_html())

# Data submission endpoint
@app.route("/submit", methods=["POST"])
def submit():
    entry = {
        "name": request.form["name"],
        "age": int(request.form["age"]),
        "salary": float(request.form["salary"]),
        "position": request.form["position"],
        "yoe": int(request.form["yoe"]),
    }

    new_df = pd.DataFrame([entry])
    if os.path.exists(DATA_FILE):
        new_df.to_csv(DATA_FILE, mode='a', header=False, index=False)
    else:
        new_df.to_csv(DATA_FILE, index=False)

    return get_table_html()

# Analyze all data using the model
@app.route("/analyze", methods=["POST"])
def analyze():
    if not os.path.exists(DATA_FILE):
        return render_template("index.html", prediction="No data to analyze.", inc=0, dec=0, count=0, table="<p>No data yet.</p>")

    df = pd.read_csv(DATA_FILE)
    X = preprocessor.transform(df.drop(columns=["name"]))
    preds = model.predict(X)

    inc = (preds == "increase").sum()
    dec = (preds == "decrease").sum()
    count = len(preds)
    summary = f"Suggest increasing stock for {inc} out of {count} entries."

    return render_template("index.html", prediction=summary, inc=inc, dec=dec, count=count, table=get_table_html())

# Fetch a person's details by name
@app.route("/fetch_person", methods=["POST"])
def fetch_person():
    name_to_fetch = request.form["fetch_name"]

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        person = df[df["name"].str.lower() == name_to_fetch.strip().lower()]
        if not person.empty:
            person_info = person.iloc[-1].to_dict()  # Latest matching entry
            return render_template("index.html", person_info=person_info, table=get_table_html())
        else:
            return render_template("index.html", person_info={"Error": "Not Found"}, table=get_table_html())
    else:
        return render_template("index.html", person_info={"Error": "No data available"}, table="<p>No data yet.</p>")
    
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files.get('csv_file')
    
    if not file or file.filename == '':
        flash('No file selected')
        return redirect('/')

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)

        # Optional: Validate/trim columns
        expected_columns = ['name', 'age', 'salary', 'position', 'yoe', 'label', 'prediction']
        df = df[[col for col in expected_columns if col in df.columns]]

        # ❗ Replace the existing dataset
        df.to_csv('data.csv', index=False)

        flash(f'CSV uploaded and replaced successfully!')
        return redirect('/')

    flash('Invalid file type')
    return redirect('/')

@app.route('/update_entry', methods=['POST'])
def update_entry():
    name = request.form['name']
    age = int(request.form['age'])
    salary = int(request.form['salary'])
    position = request.form['position']
    yoe = int(request.form['yoe'])

    df = pd.read_csv(DATA_FILE)
    if name in df['name'].values:
        df.loc[df['name'] == name, ['age', 'salary', 'position', 'yoe']] = [age, salary, position, yoe]
        df.to_csv(DATA_FILE, index=False)
        return redirect(url_for('index'))
    else:
        flash('Name not found!')
        return redirect(url_for('index'))


# Endpoint for live table updates
@app.route("/table")
def table():
    return get_table_html()

# Run app
if __name__ == "__main__":
    app.run(debug=True)
