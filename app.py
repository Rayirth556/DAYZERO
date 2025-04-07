from flask import Flask, render_template, request, redirect, send_file, jsonify
import pandas as pd
import os

app = Flask(__name__)
DATA_FILE = 'data.csv'

COLUMNS = [
    "RowNumber", "CustomerId", "Surname", "First Name", "Date of Birth", "Gender", "Marital Status",
    "Number of Dependents", "Occupation", "Income", "Education Level", "Address", "Contact Information",
    "Customer Tenure", "Customer Segment", "Preferred Communication Channel", "Credit Score",
    "Credit History Length", "Outstanding Loans", "Churn Flag", "Churn Reason", "Churn Date", "Balance",
    "NumOfProducts", "NumComplaints"
]

def load_data():
    """Load data from the CSV file, creating an empty DataFrame if missing or invalid."""
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE)
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)

@app.route('/', methods=['GET'])
def index():
    df = load_data()
    return render_template(
        'index.html',
        columns=df.columns.tolist(),
        data=df.to_dict(orient='records'),
        fetched_record=None,
        edit_index=None
    )

@app.route('/submit', methods=['POST'])
def submit():
    df = load_data()
    new_row = {col: request.form.get(col, "") for col in df.columns}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return redirect('/')

@app.route('/fetch', methods=['POST'])
def fetch():
    df = load_data()
    customer_id = request.form.get('fetch_customer_id', '').strip()

    result = df[df['CustomerId'].astype(str).str.strip() == customer_id]

    if not result.empty:
        index = int(result.index[0])
        record = result.iloc[0].to_dict()
        return render_template(
            'index.html',
            columns=df.columns.tolist(),
            data=df.to_dict(orient='records'),
            fetched_record=record,
            edit_index=index
        )

    return render_template(
        'index.html',
        columns=df.columns.tolist(),
        data=df.to_dict(orient='records'),
        fetched_record=None,
        edit_index=None,
        message="No match found."
    )

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            df.to_csv(DATA_FILE, index=False)
        except Exception as e:
            print(f"Upload error: {e}")
    return redirect('/')

@app.route('/download', methods=['GET'])
def download():
    if os.path.exists(DATA_FILE):
        return send_file(DATA_FILE, as_attachment=True)
    return "No data available for download."

@app.route('/update', methods=['POST'])
def update():
    df = load_data()
    edit_index = request.form.get('edit_index')

    if edit_index and edit_index.isdigit():
        idx = int(edit_index)
        if 0 <= idx < len(df):
            updated_row = {col: request.form.get(col, "") for col in df.columns}
            for col in df.columns:
                df.at[idx, col] = updated_row[col]
            df.to_csv(DATA_FILE, index=False)

    return redirect('/')

@app.route('/save_spreadsheet', methods=['POST'])
def save_spreadsheet():
    incoming = request.get_json()
    if not incoming or 'data' not in incoming:
        return jsonify({"error": "Invalid data"}), 400

    data = incoming['data']
    if not isinstance(data, list) or not data:
        return jsonify({"error": "Empty or malformed data"}), 400

    try:
        df = pd.DataFrame(data, columns=COLUMNS)
        df.to_csv(DATA_FILE, index=False)
        return jsonify({"message": "Spreadsheet data saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save data: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
