from flask import Flask, render_template, request, redirect, send_file, url_for
import pandas as pd
import os

app = Flask(__name__)
DATA_FILE = 'data.csv'


def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=[
            "RowNumber", "CustomerId", "Surname", "First Name", "Date of Birth", "Gender", "Marital Status",
            "Number of Dependents", "Occupation", "Income", "Education Level", "Address", "Contact Information",
            "Customer Tenure", "Customer Segment", "Preferred Communication Channel", "Credit Score",
            "Credit History Length", "Outstanding Loans", "Churn Flag", "Churn Reason", "Churn Date", "Balance",
            "NumOfProducts", "NumComplaints"
        ])


@app.route('/', methods=['GET'])

def index():
    df = load_data()
    return render_template(
        'index.html',
        table=df.to_html(index=False),
        columns=df.columns
    )


@app.route('/submit', methods=['POST'])
def submit():
    df = load_data()

    # Get form values
    form_data = {col: request.form.get(col) for col in df.columns}

    # Append the new row
    df = pd.concat([df, pd.DataFrame([form_data])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    return redirect('/')


@app.route('/fetch', methods=['POST'])
def fetch():
    df = load_data()

    first_name = request.form.get('fetch_first_name', '').strip()
    surname = request.form.get('fetch_surname', '').strip()

    result = df[(df['First Name'].str.strip().str.lower() == first_name.lower()) &
                (df['Surname'].str.strip().str.lower() == surname.lower())]

    if not result.empty:
        return render_template('index.html', table=result.to_html(index=False), fetch_result=True)
    else:
        return render_template('index.html', table=df.to_html(index=False), fetch_result=False, message="No match found.")


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        df.to_csv(DATA_FILE, index=False)
    return redirect('/')


@app.route('/download', methods=['GET'])
def download():
    return send_file(DATA_FILE, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
