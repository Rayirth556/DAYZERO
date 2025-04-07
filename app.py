import joblib
from datetime import datetime
from flask import Flask, render_template, request, redirect, send_file, url_for
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import LabelEncoder

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
        table=df.head(100).to_html(index=False),
        columns=df.columns
    )


@app.route('/submit', methods=['POST'])
def submit():
    df = load_data()
    form_data = {col: request.form.get(col) for col in df.columns}
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

@app.route('/analysis')
def analysis():
    # Load model and preprocessing tools
    model = joblib.load('xgboost_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('target_encoder.pkl')

    # Load data
    data = load_data()
    if data.empty:
        return "No data available for analysis."

    # === Preprocessing ===
    data['Date of Birth'] = pd.to_datetime(data['Date of Birth'], errors='coerce')
    data['Age'] = ((datetime(2025, 4, 7) - data['Date of Birth']).dt.days / 365.25).astype('str')

    drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'First Name', 'Address', 'Contact Information',
                 'Date of Birth', 'Churn Date', 'Churn Reason']
    cat_cols = ['Gender', 'Marital Status', 'Customer Segment', 'Preferred Communication Channel']

    X = data.drop(columns=['Churn Flag'], errors='ignore')
    y = data['Churn Flag'] if 'Churn Flag' in data else pd.Series([0]*len(data))

    X_original = X[['Gender', 'Marital Status', 'Occupation']].copy()

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X, _ = X.align(pd.DataFrame(columns=model.feature_names_in_), join='right', axis=1, fill_value=0)

    for col in ['Occupation', 'Education Level']:
        if col in X.columns:
            encoder = LabelEncoder()
            encoder.fit(X[col])         # Pass the column (Series) it
            X[col] = encoder.transform(X[col])  # Then transform


    X.drop(columns=[col for col in drop_cols if col in X.columns], inplace=True)

    numerical_cols = ['Income', 'Credit Score', 'Credit History Length', 'Outstanding Loans', 'Balance',
                      'NumOfProducts', 'NumComplaints', 'Customer Tenure', 'Age']
    X[numerical_cols] = scaler.transform(X[numerical_cols])

    # Add engineered features
    X['Debt_to_Income'] = X['Outstanding Loans'] / X['Income'].replace(0, 1e-6)
    X['Complaint_Rate'] = X['NumComplaints'] / X['Customer Tenure'].replace(0, 1e-6)
    X['Engagement_Score'] = X['NumOfProducts'] * X['Balance']

    # === Prediction ===
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # === Visualization ===
    import matplotlib.gridspec as gridspec

    plot_data = pd.concat([
        pd.DataFrame(X[numerical_cols]),
        pd.Series(y, name='Churn Flag')
    ], axis=1).melt(id_vars='Churn Flag', var_name='Feature', value_name='Value')

    pred_data = pd.DataFrame({
        'Predicted Probability': y_proba,
        'Churn Flag': y
    })

    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).nlargest(10, 'Importance')

    cat_pred_data = pd.concat([X_original.reset_index(drop=True), pd.Series(y_pred, name='Predicted Churn')], axis=1)
    occupation_churn = cat_pred_data.groupby('Occupation')['Predicted Churn'].mean().reset_index()
    top_churn_occupations = occupation_churn.sort_values(by='Predicted Churn', ascending=False).head(10)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    ax1 = fig.add_subplot(spec[0, 0])
    sns.kdeplot(data=pred_data, x='Predicted Probability', hue='Churn Flag', fill=True, common_norm=False, palette='Set2', ax=ax1)
    ax1.set_title('Predicted Probability Distribution by Churn')
    ax1.axvline(0.5, color='gray', linestyle='--')

    ax2 = fig.add_subplot(spec[0, 1])
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax2)
    ax2.set_title('Top 10 Feature Importances')

    ax3 = fig.add_subplot(spec[1, 0])
    sns.boxplot(x='Feature', y='Value', hue='Churn Flag', data=plot_data, palette='Set3', ax=ax3)
    ax3.set_title('Numerical Feature Distributions by Churn')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.set_ylim(-3, 3)

    ax4 = fig.add_subplot(spec[1, 1])
    sns.barplot(x='Occupation', y='Predicted Churn', data=top_churn_occupations, palette='Set1', ax=ax4)
    ax4.set_title('Top 10 Occupations by Churn Rate')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_ylim(0, 1)

    plt.suptitle('Churn Prediction Dashboard', fontsize=18, y=1.02)

    # Convert plot to image and embed in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template("analysis.html", plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
