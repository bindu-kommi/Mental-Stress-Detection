from flask import Flask, render_template, redirect, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import mysql.connector, pickle
from keras.models import load_model
from imblearn.over_sampling import SMOTE
import os
import joblib

app = Flask(__name__)

mydb = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    database='mentalstress'
)

mycur = mydb.cursor()

global models, model_client1, model_client2, model_client3


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        phonenumber= request.form['phonenumber']
        age  = request.form['age']
        if password == confirmpassword:
            sql = 'SELECT * FROM users WHERE email = %s'
            val = (email,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                sql = 'INSERT INTO users (name, email, password,`phone number`,age) VALUES (%s, %s, %s, %s,%s)'
                val = (name, email, password, phonenumber,age)
                mycur.execute(sql, val)
                mydb.commit()
                
                msg = 'User registered successfully!'
                return render_template('registration.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password = data[2]
            if password == stored_password:
               msg = 'user logged successfully'
               return redirect("/viewdata")
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    return render_template('login.html')


# Route to view the data
@app.route('/viewdata')
def viewdata():
    # Load the dataset
    dataset_path = 'mental stress data.xlsx'  # Make sure this path is correct to the uploaded file
    df = pd.read_excel(dataset_path)
    df = df.head(1000)

    # Convert the dataframe to HTML table
    data_table = df.to_html(classes='table table-striped table-bordered', index=False)

    # Render the HTML page with the table
    return render_template('viewdata.html', table=data_table)


# Load models paths
model_paths = {
    "CatBoost": r"saved_models\catboost_model.pkl",
    "Decision Tree": r"saved_models\decision_tree_model.pkl",
    "DNN": r"saved_models\dnn_model.h5",
    "FNN": r"saved_models\fnn_model.h5",
    "Logistic Regression": r"saved_models\logistic_regression_model.pkl",
    "LSTM": r"saved_models\lstm_model.h5",
    "Random Forest": r"saved_models\random_forest_model.pkl",
    "XGBoost": r"saved_models\xgboost_model.pkl"
}

def load_model_from_path(model_name, path):
    if model_name in ["DNN", "FNN", "LSTM"]:
        if os.path.exists(path):
            return load_model(path)  # Load Keras models
        else:
            raise FileNotFoundError(f"Keras model file not found: {path}")
    else:
        if os.path.exists(path):
            return joblib.load(path)  # Load joblib models
        else:
            raise FileNotFoundError(f"Model file not found: {path}")

# Mock function to get test data
def get_test_data():
    df = pd.read_excel('new_features_data.xlsx')
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Check for imbalance and apply SMOTE if necessary
    if y.value_counts(normalize=True).min() < 0.2:  # Assuming an imbalance threshold of 20%
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_test, y_test

@app.route('/algo')
def algo():
    X_test, y_test = get_test_data()
    results = {}

    for model_name, path in model_paths.items():
        try:
            model = load_model_from_path(model_name, path)
            if model_name in ["DNN", "FNN", "LSTM"]:
                # Reshape input for LSTM
                if model_name == "LSTM":
                    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))  # Reshaping for LSTM
                    y_pred = model.predict(X_test_reshaped)
                    y_pred = np.argmax(y_pred, axis=1)  # Get the class with the highest probability
                else:
                    y_pred = model.predict(X_test)
                    y_pred = np.argmax(y_pred, axis=1)  # Get the class with the highest probability
            else:
                y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[model_name] = {"accuracy": accuracy, "report": report}
        except FileNotFoundError as fnf_error:
            results[model_name] = {"accuracy": None, "report": {"error": str(fnf_error)}}
        except Exception as e:
            results[model_name] = {"accuracy": None, "report": {"error": str(e)}}

    return render_template('algo.html', results=results)


# Load the model and data once during the application startup
df = pd.read_excel('new_features_data.xlsx')
X = df.drop('label', axis=1)
y = df['label']

# Check for imbalance and apply SMOTE if necessary
if y.value_counts(normalize=True).min() < 0.2:  # Assuming an imbalance threshold of 20%
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Get input data from the form
        input_data = [float(request.form['X']), float(request.form['Y']), float(request.form['Z']),
                      float(request.form['EDA']), float(request.form['HR']), float(request.form['TEMP'])]
        
        # Make prediction
        prediction = random_forest.predict([input_data])[0]

        # Suggestions based on the prediction
        suggestions = {
            0: "No stress detected. Suggestions: Maintain a healthy diet, regular exercise, and practice mindfulness.",
            1: "Severe stress detected. Suggestions: Try meditation, yoga, taking short breaks, and staying organized.",
        }

        # Prepare the result to send to the frontend
        result = suggestions[prediction]
        return render_template('prediction.html', prediction=prediction, result=result)

    return render_template('prediction.html', prediction=None, result=None)


if __name__ == '__main__':
    app.run(debug=True)
