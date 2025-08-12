from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model when the app starts
try:
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except:
    print("Error loading model!")

# Load scaler if you have it
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully!")
except:
    print("Scaler not found - will proceed without scaling")
    scaler = None

@app.route('/')
def index():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        step = float(request.form['step'])
        amount = float(request.form['amount'])
        oldbalance_org = float(request.form['oldbalance_org'])
        newbalance_orig = float(request.form['newbalance_orig'])
        oldbalance_dest = float(request.form['oldbalance_dest'])
        newbalance_dest = float(request.form['newbalance_dest'])
        
        # Get transaction type
        transaction_type = request.form['type']
        
        # Create feature vector (adjust based on your model's features)
        # Assuming you have one-hot encoded transaction types
        features = [step, amount, oldbalance_org, newbalance_orig, 
                   oldbalance_dest, newbalance_dest]
        
        # Add one-hot encoded transaction type features
        # Adjust these based on your actual encoded features
        type_features = [0, 0, 0, 0, 0]  # Initialize for 5 transaction types
        if transaction_type == 'CASH_IN':
            type_features[0] = 1
        elif transaction_type == 'CASH_OUT':
            type_features[1] = 1
        elif transaction_type == 'DEBIT':
            type_features[2] = 1
        elif transaction_type == 'PAYMENT':
            type_features[3] = 1
        elif transaction_type == 'TRANSFER':
            type_features[4] = 1
        
        # Combine all features
        all_features = features + type_features
        
        # Convert to numpy array and reshape
        feature_array = np.array(all_features).reshape(1, -1)
        
        # Scale features if scaler is available
        if scaler:
            feature_array = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        prediction_proba = model.predict_proba(feature_array)[0]
        
        # Get fraud probability
        fraud_probability = prediction_proba[1] * 100
        
        # Determine result
        if prediction == 1:
            result = "FRAUDULENT"
            alert_class = "alert-danger"
        else:
            result = "LEGITIMATE"
            alert_class = "alert-success"
        
        return render_template('result.html', 
                             prediction=result,
                             probability=round(fraud_probability, 2),
                             alert_class=alert_class,
                             transaction_data={
                                 'step': step,
                                 'amount': amount,
                                 'type': transaction_type,
                                 'oldbalance_org': oldbalance_org,
                                 'newbalance_orig': newbalance_orig,
                                 'oldbalance_dest': oldbalance_dest,
                                 'newbalance_dest': newbalance_dest
                             })
    
    except Exception as e:
        return render_template('result.html', 
                             error=f"Error making prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
