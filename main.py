from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/predict_button', methods=['POST'])
def predict_button():
    # Access form data using request.form
    Sex = request.form['Sex']
    AgeCategory = request.form["AgeCategory"]
    Glucose = request.form["Glucose"]
    Pregnancy = request.form["Pregnancy"]
    Bp = request.form["Bp"]
    Skin = request.form["Skin"]
    Insulin = request.form["Insulin"]
    dpf = request.form["dpf"]
    blood = request.form["blood"]
    BMI = request.form["BMI"]

    # Perform any necessary actions before redirection

    Sex = 1 if Sex == "Male" else 0
    AgeCategory = int(AgeCategory)
    Glucose = float(Glucose)
    Pregnancy = float(Pregnancy)
    Bp = float(Bp)
    Skin = float(Skin)
    Insulin = float(Insulin)
    dpf = float(dpf)
    BMI = float(BMI)

    # Create a list of transformed features
    transformed_features = [Pregnancy, Glucose, Bp, Skin, Insulin, BMI, dpf, AgeCategory]

    # Scale the features using MinMaxScaler
    transformed_features_scaled = scaler.fit_transform([transformed_features])

    # Make prediction
    prediction = loaded_data.predict(transformed_features_scaled)

    if prediction == 1:
        result = "Diabetes predicted"
        # Include specific messages based on blood type conditions
        if (blood == "positive_o" or blood == "negative_o") and result == 1:
            message = "A high-protein diet heavy on lean meat, poultry, fish, and vegetables, and light on grains, beans, and dairy. Doctor recommends various supplements to help with tummy troubles and other issues he says people with type O tend to have."
        elif (blood == "positive_a" or blood == "negative_a") and result == 1:
            message = "A meat-free diet based on fruits and vegetables, beans and legumes, and whole grains -- ideally, organic and fresh, because D'Adamo says people with type A blood have a sensitive immune system."
        elif (blood == "positive_b" or blood == "negative_b") and result == 1:
            message = "Avoid corn, wheat, buckwheat, lentils, tomatoes, peanuts, and sesame seeds. Chicken is also problematic, D'Adamo says. He encourages eating green vegetables, eggs, certain meats, and low-fat dairy."
        else:
            message = "Foods to focus on include tofu, seafood, dairy, and green vegetables. He says people with type AB blood tend to have low stomach acid. Avoid caffeine, alcohol, and smoked or cured meats."
    else:
        result = "No Diabetes predicted"
        message = "No specific dietary recommendations for non-diabetic predictions."

    return render_template('predict_page.html', prediction=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
