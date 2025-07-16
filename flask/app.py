from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         Gender = float(request.form['Gender'])
#         Hemoglobin = float(request.form['Hemoglobin'])
#         MCH = float(request.form['MCH'])
#         MCHC = float(request.form['MCHC'])
#         MCV = float(request.form['MCV'])

#         data = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])
#         df = pd.DataFrame(data, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
#         prediction = model.predict(df)

#         if prediction[0] == 0:
#             result = "You don't have anemic diseases"
#         else:
#             result = "You have anemic diseases"

#         return render_template('predict.html', prediction_text="Hence, based on calculation: " + result)
    
#     except Exception as e:
#         return f"❌ Internal error: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Gender = float(request.form['Gender'])
        Hemoglobin = float(request.form['Hemoglobin'])
        MCH = float(request.form['MCH'])
        MCHC = float(request.form['MCHC'])
        MCV = float(request.form['MCV'])

        data = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])
        df = pd.DataFrame(data, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
        prediction = model.predict(df)

        if prediction[0] == 0:
            result = "You don't have anemic diseases"
        else:
            result = "You have anemic diseases"

        return render_template(
            'predict.html',
            prediction_text="Hence, based on calculation: " + result,
            hemoglobin=Hemoglobin,
            mch=MCH,
            mchc=MCHC,
            mcv=MCV
        )
    
    except Exception as e:
        return f"❌ Internal error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
