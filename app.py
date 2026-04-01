from flask import Flask, render_template, request
import numpy as np
import joblib
import xgboost as xgb

app = Flask(__name__) # it allows python to run on a web server so users can interack with them through a web browser.

# Load trained model
model=xgb.XGBClassifier()
model.load_model("wine_model.json")# This loads the trained model that you saved earlier using joblib.dump()
model.n_classes_ = 6 

original_min = joblib.load("original_min.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        
        prediction = model.predict(final_features)[0]
        prediction=int(prediction+original_min)

        return render_template("index.html",
                               prediction_text=f"Predicted Wine Quality: {prediction}")
    except Exception as e:
        print(f"Error: {e}")  #  shows real error in terminal
        return render_template("index.html",
                               prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)