from flask import Flask, request, render_template 
import pickle 
import numpy as np 
 
app = Flask(__name__) 
 
# Load the trained model 
model = pickle.load(open('E:\AI\liner regration\linear_model.pkl', 'rb')) 
 
@app.route('/') 
def home(): 
    return render_template('index.html') 
 
@app.route('/predict', methods=['POST']) 
def predict(): 
    try: 
        x = float(request.form['x'])  # Change as per your model inputs 
        
 
        features = np.array([[x]]) 
        prediction = model.predict(features) 
 
        return render_template('index.html', result=f'Predicted Value: {prediction[0]:.2f}') 
    except Exception as e: 
        return render_template('index.html', result=f'Error: {str(e)}') 
 
if __name__ == '__main__': 
    app.run(debug=True)