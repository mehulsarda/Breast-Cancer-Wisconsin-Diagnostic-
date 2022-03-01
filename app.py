from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
#        'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
#        'smoothness_se', 'compactness_se', 'concave_points_se', 'symmetry_se',
#        'symmetry_worst', 'fractal_dimension_worst']

model = load_model(
    './models/Breast_Cancer_Wisconsin_(Diagnostic)_2.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/predicted', methods=['POST'])
def predicted():

    radius_mean = float(request.form['radius_mean'])
    texture_mean = float(request.form['texture_mean'])
    smoothness_mean = float(request.form['smoothness_mean'])
    compactness_mean = float(request.form['compactness_mean'])
    symmetry_mean = float(request.form['symmetry_mean'])
    fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
    radius_se = float(request.form['radius_se'])
    texture_se = float(request.form['texture_se'])
    smoothness_se = float(request.form['smoothness_se'])
    compactness_se = float(request.form['compactness_se'])
    concave_points_se = float(request.form['concave_points_se'])
    symmetry_se = float(request.form['symmetry_se'])
    symmetry_worst = float(request.form['symmetry_worst'])
    fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

    values = [[radius_mean, texture_mean, smoothness_mean, compactness_mean, symmetry_mean, fractal_dimension_mean, radius_se,
              texture_se, smoothness_se, compactness_se, concave_points_se, symmetry_se, symmetry_worst, fractal_dimension_worst]]

    print(values)

    prediction = model.predict(values)

    print(prediction[0][0])

    if prediction[0][0] >= 0.5 and prediction[0][0] <= 1:
        ans = 'fatal'
    elif prediction[0][0] >= 0 and prediction[0][0] < 0.5:
        ans = 'non fatal'
    else:
        ans = 'error'

    return render_template('predicted.html', data=ans)


if __name__ == '__main__':
    app.run()
