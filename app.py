from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # get form data
        tsh = request.form.get('tsh')
        t3 = request.form.get('t3')
        tt4 = request.form.get('tt4')
        t4u = request.form.get('t4u')
        fti = request.form.get('fti')

        # call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(tsh, t3, tt4, t4u, fti)
            # pass prediction to template
            return render_template('predict.html', prediction=prediction)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass


def preprocessDataAndPredict(tsh, t3, tt4, t4u, fti):
    # keep all inputs in array
    test_data = [tsh, t3, tt4, t4u, fti]
    print(test_data)

    # convert value data into numpy array
    test_data = np.array(test_data)

    # reshape array
    test_data = test_data.reshape(1, -1)
    print(test_data)

    # open file
    file = open("randomforest_model.pkl", "rb")

    # load trained model
    trained_model = joblib.load(file)

    # predict
    prediction = trained_model.predict(test_data)

    return prediction

    pass


if __name__ == '__main__':
    app.run(debug=True)