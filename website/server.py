from flask import Flask, render_template, request, jsonify, Response
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from pipeline import preprocessing
import eventAPI as a

# Create the app object that will route our calls
app = Flask(__name__)
# Add a single endpoint that we can use for testing


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'


client = a.EventAPIClient()

# model = pickle.load(open('../models/website/lr_model.pkl', 'rb'))


@app.route('/recommend', methods=['POST'])
def recommend():
    req = request.get_json()

    p1 = req['pick1']
    p2 = req['pick2']
    p3 = req['pick3']
    p4 = req['pick4']

    recommendations = "None"

    return jsonify({'recommendations': str(recommendations)})


@app.route('/fraud', methods=['GET'])
def fraud():
    return render_template('fraud_ClusteredMarker.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3333, debug=True)
