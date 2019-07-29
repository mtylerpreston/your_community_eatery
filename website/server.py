import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from src.functions import get_recs
import pickle


# Create the app object that will route our calls
app = Flask(__name__)

# Load similarities, bus_review_df, and item map to use for recommending
with open('models/item_map.pkl', 'rb') as file:
    item_map = pickle.load(file)

with open('models/KNNBaseline_similarities.pkl', 'rb') as file:
    similarities = pickle.load(file)

df = pd.read_json('data/processed_df.json', orient='records')


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    req = request.get_json()
    selections = [req['pick1'], req['pick2'], req['pick3'], req['pick4']]
    recs = get_recs(similarities, df, item_map, 3, selections)
    recs = list(recs.name)
    recs = 'We are pleased to recommend the following:\n' + recs[0] + '\n' + recs[1] + '\n' + recs[2]
    return jsonify({'recommendations': recs})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3333, debug=True)
