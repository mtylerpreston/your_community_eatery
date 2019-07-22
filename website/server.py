from flask import Flask,render_template, request,jsonify,Response
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from pipeline import preprocessing
import eventAPI as a


#Create the app object that will route our calls
app = Flask(__name__)
# Add a single endpoint that we can use for testing
@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/hello', methods = ['GET'])
def hello():
    return 'Hello, World!'

client = a.EventAPIClient()
model = pickle.load(open('website/lr_model.pkl','rb'))


@app.route('/score',  methods = ['POST'])
def score():
    response = client.get_data()
    processed_response = preprocessing(response)
    predictions = model.predict_proba(processed_response)
    lst = []
    for e,i in enumerate(predictions):
	    if i[1] >= 0.99:
	        lst.append(response.name[e]+ ': High risk')
	    elif i[1] >= 0.5:
	        lst.append(response.name[e]+ ': Medium risk')
	    else:
	        lst.append(response.name[e]+ ': Low risk')

    return jsonify({'prediction': str(lst)})

@app.route('/fraud',methods = ['GET'])
def fraud():
    return render_template('fraud_ClusteredMarker.html')

if __name__ == '__main__':
	app.run(host='0.0.0.0',port = 3333,debug = True)
