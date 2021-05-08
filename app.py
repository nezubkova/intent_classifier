from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import joblib
import nltk
import string
import numpy as np

app = Flask(__name__)
api = Api(app)

classifier = joblib.load('classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')
st = nltk.stem.SnowballStemmer('english')

def stem_text(text):
    return ' '.join([st.stem(word) for word in nltk.word_tokenize(text.lower()) if word not in list(string.punctuation)])


class Classify(Resource):
    
    def post():

        parser = reqparse.RequestParser()
        parser.add_argument('text')

        args = parser.parse_args()
        text = args['text']

        tf_idf = vectorizer.transform([stem_text(text)])

        prediction = model.predict(tf_idf)

        output = {'prediction': cats[int(prediction)]}
        
        return output


api.add_resource(Classify, '/classify')

if __name__ == '__main__':
    app.run(debug=True, port='1080')
