from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import Model

app = Flask(__name__)
api = Api(app)

model = Model().train()


# argument parsing
parser = reqparse.RequestParser()
['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
parser.add_argument('sepal_length')
parser.add_argument('sepal_width')
parser.add_argument('petal_length')
parser.add_argument('petal_width')


class PredictIris(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        sl = float(args['sepal_length'])
        sw = float(args['sepal_width'])
        pl = float(args['petal_length'])
        pw = float(args['petal_width'])
        pre_vec = [sl, sw, pl, pw]
        print(pre_vec)

        vec = np.array(pre_vec).reshape(1,-1)
        prediction = model.predict(vec)
        print("prediction:", prediction)

        pred_text = str(prediction)

        # create JSON object
        output = {'prediction': pred_text}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictIris, '/')


if __name__ == '__main__':
    app.run(debug=True)