import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self):
        self.model = None
    

    def _read_dataset(self):
        url = "./resource/iris.csv"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        return pd.read_csv(url, names=names)
    
    def _get_train_test(self, dataset):
        # Split-out validation dataset
        array = dataset.values
        X = array[:,0:4]
        Y = array[:,4]
        validation_size = 0.20
        seed = 7
        return  model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    def train(self):
        data = self._read_dataset()
        X_train, X_validation, Y_train, Y_validation = self._get_train_test(data)
        self.model = LogisticRegression()
        self.model.fit(X_train, Y_train)
        predictions = self.predict(X_validation)
        # print(accuracy_score(Y_validation, predictions))
        return self
    
    def predict(self, x):
        return self.model.predict(x)

if __name__ == "__main__":
    Model().train()