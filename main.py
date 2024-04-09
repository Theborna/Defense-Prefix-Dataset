from dataset import dataset
obj = type('obj', (object,), {'dataset' : 'food'})
preprocessor = None
data = dataset(obj, preprocessor)