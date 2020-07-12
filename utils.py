import keras.backend as K

def RMSE(output, target):
    return K.sqrt(K.mean((output - target) ** 2))

def MAE(true, preds):
    return sum(abs(true-preds))/true.shape[0]