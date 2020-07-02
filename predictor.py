import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', help='Run in demo mode', action='store_true')
args = parser.parse_args()

from tensorflow import keras
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Concatenate, SimpleRNN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from inputmismatcherror import InputMismatchError

plt.style.use("ggplot")

demo = args.d

#CONFIG
csv = pd.read_csv("Coinbase_BTCUSD_1h.csv")
size = 1000

#Training parameters
epochs = 300
batchSize = 20
validationSplit = 0.1
inputPeriods = 30
outputPeriods = 10

#Model parameters
activ = "softsign"
droprate = 0.2

#Compile parameters
optimizer = 'nadam'
loss = 'mse'
metrics = ['accuracy']



def makeModel():
    pricesIn = Input(shape=(inputPeriods,1,), name='prices')
    volumesIn = Input(shape=(inputPeriods,1,), name='volumes')

    x = Concatenate()([pricesIn, volumesIn])
    x = GRU(60, activation=activ, recurrent_dropout=droprate)(x)
    x = Dropout(droprate)(x)
    x = Dense(60, activation=activ)(x)
    x = Dense(60, activation=activ)(x)
    x = Dense(60, activation=activ)(x)
    output = Dense(outputPeriods, activation='linear')(x)

    return keras.Model(inputs=[pricesIn, volumesIn], outputs=output, name='model')


# Data Preprocessing
def getData(data, column):
    data = data.set_index("Date")[[column]].head(size)
    data = data.set_index(pd.to_datetime(data.index, yearfirst=True, format='%Y-%m-%d %I-%p'))

    # Normalizing/Scaling the Data
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    return data, scaler

def checkData(data):
    if(data[0].shape != data[1].shape):
        raise InputMismatchError([data[0].shape(), data[1].shape()])

def evaluationData(data, column):
    data = data.set_index("Date")[[column]].tail(5000)
    data = data.set_index(pd.to_datetime(data.index, yearfirst=True, format='%Y-%m-%d %I-%p'))

    # Normalizing/Scaling the Data
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    return data, scaler


prices, priceScaler = getData(csv, 'Close')
volumes, garbage = getData(csv, 'VolumeBTC')

evalPrices, evalScaler = evaluationData(csv, 'Close')
evalVolumes, garbage = evaluationData(csv, 'VolumeBTC')


def graphTrainingResults(results):
    """
    Plots the loss and accuracy for the training and testing data
    """
    history = results.history
    plt.figure(figsize=(12,4))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("loss.png")
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig("accuracy.png")
    plt.show()

def splitData(data):
    """
    Splits the univariate time sequence
    """
    inputData, targetData = [], []

    for i in range(len(data)):
        endIn = i + inputPeriods
        endOut = endIn + outputPeriods

        if endOut > len(data):
            break

        intervalsIn, intervalsOut = data[i:endIn], data[endIn:endOut]

        inputData.append(intervalsIn)
        targetData.append(intervalsOut)

    return np.array(inputData), np.array(targetData)

def graphPredictions(real, predictions):
    plt.figure(figsize=(16,8))
    plt.plot(predictions, label='Predicted')
    plt.plot(real, label='Actual')
    plt.title(f"Predicted vs Actual Closing Prices")
    plt.ylabel("Price")
    plt.xlabel("Interval")
    plt.legend()
    plt.savefig("predictions.png")
    plt.show()

def predictions(trainedModel, predictIn, volumeIn, targets):
    predictIn = predictIn[-1].reshape(1, inputPeriods, 1)
    volumeIn = volumeIn[-1].reshape(1, inputPeriods, 1)
    targets = targets[-1].reshape(1, outputPeriods)

    predicted = trainedModel.predict({'prices':predictIn, 'volumes':volumeIn}).tolist()[0]

    # Getting the actual values from the last available y variable which correspond to its respective X variable
    predictIn = priceScaler.inverse_transform(predictIn.reshape(-1, 1)).tolist()
    actual = predictIn + priceScaler.inverse_transform(targets.reshape(-1, 1)).tolist()

    # Transforming values back to their normal prices
    predicted = predictIn + priceScaler.inverse_transform(np.array(predicted).reshape(-1,1)).tolist()

    graphPredictions(actual, predicted)

def runDemo(price, volume, targets):
    best = keras.models.load_model('best')

    n1 = 180
    n2 = 233
    n3 = 742

    #predictions(best, price[:n1], volume[:n1], targets[:n1])
    #predictions(best, price[:n2], volume[:n2], targets[:n2])
    predictions(best, price[:n3], volume[:n3], targets[:n3])

    exit()

# Splitting the data into appropriate sequences
priceData, targetData = splitData(list(prices.Close))
volumeData, garbage = splitData(list(volumes.VolumeBTC))

evalPriceData, evalTargetData = splitData(list(evalPrices.Close))
evalVolumeData, garbage = splitData(list(evalVolumes.VolumeBTC))

checkData([priceData, volumeData])
checkData([evalPriceData, evalVolumeData])

# Reshaping the X variable from 2D to 3D
priceData = priceData.reshape((priceData.shape[0], priceData.shape[1], 1))
volumeData = volumeData.reshape((volumeData.shape[0], volumeData.shape[1], 1))

evalPriceData = evalPriceData.reshape((evalPriceData.shape[0], evalPriceData.shape[1], 1))
evalVolumeData = evalVolumeData.reshape((evalVolumeData.shape[0], evalVolumeData.shape[1], 1))

if(demo):
    runDemo(priceData, volumeData, targetData)

model = makeModel()
model.summary()


model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

results = model.fit({'prices':priceData, 'volumes':volumeData}, targetData, epochs=epochs, batch_size=batchSize, validation_split=validationSplit, shuffle=False)

graphTrainingResults(results)

evalResults = model.evaluate({'prices':evalPriceData, 'volumes':evalVolumeData}, evalTargetData, return_dict=True)
print(f'Evaluation Results: {evalResults}')

predictions(model, priceData, volumeData, targetData)

model.save('model')
