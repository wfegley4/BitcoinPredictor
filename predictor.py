import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', help='Run in demo mode', action='store_true')
args = parser.parse_args()

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, SimpleRNN, GRU, Reshape
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
epochs = 25
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

def evalData(data, column):
    data = data.set_index("Date")[[column]].tail(5000)
    data = data.set_index(pd.to_datetime(data.index, yearfirst=True, format='%Y-%m-%d %I-%p'))

    # Normalizing/Scaling the Data
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    return data, scaler


prices, priceScaler = getData(csv, 'Close')
volumes, garbage = getData(csv, 'VolumeBTC')

evalPrices, evalScaler = evalData(csv, 'Close')
evalVolumes, garbage = evalData(csv, 'VolumeBTC')


def visualize_training_results(results):
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

def split_sequence(seq, n_steps_in, n_steps_out):
    """
    Splits the univariate time sequence
    """
    X, y = [], []

    for i in range(len(seq)):
        end = i + n_steps_in
        out_end = end + n_steps_out

        if out_end > len(seq):
            break

        seq_x, seq_y = seq[i:end], seq[end:out_end]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

def graphPredictions(real, predictions, note):
    plt.figure(figsize=(16,8))
    plt.plot(predictions, label='Predicted')
    plt.plot(real, label='Actual')
    plt.title(f"Predicted vs Actual Closing Prices")
    plt.ylabel("Price")
    plt.xlabel(note)
    plt.legend()
    plt.savefig("BTC_validation.png")
    plt.show()

def predictions(trainedModel, predictIn, volumeIn, targets):
    predictIn = predictIn[-1].reshape(1, inputPeriods, 1)
    volumeIn = volumeIn[-1].reshape(1, inputPeriods, 1)
    targets = targets[-1].reshape(1, outputPeriods)

    predicted = trainedModel.predict({'prices':predictIn, 'volumes':volumeIn}).tolist()[0]
    evalResult = trainedModel.evaluate({'prices':predictIn, 'volumes':volumeIn}, targets, return_dict=True)

    # Getting the actual values from the last available y variable which correspond to its respective X variable
    predictIn = priceScaler.inverse_transform(predictIn.reshape(-1, 1)).tolist()
    actual = predictIn + priceScaler.inverse_transform(targets.reshape(-1, 1)).tolist()

    # Transforming values back to their normal prices
    predicted = predictIn + priceScaler.inverse_transform(np.array(predicted).reshape(-1,1)).tolist()

    evalNote = f'Loss: {round(evalResult.get("loss"), 6)} Accuracy: {round(evalResult.get("accuracy"), 6)}'

    graphPredictions(actual, predicted, evalNote)

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
priceData, targetData = split_sequence(list(prices.Close), inputPeriods, outputPeriods)
volumeData, garbage = split_sequence(list(volumes.VolumeBTC), inputPeriods, outputPeriods)

evalPriceData, evalTargetData = split_sequence(list(evalPrices.Close), inputPeriods, outputPeriods)
evalVolumeData, garbage = split_sequence(list(evalVolumes.VolumeBTC), inputPeriods, outputPeriods)

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

visualize_training_results(results)

evalResults = model.evaluate({'prices':evalPriceData, 'volumes':evalVolumeData}, evalTargetData, return_dict=True)
print(f'Evaluation Results: {evalResults}')

predictions(model, priceData, volumeData)

model.save('model')
