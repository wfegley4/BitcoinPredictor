# BitcoinPredictor
A Tensorflow powered Recurrent Neural Network to predict the future price of Bitcoin.

This project is not to be used as investment advice.
The purpose of this project is not to achieve incredibly accurate price predictions, but rather to be used as a tool to teach myself about neural networks.

The current model uses 30 consecutive intervals of close price and volume data to predict the close price of the next 10 consecutive intervals.
The best performing model is frozen and its data is saved in the 'best' folder, and can be loaded by Tensorflow for use at any time.

Historical Coinbase hourly data is from https://www.cryptodatadownload.com/.
Thanks to this guide for helping me when I first started this project: https://towardsdatascience.com/predicting-bitcoin-prices-with-deep-learning-438bc3cf9a6f.
