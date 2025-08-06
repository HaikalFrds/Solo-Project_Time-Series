# Data & Utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
#import pmdarima as pm
from numpy import log

# Import keras lstm model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, BatchNormalization
from keras.callbacks import EarlyStopping

# Import ARIMA model
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scipy import stats
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import MinMaxScaler

# Evaluation Metrics
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score