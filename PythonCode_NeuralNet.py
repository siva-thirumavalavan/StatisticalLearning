import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout

data = pd.read_csv("../Data/Boston.csv")
train = data.drop('crim', axis=1).values
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
test = data['crim'].values
x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.2)


def train_model(x_train, y_train, x_test, y_test, l1, l2):
    """
    Receive train and test sets along with regularisation penalties.
    Returns RMSE for the provided test set.
    """
    model = Sequential()
    model.add(Input(shape=(14,)))
    model.add(Dense(16, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)))
    model.add(Dense(32, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)))
    model.add(Dense(1))
    es_callback = tf.keras.callbacks.EarlyStopping('val_root_mean_squared_error',
                                                   patience=2,
                                                   restore_best_weights=True)
    model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=60, verbose=0,
                        callbacks=[es_callback])
    return round(history.history['val_root_mean_squared_error'][-1], 2)


def cv_nn(l1, l2):
    """
    Performs K-Fold cross validation using l1 and l2 penalties.
    Uses Neural Network training function for each fold.
    Returns mean RMSE of all the folds.
    """
    kf = KFold(n_splits=3)
    rmse_list = []
    for train_ind, test_ind in kf.split(train):
        x_train = train[train_ind, :]
        x_test = train[test_ind, :]
        y_train = test[train_ind]
        y_test = test[test_ind]
        rmse_list.append(train_model(x_train, y_train, x_test, y_test, l1, l2))
    return np.mean(rmse_list)


def converter(al, lam):
    """
    Converts alpha and lambda parameters (as specified in R Code) into
    tensorflow format.
    """
    return lam*al, lam*(1-al)


# Create lists to store grid search inputs and results
alphas, lambdas, rmses = [], [], []
al_grid = [x/10 for x in range(11)]
lam_grid = [np.exp(i) for i in np.arange(-5.5, 5, 0.3)]

# Perform looped grid search
for al in al_grid:
    for lam in lam_grid:
        l1, l2 = converter(al, lam)
        alphas.append(al)
        lambdas.append(lam)
        rmses.append(cv_nn(l1, l2))
    print(f"Completed Alpha {al}")

# Update the results to a dataframe
metrics = pd.DataFrame({'alpha': alphas, 'lambda': lambdas, 'rmse': rmses})
plt.figure(figsize=(15, 12))
al_list = list(metrics['alpha'].unique())

# Plot the results for each alpha value (Proportion of each regularisation)
for al in al_list:
    mat = metrics[metrics['alpha'] == al].reset_index(drop=True)
    plt.plot(mat['lambda'], mat['rmse'])
plt.legend(al_list)
