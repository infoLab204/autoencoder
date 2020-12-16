
# Load library
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras import optimizers
from keras import initializers

import datetime

def SBAE(data_type, trainX, trainY, testX, testY, z_code,epoches, batch_size) :
    # Load MNIST train and test data

    X_train = np.loadtxt(trainX, delimiter=',', dtype=None)
    Y_train = np.loadtxt(trainY, delimiter=',', dtype=None)

    X_test = np.loadtxt(testX, delimiter=',', dtype=None)
    Y_test = np.loadtxt(testY, delimiter=',', dtype=None)

    # z_list : define experiment code(Z) size

    #z_list = [4, 8, 12, 16, 20]
    z_list = [z_code]

    autoencoder = [[] for i in range(len(z_list))]

    # E : epoch, BS = batch size

    # E = 200
    # BS = 100

    E = epoches
    BS = batch_size

    # Train model and save data(code(Z), output and total loss data)

    model_index = 0

    t01 = datetime.datetime.now()

    total_summary_loss_data = ['model_type', 'z_size', 'train_loss', 'test_loss']

    # Define first pre-training(784 -> 400) model

    INPUT_SIZE = 784
    HIDDEN_SIZE = 400

    if data_type=="digit" : 
        w_initializer = initializers.glorot_uniform(seed=None)
        b_initializer = initializers.glorot_uniform(seed=None)

        dense1 = Input(shape=(INPUT_SIZE,))
        dense2 = Dense(HIDDEN_SIZE, activation='tanh', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense1)
        dense3 = Dense(INPUT_SIZE, activation='relu', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense2)

        autoencoder[model_index] = Model(dense1, dense3)

        adagrad = optimizers.Adagrad(lr=0.01)
        autoencoder[model_index].compile(loss='mean_squared_error', optimizer=adagrad)

        autoencoder[model_index].fit(X_train, X_train, epochs=E, batch_size=BS, verbose=0)
        
    else : 
        w_initializer = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        b_initializer = initializers.glorot_normal(seed=None)

        dense1 = Input(shape=(INPUT_SIZE,))
        dense2 = Dense(HIDDEN_SIZE, activation='tanh', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense1)
        dense3 = Dense(INPUT_SIZE, activation='relu', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense2)

        autoencoder[model_index] = Model(dense1, dense3)

        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        autoencoder[model_index].compile(loss='mean_squared_error', optimizer=sgd)

    autoencoder[model_index].fit(X_train, X_train, epochs=E, batch_size=BS, verbose=0)
    
    pre_train_w1 = autoencoder[model_index].layers[1].get_weights()
    pre_train_w2 = autoencoder[model_index].layers[2].get_weights()

    get_pre_train_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[1].output])
    X_train2 = get_pre_train_z([X_train])[0]

    for z_size in z_list : 

        t11 = datetime.datetime.now()

        # Define second pre-training(400 -> z_size) models

        INPUT_SIZE = 400
        HIDDEN_SIZE = z_size

        dense1 = Input(shape=(INPUT_SIZE,))
        dense2 = Dense(HIDDEN_SIZE, activation='tanh', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense1)
        dense3 = Dense(INPUT_SIZE, activation='relu', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense2)

        autoencoder[model_index] = Model(dense1, dense3)
    
        if data_type =="digit" : 
            adagrad = optimizers.Adagrad(lr=0.01)
            autoencoder[model_index].compile(loss='mean_squared_error', optimizer=adagrad)

            autoencoder[model_index].fit(X_train2, X_train2, epochs=E, batch_size=BS, verbose=0)
        else : 
            sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            autoencoder[model_index].compile(loss='mean_squared_error', optimizer=sgd)

            autoencoder[model_index].fit(X_train2, X_train2, epochs=E, batch_size=BS, verbose=0)

        pre_train_w3 = autoencoder[model_index].layers[1].get_weights()
        pre_train_w4 = autoencoder[model_index].layers[2].get_weights()

        # Define stacked models 

        INPUT_SIZE = 784
        HIDDEN_SIZE1 = 400
        HIDDEN_SIZE2 = z_size

        dense1 = Input(shape=(INPUT_SIZE,))
        dense2 = Dense(HIDDEN_SIZE1, activation='tanh')(dense1)
        dense3 = Dense(HIDDEN_SIZE2, activation='tanh')(dense2)
        dense4 = Dense(HIDDEN_SIZE1, activation='relu')(dense3)
        dense5 = Dense(INPUT_SIZE, activation='relu')(dense4)

        autoencoder[model_index] = Model(dense1, dense5)

        autoencoder[model_index].layers[1].set_weights(pre_train_w1)
        autoencoder[model_index].layers[2].set_weights(pre_train_w3)
        autoencoder[model_index].layers[3].set_weights(pre_train_w4)
        autoencoder[model_index].layers[4].set_weights(pre_train_w2)

        if data_type=="digit" : 
            adagrad = optimizers.Adagrad(lr=0.01)
            autoencoder[model_index].compile(loss='mean_squared_error', optimizer=adagrad)

            autoencoder[model_index].fit(X_train, X_train, epochs=E, batch_size=BS, verbose=0)
        else : 
            sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            autoencoder[model_index].compile(loss='mean_squared_error', optimizer=sgd)

            autoencoder[model_index].fit(X_train, X_train, epochs=E, batch_size=BS, verbose=0)

        # Get output and calculate loss

        get_output = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[4].output])
        train_output = get_output([X_train])[0]
        test_output = get_output([X_test])[0]

        train_loss = np.sum((X_train - train_output)**2) / (X_train.shape[0] * X_train.shape[1])
        test_loss = np.sum((X_test - test_output)**2) / (X_test.shape[0] * X_test.shape[1])

        summary_loss_data = ['SBAE', z_size, train_loss, test_loss]

        total_summary_loss_data = np.vstack((total_summary_loss_data, summary_loss_data))

        np.savetxt("MNIST_SBAE_total_loss_data.csv", total_summary_loss_data, delimiter=',', fmt='%s')

        np.savetxt("MNIST_SBAE_test_output_" + str(z_size) + ".csv", test_output, delimiter=',')    

        # Get code(Z)

        get_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[2].output])
        test_z = get_z([X_test])[0]

        np.savetxt("MNIST_SBAE_test_z_" + str(z_size) + ".csv", test_z, delimiter=',')    

        # Calculate run time

        t12 = datetime.datetime.now()

        print(str(z_size) + " finish! run time " + str(t12 - t11))

        model_index = model_index + 1

    t02 = datetime.datetime.now()

    print("total run time " + str(t02 - t01))

    # Print total loss
    print(total_summary_loss_data)
