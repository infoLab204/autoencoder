import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras import optimizers
from keras import initializers

import datetime

def BAE(data_type, trainX, trainY, testX, testY, z_code, epoches, batch_size) :
    

    # Load MNIST train and test data

    X_train = np.loadtxt(trainX, delimiter=',', dtype=None)
    Y_train = np.loadtxt(trainY, delimiter=',', dtype=None)

    X_test = np.loadtxt(testX, delimiter=',', dtype=None)
    Y_test = np.loadtxt(testY, delimiter=',', dtype=None)
    
    # z_list : define experiment code(Z) size
#    z_list = [4, 8, 12, 16, 20]
    z_list=[z_code]
    autoencoder = [[] for i in range(len(z_list))]

    # E : epoch, BS = batch size
#     E = 200
#     BS = 100
    E = epoches
    BS = batch_size

    # Train model and save data(code(Z), output and total loss data)

    model_index = 0

    t01 = datetime.datetime.now()

    total_summary_loss_data = ['model_type', 'z_size', 'train_loss', 'test_loss']

    for z_size in z_list : 

        t11 = datetime.datetime.now()

        # Define models 

        INPUT_SIZE = 784
        HIDDEN_SIZE = z_size

        if data_type=="digit" :
            w_initializer = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
            b_initializer = initializers.zeros()

            dense1 = Input(shape=(INPUT_SIZE,))
            dense2 = Dense(HIDDEN_SIZE, activation='linear', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense1)
            dense3 = Dense(INPUT_SIZE, activation='sigmoid', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense2)

            autoencoder[model_index] = Model(dense1, dense3)

            adam = optimizers.Adam(lr=0.001)
            autoencoder[model_index].compile(loss='mean_squared_error', optimizer=adam)
            
            autoencoder[model_index].fit(X_train, X_train, epochs=E, batch_size=BS, verbose=0)
            
        else : 
            w_initializer = initializers.glorot_uniform(seed=None)
            b_initializer = initializers.glorot_uniform(seed=None)

            dense1 = Input(shape=(INPUT_SIZE,))
            dense2 = Dense(HIDDEN_SIZE, activation='linear', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense1)
            dense3 = Dense(INPUT_SIZE, activation='sigmoid', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense2)

            autoencoder[model_index] = Model(dense1, dense3)

            adagrad = optimizers.Adagrad(lr=0.01)
            autoencoder[model_index].compile(loss='mean_squared_error', optimizer=adagrad)

            autoencoder[model_index].fit(X_train, X_train, epochs=E, batch_size=BS, verbose=0)

        # Get output and calculate loss

        get_output = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[2].output])
        train_output = get_output([X_train])[0]
        test_output = get_output([X_test])[0]

        train_loss = np.sum((X_train - train_output)**2) / (X_train.shape[0] * X_train.shape[1])
        test_loss = np.sum((X_test - test_output)**2) / (X_test.shape[0] * X_test.shape[1])

        summary_loss_data = ['BAE', z_size, train_loss, test_loss]

        total_summary_loss_data = np.vstack((total_summary_loss_data, summary_loss_data))

        np.savetxt("BAE_total_loss_data.csv", total_summary_loss_data, delimiter=',', fmt='%s')

        np.savetxt("BAE_test_output_" + str(z_size) + ".csv", test_output, delimiter=',')    

        # Get code(Z)
        get_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[1].output])
        test_z = get_z([X_test])[0]

        np.savetxt("BAE_test_z_" + str(z_size) + ".csv", test_z, delimiter=',')    

        # Calculate run time

        t12 = datetime.datetime.now()

        print(str(z_size) + " finish! run time " + str(t12 - t11))

        model_index = model_index + 1

    t02 = datetime.datetime.now()

    print("total run time " + str(t02 - t01))

    
    # Print total loss
    print(total_summary_loss_data)
