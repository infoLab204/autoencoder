import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras import optimizers
from keras import initializers

import datetime

def LAE(data_type, trainX, trainY, testX, testY, z_code, epoches, batch_size) :
    # Load MNIST train and test data

    X_train = np.loadtxt(trainX, delimiter=',', dtype=None)
    Y_train = np.loadtxt(trainY, delimiter=',', dtype=None)

    X_test = np.loadtxt(testX, delimiter=',', dtype=None)
    Y_test = np.loadtxt(testY, delimiter=',', dtype=None)

    # z_list : define experiment code(Z) size
    z_list = [z_code]

    autoencoder = [[] for i in range(len(z_list))]

    # E : epoch, BS = batch size
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
        HIDDEN_SIZE1 = 2000
        HIDDEN_SIZE2 = z_size
        
        if data_type=="digit" : 
            w_initializer = initializers.Orthogonal(gain=1.0, seed=None)
            b_initializer = initializers.random_normal(mean=0.0, stddev=0.05, seed=None)
        else : 
            w_initializer = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            b_initializer = initializers.glorot_normal(seed=None)
            
        dense1 = Input(shape=(INPUT_SIZE,))
        dense2 = Dense(HIDDEN_SIZE1, activation='relu', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense1)
        dense3 = Dense(HIDDEN_SIZE2, activation='linear', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense2)
        dense4 = Dense(HIDDEN_SIZE1, activation='relu', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense3)
        dense5 = Dense(INPUT_SIZE, activation='linear', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense4)

        autoencoder[model_index] = Model(dense1, dense5)

        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        autoencoder[model_index].compile(loss='mean_squared_error', optimizer=sgd)

        autoencoder[model_index].fit(X_train, X_train, epochs=E, batch_size=BS, verbose=0)

        # Get output and calculate loss

        get_output = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[4].output])
        train_output = get_output([X_train])[0]
        test_output = get_output([X_test])[0]

        train_loss = np.sum((X_train - train_output)**2) / (X_train.shape[0] * X_train.shape[1])
        test_loss = np.sum((X_test - test_output)**2) / (X_test.shape[0] * X_test.shape[1])

        summary_loss_data = ['LAE', z_size, train_loss, test_loss]

        total_summary_loss_data = np.vstack((total_summary_loss_data, summary_loss_data))

        np.savetxt("LAE_total_loss_data.csv", total_summary_loss_data, delimiter=',', fmt='%s')

        # Get code(Z)

        get_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[2].output])
        test_z = get_z([X_test])[0]

        np.savetxt("LAE_test_z_" + str(z_size) + ".csv", test_z, delimiter=',')

        # Calculate run time

        t12 = datetime.datetime.now()

        print(str(z_size) + " finish! run time " + str(t12 - t11))

        model_index = model_index + 1

    t02 = datetime.datetime.now()

    print("total run time " + str(t02 - t01))


    # Print total loss
    print(total_summary_loss_data)


    # Calculate and save LAE reconstruct (* LAE reconstruct = LAE output)
    for z_index in range(0, len(z_list)) :

        z_size = z_list[z_index]

        test_recon = [[]]

        get_z = K.function([autoencoder[z_index].layers[0].input],[autoencoder[z_index].layers[2].output])
        temp_Z = get_z([X_test])[0]

        for data_index in range(0, 10000) :

            temp_vec = np.dot(temp_Z[data_index], autoencoder[z_index].layers[3].get_weights()[0]) + autoencoder[z_index].layers[3].get_weights()[1]        

            B = []
            for k in range(0, len(temp_vec)) :
                if(temp_vec[k] >= 0.0) :
                    B.extend([k])

            w34 = autoencoder[z_index].layers[3].get_weights()[0]
            w34 = w34[:, B]
            w45 = autoencoder[z_index].layers[4].get_weights()[0]
            w45 = w45[B, :]

            W35 = np.dot(w34, w45)

            b3 = autoencoder[z_index].layers[3].get_weights()[1]
            b3 = b3[B]
            b4 = autoencoder[z_index].layers[4].get_weights()[1]

            temp_recon = np.dot(temp_Z[data_index], W35) + np.dot(b3, w45) + b4

            if(data_index == 0) :
                test_recon = temp_recon
            else :
                test_recon = np.vstack((test_recon, temp_recon))

            if((data_index + 1) % 1000 == 0) :
                print(str(data_index + 1) + " finish!")

        np.savetxt("LAE_test_output_" + str(z_size) + ".csv", test_recon, delimiter=',')

        print(str(z_size) + " finish! \n")
