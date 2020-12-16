
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras import optimizers
from keras import initializers

import datetime

from matplotlib import pyplot

def load_data(type, ntrain, ntest) :
    if type=="digit" : 
        # Load library
        from keras.datasets import mnist
        # Load MNIST data
        (X_train, Y_train),(X_test, Y_test)=mnist.load_data()
    else : 
        from keras.datasets import fashion_mnist
        # Load Fashion-MNIST data
        (X_train, Y_train),(X_test, Y_test)=fashion_mnist.load_data()
        
    # Normalize(data range : 0 ~ 255 -> 0 ~ 1) and reshape(3-dimensional -> 2-dimensional) data
    X_train = X_train.reshape(ntrain, 784).astype('float32') / 255.0
    X_test = X_test.reshape(ntest, 784).astype('float32') / 255.0

    # Mix data
    TRAIN_LIST = np.random.choice(range(ntrain), ntrain, replace=False)
    TEST_LIST = np.random.choice(range(ntest), ntest, replace=False)

    X_train2 = X_train[TRAIN_LIST]
    Y_train2 = Y_train[TRAIN_LIST]
    X_test2 = X_test[TEST_LIST]
    Y_test2 = Y_test[TEST_LIST]

    # Save data
    np.savetxt("MNIST_train.csv", X_train2, delimiter=',')
    np.savetxt("MNIST_train_label.csv", Y_train2, delimiter=',')

    np.savetxt("MNIST_test.csv", X_test2, delimiter=',')
    np.savetxt("MNIST_test_label.csv", Y_test2, delimiter=',')

# end of load_data()   
       

def proposed(type, train, test, code, epoch, batch) :
    # Load MNIST train and test data

    X_train = np.loadtxt(train, delimiter=',', dtype=None)
    X_test = np.loadtxt(test, delimiter=',', dtype=None)
    

    # z_list : define experiment code(Z) size
    z_list = [code]

    autoencoder = [[] for i in range(len(z_list))]

    # E : epoch, BS = batch size
    E = epoch
    BS = batch

    # Train model and save data(code(Z), output and total loss data)
    model_index = 0

    total_summary_loss_data = ['model_type', 'z_size', 'train_loss', 'test_loss']

    for z_size in z_list : 

        # Define models 

        INPUT_SIZE = 784
        HIDDEN_SIZE1 = 2000
        HIDDEN_SIZE2 = z_size
        
        if type=="digit" : 
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

        np.savetxt("proposed_total_loss.csv", total_summary_loss_data, delimiter=',', fmt='%s')

        # Get code(Z)

        get_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[2].output])
        test_z = get_z([X_test])[0]

        np.savetxt("proposed_test_code_"+str(code)+".csv", test_z, delimiter=',')

        model_index = model_index + 1


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

        np.savetxt("proposed_test_out_"+str(z_size)+".csv", test_recon, delimiter=',')
     

    # Print total loss
    print(total_summary_loss_data)        
    print("learning proposed autoencoder modol finish! \n")   

# end of proposed()    
    
def basic(type, train,test, code, epoch, batch) :
    

    # Load MNIST train and test data
    X_train = np.loadtxt(train, delimiter=',', dtype=None)
    X_test = np.loadtxt(test, delimiter=',', dtype=None)
    
    
    # z_list : define experiment code(Z) size
    z_list=[code]
    autoencoder = [[] for i in range(len(z_list))]

    # E : epoch, BS = batch size
    E = epoch
    BS = batch

    # Train model and save data(code(Z), output and total loss data)

    model_index = 0


    total_summary_loss_data = ['model_type', 'z_size', 'train_loss', 'test_loss']

    for z_size in z_list : 


        # Define models 

        INPUT_SIZE = 784
        HIDDEN_SIZE = z_size

        if type=="digit" :
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

        np.savetxt("basic_total_loss.csv", total_summary_loss_data, delimiter=',', fmt='%s')

        np.savetxt("basic_test_out_"+str(z_size)+".csv", test_output, delimiter=',')    

        # Get code(Z)
        get_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[1].output])
        test_z = get_z([X_test])[0]

        np.savetxt("basic_test_code_"+str(z_size)+".csv", test_z, delimiter=',')    

        model_index = model_index + 1

    
    # Print total loss
    print(total_summary_loss_data)   
    print("learning basic autoencoder model finish! \n")   
    
# end of basic()
    
def stacked(type, train, test, code,epoch, batch) :
    
    # Load MNIST train and test data
    X_train = np.loadtxt(train, delimiter=',', dtype=None)
    X_test = np.loadtxt(test, delimiter=',', dtype=None)
 
    # z_list : define experiment code(Z) size
    z_list = [code]

    autoencoder = [[] for i in range(len(z_list))]

    # E : epoch, BS = batch size
    E = epoch
    BS = batch

    # Train model and save data(code(Z), output and total loss data)

    model_index = 0

    total_summary_loss_data = ['model_type', 'z_size', 'train_loss', 'test_loss']

    # Define first pre-training(784 -> 400) model

    INPUT_SIZE = 784
    HIDDEN_SIZE = 400

    if type=="digit" : 
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

        # Define second pre-training(400 -> z_size) models

        INPUT_SIZE = 400
        HIDDEN_SIZE = z_size

        dense1 = Input(shape=(INPUT_SIZE,))
        dense2 = Dense(HIDDEN_SIZE, activation='tanh', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense1)
        dense3 = Dense(INPUT_SIZE, activation='relu', kernel_initializer=w_initializer, bias_initializer=b_initializer)(dense2)

        autoencoder[model_index] = Model(dense1, dense3)
    
        if type =="digit" : 
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

        if type=="digit" : 
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

        np.savetxt("stacked_total_loss.csv", total_summary_loss_data, delimiter=',', fmt='%s')

        np.savetxt("stacked_test_out_"+str(z_size)+".csv", test_output, delimiter=',')    

        # Get code(Z)

        get_z = K.function([autoencoder[model_index].layers[0].input],[autoencoder[model_index].layers[2].output])
        test_z = get_z([X_test])[0]

        np.savetxt("stacked_test_code_"+str(z_size)+".csv", test_z, delimiter=',')    

        model_index = model_index + 1

    # Print total loss
    print(total_summary_loss_data)    
    print("learning stacked autoencoder modol finish! \n") 
    
# end of stacked()    
    
    
    
def recon(text, text_label, model, code) : 

    # Load data
    X_test = np.loadtxt(text, delimiter=',', dtype=None)
    Y_test = np.loadtxt(text_label, delimiter=',', dtype=None)

    z_list=[code]

    test_recon = [[]] * len(z_list)
    
    
    if model=="LAE" : model="proposed" 
    elif model=="BAE" : model="basic" 
    elif model=="SAE"  : model="stacked"
    else : model="pca"    
        
        
    for z_index in range(len(z_list)) :
        z_size = z_list[z_index]   
        test_recon[z_index] = np.loadtxt(model + "_test_out_" + str(z_size) + ".csv", delimiter=',', dtype=None)
        
            
    # Split each class
    class_num = 10
    LT = [[]] * class_num

    for k in range(10) :
        LT[k] = []

    for i in range(len(Y_test)) :
        for j in range(len(LT)) :
            if Y_test[i] == j :
                LT[j] = np.append(LT[j], i)
                LT[j] = [int(LT[j]) for LT[j] in LT[j]]    


    # Image select in each class
    selected = [LT[0][0], LT[1][0], LT[2][0], LT[3][0], LT[4][0], LT[5][0], LT[6][0], LT[7][0], LT[8][0], LT[9][0]]

    X_test_selected = list(X_test[selected])

    test_recon_selected = [[]] * len(z_list)
 
    for z_index in range(len(z_list)) :
        test_recon_selected[z_index] = list(test_recon[z_index][selected])

    # Image consist of 5 rows
    # 1st row : Test data
    # 2nd row : seleced Model reconstruct

    # Image consist of 10 columns
    # 1st column : (MNIST or Fashion-MNIST) 0th class
    # ...
    # 10th column : (MNIST or Fashion-MNIST) 9th class    
 
    n = 10
    pyplot.figure(figsize=(10, 5))
    pyplot.subplots_adjust(top=1.15, bottom=0, right=1, left=0, hspace=0, wspace=0)

    for i in range(n) :
        ax = pyplot.subplot(5, n, i + 1)
        pyplot.imshow(X_test_selected[i].reshape(28, 28))
        pyplot.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for z_index in range(len(z_list)) : 
        z_size = z_list[z_index]
        print("z size : " + str(z_size))
        
        n = 10
        pyplot.figure(figsize=(10, 5))
        pyplot.subplots_adjust(top=1.15, bottom=0, right=1, left=0, hspace=0, wspace=0)

        for i in range(n) :
            ax = pyplot.subplot(5, n, i + 1 + n)
            pyplot.imshow(test_recon_selected[z_index][i].reshape(28, 28))
            pyplot.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
    print("image reconstruction finish! \n")    
    
# end of recon()    
    

    
def split(test, test_label, model, code) : 

    # Load data
    X_test = np.loadtxt(test, delimiter=',', dtype=None)
    Y_test = np.loadtxt(test_label, delimiter=',', dtype=None)

#    z_list = [4, 8, 12, 16, 20]

    z_list=[code]
    test_recon = [[]] * len(z_list)

    if model=="LAE" : model="proposed" 
    elif model=="BAE" : model="basic" 
    elif model=="SAE"  : model="stacked"
    else : model="pca" 
        
   
    for z_index in range(len(z_list)) :

        z_size = z_list[z_index]
        test_recon[z_index] = np.loadtxt(model+"_test_out_" + str(z_size) + ".csv", delimiter=',', dtype=None)
        

    # Split each class
    class_num = 10

    LT = [[]] * class_num

    for k in range(10) :
        LT[k] = []

    for i in range(len(Y_test)) :
        for j in range(len(LT)) :
            if Y_test[i] == j :
                LT[j] = np.append(LT[j], i)
                LT[j] = [int(LT[j]) for LT[j] in LT[j]]   

    X_test_class = [[]] * class_num
    test_recon_class = [[]] * len(z_list)


    for z_index in range(len(z_list)) :
        test_recon_class[z_index] = [[]] * class_num
     


    for class_index in range(class_num) :
        X_test_class[class_index] = X_test[LT[class_index]]

        for z_index in range(len(z_list)) :
            test_recon_class[z_index][class_index] = test_recon[z_index][LT[class_index]]

    # Save data
    for class_index in range(10) :
        np.savetxt("MNIST_test_class" + str(class_index) + ".csv", X_test_class[class_index], delimiter=',')

        for z_index in range(len(z_list)) :

            z_size = z_list[z_index]
            np.savetxt(model+"_test_out_" + str(z_size) + "_class" + str(class_index) + ".csv", test_recon_class[z_index][class_index], delimiter=',')
            
    print("finish!")  
    
    
# end of split()    
