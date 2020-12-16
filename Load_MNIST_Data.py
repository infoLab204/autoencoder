import numpy as np

def choice_MINST_data(data_type, train, test) :
    if data_type=="digit" : 
        # Load library
        from keras.datasets import mnist
        # Load MNIST data
        (X_train, Y_train),(X_test, Y_test)=mnist.load_data()
    else : 
        from keras.datasets import fashion_mnist
        # Load Fashion-MNIST data
        (X_train, Y_train),(X_test, Y_test)=fashion_mnist.load_data()
        
    # Normalize(data range : 0 ~ 255 -> 0 ~ 1) and reshape(3-dimensional -> 2-dimensional) data
    X_train = X_train.reshape(train, 784).astype('float32') / 255.0
    X_test = X_test.reshape(test, 784).astype('float32') / 255.0

    # Mix data
    TRAIN_LIST = np.random.choice(range(train), train, replace=False)
    TEST_LIST = np.random.choice(range(test), test, replace=False)

    X_train2 = X_train[TRAIN_LIST]
    Y_train2 = Y_train[TRAIN_LIST]
    X_test2 = X_test[TEST_LIST]
    Y_test2 = Y_test[TEST_LIST]

    # Save data
    np.savetxt("MNIST_X_train.csv", X_train2, delimiter=',')
    np.savetxt("MNIST_Y_train.csv", Y_train2, delimiter=',')

    np.savetxt("MNIST_X_test.csv", X_test2, delimiter=',')
    np.savetxt("MNIST_Y_test.csv", Y_test2, delimiter=',')
