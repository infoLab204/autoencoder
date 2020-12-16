
# Load library
import numpy as np

def split_data(testX, testY, choice_model, z_code) : 

    # Load data
    X_test = np.loadtxt(testX, delimiter=',', dtype=None)
    Y_test = np.loadtxt(testY, delimiter=',', dtype=None)

#    z_list = [4, 8, 12, 16, 20]

    z_list=[z_code]
    test_recon = [[]] * len(z_list)

   
    for z_index in range(len(z_list)) :

        z_size = z_list[z_index]
        test_recon[z_index] = np.loadtxt(choice_model+"_test_output_" + str(z_size) + ".csv", delimiter=',', dtype=None)
        

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
        np.savetxt("MNIST_X_test_class" + str(class_index) + ".csv", X_test_class[class_index], delimiter=',')

        for z_index in range(len(z_list)) :

            z_size = z_list[z_index]
            np.savetxt(choice_model+"_test_recon_" + str(z_size) + "_class" + str(class_index) + ".csv", test_recon_class[z_index][class_index], delimiter=',')
            
        print(str(class_index) + " finish!")        
