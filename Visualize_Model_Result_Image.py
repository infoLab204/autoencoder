
# Load library
import numpy as np
from matplotlib import pyplot

def visualize_image(textX, textY, choice_model, z_code) : 

    # Choice MNIST or Fashion-MNIST

    data_type = 'MNIST'

    # Load data
    X_test = np.loadtxt(textX, delimiter=',', dtype=None)
    Y_test = np.loadtxt(textY, delimiter=',', dtype=None)

    z_list=[z_code]

    test_recon = [[]] * len(z_list)
    
    for z_index in range(len(z_list)) :
        z_size = z_list[z_index]   
        test_recon[z_index] = np.loadtxt(choice_model + "_test_output_" + str(z_size) + ".csv", delimiter=',', dtype=None)
        
            
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
    # 2nd row : seleced Auto-Encoder(LAE) reconstruct

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
 #   z_index = 0
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

#        z_index=z_index+1
        # pyplot.savefig(data_type + "_result_" + str(z_size) + "_image.eps", bbox_inches='tight', dpi=100)
        # pyplot.savefig(data_type + "_result_" + str(z_size) + "_image.png", bbox_inches='tight', dpi=100)            
