# Lauto: A restorable autoencoder model of observation-wise linearity with Python and R


Yeongcheol Jeong, Sunhee Kim, and Chang-Yong Lee

Lauto represents Python and R scripts to assess the proposed auto-encoder model and to compare its performance with other models, such as basic autoencoder (BAE), stacked autoencoder (SAE), and principal component analysis (PCA), by using MNIST and Fashion-MNIST data sets.


We proposed a restorable autoencoder model as a non-linear method for reducing dimensionality. The Python and R scripts provide the assessment of the proposed model and compare with other models in terms of the loss function, image reconstruction, and classification results. We provide Python and R scripts together in order for the readers to reproduce the results discussed in the manuscript.

### Install prerequisites:
* R-packages: ggplot2, gridExtra, caret, e1071, nnet, dplyr
* Python: tensorflow (version 2.2 or later), keras, numpy, matplotlib, datetime

### Loading the scripts: 
   copy the following Python module and R scripts from its GitHub repository

* Python module: autoencoder.py
* Python functions: 	
    + load_data(): loading data set, such as MNIST and Fashion-MNIST
    + proposed(): learning proposed autoencoder model
    + basic(): learning basic autoencoder model
    + stacked(): learning stacked autoencoder model
    + recon(): image reconstruction of proposed and compared models
    + split(): store loss function according to the class label
* R functions:
    + pca.R: dimensionality reduction with principal component analysis
    + classification.R: performing classification analysis in terms of support vector machine
                            and multiple logistic regression


### Python scripts tutorial
* import the Python module 
    ```
    import autoencoder as auto
    
    ```
* Loading MNIST or Fashion-MNIST data sets
To load MNIST or Fashion-MNISY data from keras, run load_data() with following parameters.

    ```
    auto.load_data(type, ntrain, ntest) 
    
    ```
    + type: data type, either “digit” for MNIST or “fashion” for Fashion-MNIST
    + ntrain: number of training data		
    + ntest: number of test data    
    
    ```
    (eg) auto.load_data(“digit”, 60000, 10000)
    ```
    
    output : MNIST and Fashion-MNIST data sets and their labels
    

    (eg) MNIST_train.csv : train data set of MNIST   
           MNIST_train_label_csv:  train label data set of MNIST    
           MNIST_test.csv : test data set of MNIST   
           MNIST_test_label_csv: test label data set of MNIST    


* Learning autoencoder models
To learn the three autoencoder models, run proposed(), basic(), and stacked() for the proposed model, BAE, and SAE, respectively. The scripts will evaluate the units in the output layer. Outputs of the models will be the values of units in the output layer. 


    ```
    auto.proposed(type, train, test, code, epoch, batch)
    ```
    + type: data type, either “digit” for MNIST or “fashion” for Fashion-MNIST
    + train: train data		
    + test: test data		
    + code: number of nodes in the code layer
    + epoch: number of epochs		
    +  batch: batch size    

    ```
    (eg) auto.proposed(“digit”, “MNIST_train.csv”, “MNIST_test.csv”, 4, 200, 100)
    ```
    Output: loss function and values of units in the code and output layers.     
    (eg) proposed_total_loss.csv		proposed_test_code4.csv		proposed_test_out4.csv    
    (note) In a similar manner, learn basic() and stacked()    


* Reconstructing input images: Visualize_model_result_Image.py
To reconstruct input images, simply run Visualize_model_result_Image.py with the test images and values of units in the output layer as the input data set. Output will be the reconstructed images: test image, reconstructions by the proposed model, SAE, BAE,, and PCA.
Run Visualize_model_result_Image.py with the following parameters
    ```  
     import Visualize_Model_Result_Image as VMI
     VMI.visualize_image(testX, testY, choice_model, z_code)
     testX, testY: text data sets, choice_model : "LAE" or "BAE" or "SBAE" or "PCA",  z_code : z_size
     
     (eg) import visualize_model_result_image as VMI 
          VMI.visualize_image("MNIST_X_test.csv", "MNIST_Y_test.csv","BAE",20)
    ```   
    
* Performing PCA for the dimensionality reduction: MNIST_PCA.R
To reduce the dimensionality with PCA, simply run MNIST_PCA.R with MNIST and Fashion-MNIST as input data sets. Output will be the codes. 

    Run MNIST_PCA.R with the following parameters
    ```  
    MNIST_PCA(testX, z_code) 
    testX : text data, z_code : z_size
    
    (eg) MNIST_PCA("MNIST_X_test.csv", 4)

     ```  
* Evaluating the loss function for the proposed model, SAE, BAE, and PCA: Calculate_loss.R
To evaluate the loss function for all models, simply run Calculate_loss.R with the output of Lab_Auto_Encoder.py, Basic_Auto_Encoder.py, Stacked_Basic_Auto_Encoder.py, and MNIST_PCA.R, together with MNIST and Fashion-MNIST data sets. Output will be the loss function of all models.

    Calculate_loss.R with the following parameters
    ```  
    cal_loss(testX, choice_model, z_code)
    testX : text data, choice_model : "LAE" or "BAE" or "SBAE" or "PCA", z_code : z_size

    (eg) cal_loss("MNIST_X_test.csv","BAE",4)

     ```  
* Performing classification analysis in terms of support vector machine and multiple logistic regression: Model_classification.R
To classify MNIST and Fashion_MNIST data set, run Model_classification.R with the codes of all models as the input data. Output will be the classification results.

    Run Model_classification.R with the following parameters
    ```  
    model_class (testX, testY, choice_model, z_code,class_model)
    testX, testY, : text data set , choice_model : "LAE" or "BAE" or "SBAE" or "PCA", 
    z_code : z_size, class_model : "SVM" or "MLR"

    (eg) model_class("MNIST_X_test.csv", "MNIST_Y_test.csv","BAE", 4, "SVM")
     ```  
