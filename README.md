# Lauto: A restorable autoencoder model of observation-wise linearity with Python and R


Yeongcheol Jeong, Sunhee Kim, and Chang-Yong Lee

Lauto represents Python and R scripts to assess the proposed auto-encoder model and to compare its performance with other models, such as basic autoencoder (BAE), stacked autoencoder (SAE), and principal component analysis (PCA), by using MNIST and Fashion-MNIST data sets.


We proposed a restorable autoencoder model as a non-linear method for reducing dimensionality. The Python and R scripts provide the assessment of the proposed model and compare with other models in terms of the loss function, image reconstruction, and classification results. We provide Python and R scripts together in order for the readers to reproduce the results discussed in the manuscript.

### Install prerequisites:
* __R-packages__: ggplot2, gridExtra, caret, e1071, nnet, dplyr
* __Python__: tensorflow (version 2.2 or later), keras, numpy, matplotlib, datetime

### Loading the scripts: 
   copy the following Python module and R scripts from its GitHub repository

* Python module: __autoencoder.py__
* Python functions: 	
    + __load_data()__: loading data set, such as MNIST and Fashion-MNIST
    + __proposed()__: learning proposed autoencoder model
    + __basic()__: learning basic autoencoder model
    + __stacked()__: learning stacked autoencoder model
    + __recon()__: image reconstruction of proposed and compared models
    + __split()__: store loss function according to the class label
* R functions:
    + __pca.R__: dimensionality reduction with principal component analysis
    + __classification.R__: performing classification analysis in terms of support vector machine and multiple logistic regression


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
    + __type__: data type, either “digit” for MNIST or “fashion” for Fashion-MNIST
    + __ntrain__: number of training data		
    + __ntest__: number of test data <br>   
    
    ```
    (eg) auto.load_data(“digit”, 60000, 10000)
    ```
    
    __output__ : MNIST and Fashion-MNIST data sets and their labels
    

    __(eg)__ MNIST_train.csv : train data set of MNIST   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_train_label_csv:  train label data set of MNIST    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_test.csv : test data set of MNIST   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_test_label_csv: test label data set of MNIST    


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
    +  batch: batch size  <br>  

    ```
    (eg) auto.proposed(“digit”, “MNIST_train.csv”, “MNIST_test.csv”, 4, 200, 100)
    ```
    Output: loss function and values of units in the code and output layers.     
    (eg) proposed_total_loss.csv		proposed_test_code4.csv		proposed_test_out4.csv    
    (note) In a similar manner, learn basic() and stacked()    


* Reconstructing input images    
To reconstruct input images, simply run recon() with the test images and values of units in the output layer as the input data set. Output will be the reconstructed images: test image, reconstructions by the proposed model, SAE, BAE, and PCA.


    ```
    auto.recon(test, test_label, model, code)
    ```
    + test: test data		
    + test_label: label of each test datum
    + model: used model. “LAE” for proposed, “BAE” for basic, “SAE” for stacked,	and “PCA” for principal component analysis
    + code: number of nodes in the code layer
  

    ```  
    (eg) auto.recon(MNIST_test.csv, MNIST_test_label.csv, “LAE”, 4)
    ```
    Output: reconstructed images

* Store loss function according to the class labels    
To get loss function for each class, run split() with test data set and their class labels. Output will be loss functions of test data set for each class label.

    ```  
    auto.split(test, test_label, model, code)
    ```  
    + test: test data		
    + test_label: label of each test datum
    + model: used model: “LAE” for proposed, “BAE” for basic, “SAE” for stacked, and “PCA” for principal component analysis
    + code: number of nodes in the code layer
    
    ```
    (eg) auto.split(MNIST_test.csv, MNIST_test_label.csv, “LAE”, 4)
    ```
    Output: MNIST_loss_class0.csv

### R scripts tutorial
* Performing PCA for the dimensionality reduction    
To reduce the dimensionality with PCA, simply run pca.R with MNIST and Fashion-MNIST as input data sets. Output will be the dimensionality-reduced codes. 
    ```
    pca(test, code)
    ```
    + test: test data			
    + code: number of nodes in the code layer
    
    ```
    (eg) pca(MNIST_test.csv, 4)
    ```
    Output: pca_code4.csv   pca_out4.csv

* Performing classification analysis suing support vector machine and multiple logistic regression        
To classify MNIST and Fashion_MNIST data set, run classification.R with the codes of all models as the input data. Output will be the classification results.
    ```
    classification(test, test_label, model, code, classifier)
    ```
    + test: test data		
    + test_label: label of each test datum
    + model: used model: “LAE” for proposed, “BAE” for basic, “SAE” for stacked, and “PCA” for principal component analysis
    + code: number of nodes in the code layer
	 + classifier: either “SVM” or “MLR”
    
    ```
    (eg) classification(MNIST_test.csv, MNIST_test_label.csv, “LAE”, 4, “SVM”)
    ```
    Output: classfier_result.csv
