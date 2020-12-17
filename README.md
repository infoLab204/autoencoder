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
    + __recon()__: image reconstruction of each model
    + __split()__: store loss function according to the class label
* R functions:
    + __pca.R__: dimensionality reduction with principal component analysis
    + __classification.R__: performing classification analysis in terms of support vector machine and multiple logistic regression


### Python scripts tutorial
* __import the Python module__ 
    ```
    import autoencoder as auto    
    ```
* __Loading MNIST or Fashion-MNIST data sets__    
To load MNIST or Fashion-MNISY data from keras, run load_data() with following parameters.

    ```
    auto.load_data(type, ntrain, ntest)     
    ```
    + __type__: data type, either "digit" for MNIST or "fashion" for Fashion-MNIST
    + __ntrain__: number of training data		
    + __ntest__: number of test data <br><br>   
    
    ```
    (eg) auto.load_data("digit", 60000, 10000)
    ```
    
    __output__ : MNIST and Fashion-MNIST data sets and their labels
    

    __(eg)__ MNIST_train.csv : train data set of MNIST   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_train_label_csv:  label of train data set of MNIST    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_test.csv : test data set of MNIST   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MNIST_test_label_csv: label of test data set of MNIST    


* __Learning autoencoder models__    
To learn the three autoencoder models, run proposed(), basic(), and stacked() for the proposed model, BAE, and SAE, respectively. The scripts will evaluate the units in the output layer. Outputs of the models will be the values of units in the output layer. 


    ```
    auto.proposed(type, train, test, code, epoch, batch)
    ```
    + __type__: data type, either "digit" for MNIST or "fashion" for Fashion-MNIST
    + __train__: train data		
    + __test__: test data		
    + __code__: number of nodes in the code layer
    + __epoch__: number of epochs		
    + __batch__: batch size  <br><br>  

    ```
    (eg) auto.proposed("digit", "MNIST_train.csv", "MNIST_test.csv", 4, 200, 100)
    ```
    __Output__: loss function and values of units in the code and output layers.     
    __(eg)__ total_loss.csv&nbsp;&nbsp;&nbsp;&nbsp;test_code.csv&nbsp;&nbsp;&nbsp;&nbsp;test_out.csv    
    __(note)__ In a similar manner, learn basic() and stacked()    


* __Reconstructing input images__    
To reconstruct input images, simply run recon() with the test images and values of units in the output layer as the input data set. Output will be the reconstructed images: test image, reconstructions by the proposed model, SAE, BAE, and PCA.


    ```
    auto.recon(test, test_label, output)
    ```
    + __test__: test data		
    + __test_label__: label of each test datum
    + __output__: values of units in the output layer <br><br>
 

    ```  
    (eg) auto.recon("MNIST_test.csv", "MNIST_test_label.csv", "test_out.csv")
    ```
    __Output__: reconstructed images

* __Store loss function according to the class labels__    
To get loss function for each class, run split() with test data set and their class labels. Output will be loss functions of test data set for each class label.

    ```  
    auto.split(test, test_label,output)
    ```  
    + __test__: test data		
    + __test_label__: label of each test datum
    + __output__: values of units in the output layer <br><br>
    
    ```
    (eg) auto.split("MNIST_test.csv", "MNIST_test_label.csv", "test_out.csv")
    ```
    __Output__: MNIST_loss_class0.csv and MNIST_out_class0.csv for each class label

### R scripts tutorial
* __Performing PCA for the dimensionality reduction__   
To reduce the dimensionality with PCA, simply run pca.R with MNIST and Fashion-MNIST as input data sets. Output will be the dimensionality-reduced codes. 
    ```
    pca(test, code)
    ```
    + __test__: test data			
    + __code__: number of nodes in the code layer<br><br>
    
    ```
    (eg) pca("MNIST_test.csv", 4)
    ```
    __Output__: test_code.csv and test_out.csv

* __Performing classification analysis suing support vector machine and multiple logistic regression__        
To classify MNIST and Fashion_MNIST data set, run classification.R with the codes of all models as the input data. Output will be the classification results.
    ```
    classification(test, test_label, output, classifier)
    ```
    + __test__: test data		
    + __test_label__: label of each test datum
    + __output__: values of units in the output layer 
    + __classifier__: either "SVM" or "MLR" <br><br>


    ```
    (eg) classification("MNIST_test.csv", "MNIST_test_label.csv","test_out.csv", "SVM")
    ```
    __Output__: clssifier_result.csv     

    
 * __Evaluating the loss function for the proposed model, SAE, BAE, and PCA__   
 To evaluate the loss function for all models, simply run loss.R with the output of split(), together with MNIST and Fashion-MNIST data sets. Output will be the loss function of all models.    
 
   ```
    loss(ctest, coutput)
    ```
    + __cctest__: test data of each class label	
    + __coutput__: values of units in the output layer of each class label <br><br>


    ```
    (eg) loss("MNIST_loss_class0.csv", "MNIST_out_class0.csv")
    ```
    __Output__: loss functions of each label  

