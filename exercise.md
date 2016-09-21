# Practical Machine Learning Course Project

# Practical Machine Learning Course Project
========================================================

## Synopsis: 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement â a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The goal of this project is to predict the manner of performing unilateral dumbbell biceps curls based on data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The 5 possible methods include -
* A: exactly according to the specification 
* B: throwing the elbows to the front
* C: lifting the dumbbell only halfway 
* D: lowering the dumbbell only halfway
* E: throwing the hips to the front

#### Load libraries and setup working directory

```r
rm(list = ls(all = TRUE))

setwd('.')

library(caret)

trainingRaw <- read.csv(file="pml-training.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))
testingRaw <- read.csv(file="pml-testing.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))

trainingRaw$classe <- as.factor(trainingRaw$classe)  
```

#### Examine the data

```
## 'data.frame':  19622 obs. of  160 variables:

```

#### Cleaning variables
After investigating all the variables of the sets, it's possible to see that there are a lot of values NA or useless or empty variables for the prediction. It's request to compute the prediction only on the accelerometers values of belt, forearm, arm and dumbell. So, the non-accelerometer measures are discard with the useless variables.


```r
NAindex <- apply(trainingRaw,2,function(x) {sum(is.na(x))}) 
trainingRaw <- trainingRaw[,which(NAindex == 0)]
NAindex <- apply(testingRaw,2,function(x) {sum(is.na(x))}) 
testingRaw <- testingRaw[,which(NAindex == 0)]
```
#### Preprocessing variables

```r
v <- which(lapply(trainingRaw, class) %in% "numeric")

preObj <-preProcess(trainingRaw[,v],method=c('knnImpute', 'center', 'scale'))
trainLess1 <- predict(preObj, trainingRaw[,v])
trainLess1$classe <- trainingRaw$classe

testLess1 <-predict(preObj,testingRaw[,v])
```
#### Removing the non zero variables
Removing the variables with values near zero, that means that they have not so much meaning in the predictions

```r
nzv <- nearZeroVar(trainLess1,saveMetrics=TRUE)
trainLess1 <- trainLess1[,nzv$nzv==FALSE]

nzv <- nearZeroVar(testLess1,saveMetrics=TRUE)
testLess1 <- testLess1[,nzv$nzv==FALSE]
```

#### Create cross validation set
The training set is divided in two parts, one for training and the other for cross validation


```r
set.seed(12031987)

inTrain = createDataPartition(trainLess1$classe, p = 3/4, list=FALSE)
training = trainLess1[inTrain,]
crossValidation = trainLess1[-inTrain,]
```

#### Train model
Train model with random forest due to its highly accuracy rate. The model is build on a training set of 28 variables from the initial 160. Cross validation is used as train control method.

```r
modFit <- train(classe ~., method="rf", data=train, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )
```
````
## Time difference of 13.05325 mins

## Random Forest 
## 
## 14718 samples
##    27 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 13246, 13245, 13248, 13245, 13247, 13246, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.993     0.991  0.00116      0.00147 
##   14    0.992     0.99   0.0028       0.00354 
##   27    0.989     0.987  0.00353      0.00446 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.  
````

#### Accuracy on training set and cross validation set
Following the computation on the accuracy of trainig and cross validation set

Training set:

```r
trainingPred <- predict(modFit, training)
confusionMatrix(trainingPred, training$classe)
```
````
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 4185    0    0    0    0
         B    0 2848    0    0    0
         C    0    0 2567    0    0
         D    0    0    0 2412    0
         E    0    0    0    0 2706

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9997, 1)
    No Information Rate : 0.2843     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
````

Cross validation set

```r
cvPred <- predict(modFit, crossValidation)
confusionMatrix(cvPred, crossValidation$classe)
```
````
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1392    3    0    0    0
         B    2  944    2    0    0
         C    0    2  852    3    0
         D    0    0    1  801    3
         E    1    0    0    0  898

Overall Statistics
                                         
               Accuracy : 0.9965         
                 95% CI : (0.9945, 0.998)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9956         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9978   0.9947   0.9965   0.9963   0.9967
Specificity            0.9991   0.9990   0.9988   0.9990   0.9998
Pos Pred Value         0.9978   0.9958   0.9942   0.9950   0.9989
Neg Pred Value         0.9991   0.9987   0.9993   0.9993   0.9993
Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
Detection Rate         0.2838   0.1925   0.1737   0.1633   0.1831
Detection Prevalence   0.2845   0.1933   0.1748   0.1642   0.1833
Balanced Accuracy      0.9985   0.9969   0.9976   0.9976   0.9982
````

#### RESULTS
Predictions on the real testing set

```r
testingPred <- predict(modFit, testLess1)
testingPred
```
````
 [1] B A B A A E D B A A B C B A E E A B B B
````
