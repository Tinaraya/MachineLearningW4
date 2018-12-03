## coursera practical machine learning week 4
## course project
## 2018-11-26
## predict activity quality using devices such as Jawbone, Nike, Fitbit
## use data from accelerometers on belt,forearm,arm, dumbell

## load libraries
library(caret)
library(rattle) # fancy tree plot

## read dataset from csv files
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

## get basic ideas from dataset
dim(training)
dim(testing)
str(training)
testing$classe    #testing set doesn't contain classe information

## if 90% of data from one column is NA, remove that column
col_i <- which(colSums(is.na(training)|training=="") > 0.9 * dim(training)[1])
training <- training[,-col_i]
testing <- testing[,-col_i]

## user info and time windows should not be included (columns 1-7)
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]

## check the new dataset, each location(eg.belt) has 13 acceleration related
## data, so together there are 13*4=52 columns of useful data, plus one
## column that needs prediction (classe), together there are 53 columns
dim(training)
str(training)


## training data partition so that accuracy of different methods can be compared
set.seed(30339)
inTrain <- createDataPartition(training$classe, p=0.8, list=FALSE)
train_data <- training[inTrain,]
test_data <- training[-inTrain,]
  
## predict "classe" variable based on other variables
## cross validation: k-folds method
n_folds <-5

## random forest method, more time consuming
model_rf <- train(classe ~., method ="rf", data = train_data,verbose=FALSE,
                  trControl = trainControl(method="cv",number=n_folds))
pd_rf <- predict(model_rf,test_data)

## visulize results
## 2 predictors give pretty good accuracy, if all 52 predictors were used,
## the accuracy will even decrease a little bit
## indicating the 52 predictor may not be fully independent
## the error rate will decrease if we increase the no. of trees to around 50
## then remain constant, no. of trees to 500 will not help with error reduction
print(model_rf)
plot(model_rf, main = "Random Forest Method")
confusionMatrix(pd_rf,test_data$classe)
plot(model_rf$finalModel, main = "Random Forest Method")

## compute variable importance order and model accuracy
order_rf <- varImp(model_rf)
accu_rf <- confusionMatrix(pd_rf,test_data$classe)$overall[1]

## classification trees method
model_ct <- train(classe ~., method ="rpart", data = train_data,
                 trControl = trainControl(method="cv",number=n_folds))
pd_ct <- predict(model_ct,test_data)

##visulize results
##accuracy is only around 0.5, not a promising approach
print(model_ct)
plot(model_ct, main = "Classification Tree Method")
confusionMatrix(pd_ct,test_data$classe)
fancyRpartPlot(model_ct$finalModel)

## compute variable importance order and model accuracy
order_ct <- varImp(model_ct)
accu_ct <- confusionMatrix(pd_ct,test_data$classe)$overall[1]

## Final prediction for test set, random forest method is chosen
prediction <- predict(model_rf,testing)
