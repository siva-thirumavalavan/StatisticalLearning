rm(list = ls())
options(warn = -1)


### libraries ####
# We pass the required libraries as a list. The following snippet checks the local libraries and installs the packages that are not available
packageList <- c('Rcpp','MASS','glmnet','ggplot2','dplyr','tidyr','ANN2','corrplot','caret','writexl','sparseSVM','rstudioapi','e1071','rminer')
install.packages(setdiff(packageList,rownames(installed.packages())))
lapply(packageList,require, character.only = T)

### the code detects it's current location and sets it as the working directory. The user doesn't have to specify a working directory
workingDirectory <- unlist(strsplit(unlist(rstudioapi::getActiveDocumentContext()[2]),"/"))
workingDirectory <- paste(workingDirectory[-length(workingDirectory)],collapse = '//')
setwd(workingDirectory)

#### functions ####
#min_max scaling - to scale each feature in the range of 0 to 1
minMaxScale <- function(iterCol){
  return((iterCol-min(iterCol))/(max(iterCol)-min(iterCol)))
}

#get the list of significant variables from lm summary based on p-value
getSignificantEquation <- function(modelName,pThreshold){
  modelSummary <- as.data.frame(summary(modelName)$coefficients[-1,4])
  significantVariables <- rownames(modelSummary)[round(modelSummary[,1],2)<=pThreshold]
  return(paste(significantVariables,collapse = ' + '))
}

#to find rmse and r-square from the actual and predicted values
modelParameters <- function(predictions,model.name,actualCol){
  return(data.frame( model = model.name,
                     RMSE = round(RMSE(predictions, actualCol),3), 
                     Rsquare = round(R2(predictions, actualCol)[1],3)))
}

#to remove the estimates that are very low - we round the estimates to 2 decimal places and remove the zeros.
removeWeakEstimates <- function(iterCol){
  return(as.numeric(gsub('0$',NA,as.character(round(iterCol,2)))))
}

#### Analysis of Boston Data ####
#creating a folder to save the results
subDirectory <- 'Boston Results'
dir.create(subDirectory)
setwd(paste(workingDirectory,subDirectory,sep = '\\'))

#load the boston dataset
data(Boston)
summary(Boston)

#we are normalizing our x variables
BostonNormalised <-  Boston
BostonNormalised[,-which(names(Boston)=='crim')] <- data.frame(sapply(Boston[,-which(names(Boston)=='crim')], minMaxScale))


#we rename the columns for better understanding
names(BostonNormalised) <- c('CrimeRate','Zone','Industry','CharlesRiver','NitrogenOxide','RoomsPerDwelling','Age','Distance','RadialHighway','Tax','PTratio','Black','LowerStatus','MedianValue')

#checking distribution of each column
png('Boston Dataset - Variable Distribution.png',width = 1280, height = 720)
BostonNormalised %>% gather() %>% ggplot(aes(value))+facet_wrap(~key,scales = 'free')+geom_density()
dev.off()
png('Boston Dataset - Correlation.png',width = 1280, height = 720)
corrplot(cor(BostonNormalised))
dev.off()

#converting the dataset to long format and plotting each column against per capita crime rate
BostonLongFormat <- gather(BostonNormalised,key = 'var',value = 'value' ,-CrimeRate)
png('Boston Dataset - Y vs X_1.png',width = 1280, height = 720, res = 90)
ggplot(BostonLongFormat[BostonLongFormat$var %in% names(BostonNormalised)[1:5],],aes(x=value,y=CrimeRate, col = var)) + geom_point() + geom_smooth(method = 'lm')+ facet_grid(var ~. )
dev.off()

BostonLongFormat <- gather(BostonNormalised,key = 'var',value = 'value' ,-CrimeRate)
png('Boston Dataset - Y vs X_2.png',width = 1280, height = 720, res = 90)
ggplot(BostonLongFormat[BostonLongFormat$var %in% names(BostonNormalised)[6:10],],aes(x=value,y=CrimeRate, col = var)) + geom_point() + geom_smooth(method = 'lm')+ facet_grid(var ~. )
dev.off()


BostonLongFormat <- gather(BostonNormalised,key = 'var',value = 'value' ,-CrimeRate)
png('Boston Dataset - Y vs X_3.png',width = 1280, height = 720, res = 90)
ggplot(BostonLongFormat[BostonLongFormat$var %in% names(BostonNormalised)[11:14],],aes(x=value,y=CrimeRate, col = var)) + geom_point() + geom_smooth(method = 'lm')+ facet_grid(var ~. )
dev.off()

#### Data Split ####
set.seed(123)
trainIndex <- createDataPartition(BostonNormalised$CrimeRate,p=0.8,list=FALSE) 
Boston.train <- BostonNormalised[trainIndex,] 
Boston.test <- BostonNormalised[-trainIndex,] 
# we use cross validation techniques to split train and validation datasets
ctrlParameters  <- trainControl(method  = "cv",
                                number  = 10,
                                savePredictions = T)
Y_index <- which(names(BostonNormalised)=='CrimeRate')

## 1. Multiple Linear Regression with all the variables ####
tic <- Sys.time()
model.lr <- train(CrimeRate ~ .,
             data = Boston.train, 
             method = "lm",
             metric ='RMSE',
             trControl = ctrlParameters) 
summary(model.lr)
toc <- Sys.time()
summary(model.lr)
predictions.lr <- predict(model.lr,Boston.test)
estimates.lr <- data.frame(lr = coef(model.lr$finalModel, model.lr$bestTune$intercept)[-1])
results.lr <- modelParameters(predictions.lr,'lr',Boston.test$CrimeRate)
results.lr$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.lr$Accuracy <- NA
results.lr$alpha <- NA
results.lr$lambda <- NA

#we identify the significant variables from the summary and run another iteration with only those variables
model.lr.2 <- train(formula(paste('CrimeRate ~',getSignificantEquation(model.lr,0.05))),
                     data = Boston.train, 
                     method = "lm",
                     metric ='RMSE',
                     trControl = ctrlParameters) 
summary(model.lr.2)
predictions.lr.2 <- predict(model.lr.2,Boston.test)
modelParameters(predictions.lr,'lr',Boston.test$CrimeRate)



# 1.1 Linear Regression + Ridge ####
modelLookup("glmnet")
tic <- Sys.time()
# we tune for the hyper-parameters that gives the least RMSE value
model.lr.ridge <- train(CrimeRate ~.,
                         data = Boston.train, 
                         method = "glmnet",
                         trControl = ctrlParameters,
                         tuneGrid = expand.grid(alpha=0,lambda=10^seq(0,-5,length=1000)))
toc <- Sys.time()

png('Boston Dataset - LM Ridge.png',width = 1280, height = 720)
plot(model.lr.ridge)
dev.off()


estimates.lr.ridge <- data.frame(lr.ridge = coef(model.lr.ridge$finalModel, model.lr.ridge$bestTune$lambda)[-1,1])
predictions.lr.ridge <- predict(model.lr.ridge,as.matrix(Boston.test[,-Y_index]))
results.lr.ridge <-modelParameters(predictions.lr.ridge,'lr.ridge',Boston.test$CrimeRate)
results.lr.ridge$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.lr.ridge$Accuracy <- NA
results.lr.ridge$alpha <- 0
results.lr.ridge$lambda <- round(model.lr.ridge$bestTune$lambda,3)
results.lr.ridge

# 1.2 Linear Regression + Lasso ####

tic <- Sys.time()
# we tune for the hyperprameters that gives the least RMSE value
model.lr.lasso <- train(CrimeRate ~.,
                         data = Boston.train, 
                         method = "glmnet",
                         trControl = ctrlParameters,
                         tuneGrid = expand.grid(alpha=1,lambda=10^seq(0,-5,length=500)))
toc <- Sys.time()

png('Boston Dataset - LM Lasso.png',width = 1280, height = 720)
plot(model.lr.lasso)
dev.off()

estimates.lr.lasso <- data.frame(lr.lasso = coef(model.lr.lasso$finalModel, model.lr.lasso$bestTune$lambda)[-1,1])
predictions.lr.lasso <- predict(model.lr.lasso,as.matrix(Boston.test[,-Y_index]))
results.lr.lasso <-modelParameters(predictions.lr.lasso,'lr.lasso',Boston.test$CrimeRate)
results.lr.lasso$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.lr.lasso$Accuracy <- NA
results.lr.lasso$alpha <- 1
results.lr.lasso$lambda <- round(model.lr.lasso$bestTune$lambda,3)
results.lr.lasso

# 1.3 Linear Regression + elastic ####

tic <- Sys.time()
# we tune for the hyperprameters that gives the least RMSE value
model.lr.elastic <- train(CrimeRate ~.,
                           data = Boston.train, 
                           method = "glmnet" ,
                           trControl = ctrlParameters,
                           tuneGrid = expand.grid(alpha=c(0:10)/10,lambda=10^seq(0,-5,length=500)))
toc <- Sys.time()

png('Boston Dataset - LM elastic.png',width = 1280, height = 720)
plot(model.lr.elastic)
dev.off()

estimates.lr.elastic <- data.frame(lr.elastic = coef(model.lr.elastic$finalModel, model.lr.elastic$bestTune$lambda)[-1,1])
predictions.lr.elastic <- predict(model.lr.elastic,as.matrix(Boston.test[,-Y_index]))
results.lr.elastic <-modelParameters(predictions.lr.elastic,'lr.elastic',Boston.test$CrimeRate)
results.lr.elastic$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.lr.elastic$Accuracy <- NA
results.lr.elastic$alpha <- round(model.lr.elastic$bestTune$alpha,3)
results.lr.elastic$lambda <- round(model.lr.elastic$bestTune$lambda,3)
results.lr.elastic


## 2. Neural Network ####
# a perceptron with single hidden layer emulates a simple linear regression. This will enable us to treat the weights as estimates

tic <- Sys.time()
model.nn <- neuralnetwork(as.matrix(Boston.train[,-Y_index]),
                          as.matrix(Boston.train[,Y_index]),
                          hidden.layers =1,
                          optim.type = 'adam',
                          n.epochs = 1000,
                          regression = T,
                          learn.rates = 0.001, 
                          loss.type = 'squared', 
                          verbose = T,
                          activ.functions = 'relu',
                          L1 = 0,
                          L2 = 0)
toc <- Sys.time()
estimates.nn <- data.frame(nn = c(model.nn$Rcpp_ANN$getParams()$weights[[1]]))
predictions.nn <- predict(model.nn,as.matrix(Boston.test[,-Y_index]))
results.nn <- modelParameters(predictions.nn$predictions,'nn',Boston.test$CrimeRate)
results.nn$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.nn$Accuracy <- NA
results.nn$alpha <- NA
results.nn$lambda <- NA
results.nn



# 2.1 Neural Network + lasso ####

tic <- Sys.time()
model.nn.lasso <- neuralnetwork(as.matrix(Boston.train[,-Y_index]),
                                as.matrix(Boston.train[,Y_index]),
                                hidden.layers =1,
                                optim.type = 'adam',
                                n.epochs = 1000,
                                regression = T,
                                learn.rates = 0.001, 
                                loss.type = 'squared', 
                                verbose = F,
                                activ.functions = 'relu',
                                L1 = 1,
                                L2 = 0)
toc <- Sys.time()

estimates.nn.lasso <- data.frame(nn.lasso = c(model.nn.lasso$Rcpp_ANN$getParams()$weights[[1]]))
predictions.nn.lasso <- predict(model.nn.lasso,as.matrix(Boston.test[,-Y_index]))
results.nn.lasso <- modelParameters(predictions.nn.lasso$predictions,'nn.lasso',Boston.test$CrimeRate)
results.nn.lasso$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.nn.lasso$Accuracy <- NA
results.nn.lasso$alpha <- 1
results.nn.lasso$lambda <- NA
results.nn.lasso

# 2.2 Neural Network + ridge ####

tic <- Sys.time()
model.nn.ridge <- neuralnetwork(as.matrix(Boston.train[,-Y_index]),
                                as.matrix(Boston.train[,Y_index]),
                                hidden.layers =1,
                                optim.type = 'adam',
                                n.epochs = 1000,
                                regression = T,
                                learn.rates = 0.001, 
                                loss.type = 'squared', 
                                verbose = F,
                                activ.functions = 'relu',
                                L1 = 0,
                                L2 = 1)
toc <- Sys.time()

estimates.nn.ridge <- data.frame(nn.ridge = c(model.nn.ridge$Rcpp_ANN$getParams()$weights[[1]]))
predictions.nn.ridge <- predict(model.nn.ridge,as.matrix(Boston.test[,-Y_index]))
results.nn.ridge <- modelParameters(predictions.nn.ridge$predictions,'nn.ridge',Boston.test$CrimeRate)
results.nn.ridge$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.nn.ridge$Accuracy <- NA
results.nn.ridge$alpha <- 0
results.nn.ridge$lambda <- NA
results.nn.ridge

# 2.3 Neural Network + elastic ####

tic <- Sys.time()
model.nn.elastic <- neuralnetwork(as.matrix(Boston.train[,-Y_index]),
                                  as.matrix(Boston.train[,Y_index]),
                                  hidden.layers =1,
                                  optim.type = 'adam',
                                  n.epochs = 1000,
                                  regression = T,
                                  learn.rates = 0.001, 
                                  loss.type = 'squared', 
                                  verbose = F,
                                  activ.functions = 'relu',
                                  L1 = 1,
                                  L2 = 1)

toc <- Sys.time()

estimates.nn.elastic <-  data.frame(nn.elastic = c(model.nn.elastic$Rcpp_ANN$getParams()$weights[[1]]))
predictions.nn.elastic <- predict(model.nn.elastic,as.matrix(Boston.test[,-Y_index]))
results.nn.elastic <- modelParameters(predictions.nn.elastic$predictions,'nn.elastic',Boston.test$CrimeRate)
results.nn.elastic$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.nn.elastic$Accuracy <- NA
results.nn.elastic$alpha <- NA
results.nn.elastic$lambda <- NA
results.nn.elastic

## 3. SVM ####
tic <- Sys.time()
model.SVM <- tune.svm(CrimeRate~.,
                      data = Boston.train,
                      tunecontrol = tune.control(cross = 10),
                      kernel = 'linear')

toc <- Sys.time()
summary(model.SVM$best.model)
predictions.SVM <- predict(model.SVM$best.model,Boston.test)
estimates.SVM <- data.frame(SVM = coef(model.SVM$best.model)[-1])
results.SVM <- modelParameters(predictions.SVM,'SVM',Boston.test$CrimeRate)
results.SVM$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.SVM$Accuracy <- NA
results.SVM$alpha <- NA
results.SVM$lambda <- NA
results.SVM

svmPredictor <- function(model,data){
  return(predict(model,data))
}

variableImportance.SVM <- data.frame(Var = names(Boston.test[,-Y_index]),
  Importance = round(Importance(model.SVM$best.model,Boston.test, PRED = svmPredictor)$imp[-1],2)
)

#### 3.1 SVM + Ridge ####
# we convert our target variable into a binary class to fit the data in SVM with regularization
BostonNormalised.SVM <- BostonNormalised
BostonNormalised.SVM$CrimeRate <- 'low'
BostonNormalised.SVM$CrimeRate[BostonNormalised$CrimeRate>=quantile(BostonNormalised$CrimeRate,0.5,names = F)] <- 'high'
Boston.train.SVM <- BostonNormalised.SVM[trainIndex,] 
Boston.test.SVM <- BostonNormalised.SVM[-trainIndex,] 

tic <- Sys.time()
#we use cross-validation to find the hyper parameters that gives  us the least misclassification error
crossValidaton.ridge <- cv.sparseSVM(as.matrix(Boston.train.SVM[,-Y_index]),
                                     Boston.train.SVM[,Y_index],
                                     nfolds = 10,
                                     eval.metric = 'me')

model.SVM.ridge <- sparseSVM(as.matrix(Boston.train.SVM[,-1]),
                             Boston.train.SVM[,1],
                             alpha = 0,
                             lambda = crossValidaton.ridge$lambda.min)
toc <- Sys.time()
print(model.SVM.ridge)


estimates.SVM.ridge <- data.frame(SVM.ridge = model.SVM.ridge$weights[-1])
predictions.SVM.ridge<- predict(model.SVM.ridge,X = as.matrix(Boston.test.SVM[,-Y_index])) 
confMatrix.SVM.ridge  <- table(Predicted=predictions.SVM.ridge,Actual=Boston.test.SVM$CrimeRate)
results.SVM.ridge <- data.frame(model= 'SVM.ridge',
                                RMSE = NA,
                                Rsquare = NA,
                                runTime = round(as.numeric(difftime(toc,tic,units = 'mins')),3),
                                Accuracy = sum(diag(confMatrix.SVM.ridge ))/nrow(Boston.test.SVM),
                                alpha = 0,
                                lambda = round(crossValidaton.ridge$lambda.min,2))
results.SVM.ridge



#### 3.2 SVM + Lasso ####
tic <- Sys.time()
#we use cross-validation to find the hyper parameters that gives  us the least misclassification error
crossValidaton.lasso <- cv.sparseSVM(as.matrix(Boston.train.SVM[,-Y_index]),
                                     Boston.train.SVM[,Y_index],
                                     nfolds = 10,
                                     eval.metric = 'me')

model.SVM.lasso <- sparseSVM(as.matrix(Boston.train.SVM[,-Y_index]),
                             Boston.train.SVM[,Y_index],
                             alpha = 1,
                             lambda = crossValidaton.lasso$lambda.min)
toc <- Sys.time()
print(model.SVM.lasso)

estimates.SVM.lasso <- data.frame(SVM.lasso = model.SVM.lasso$weights[-1])
predictions.SVM.lasso<- predict(model.SVM.lasso,X = as.matrix(Boston.test.SVM[,-Y_index])) 
confMatrix.SVM.lasso  <- table(Predicted=predictions.SVM.lasso,Actual=Boston.test.SVM$CrimeRate)
results.SVM.lasso <- data.frame(model= 'SVM.lasso',
                                RMSE = NA,
                                Rsquare = NA,
                                runTime = round(as.numeric(difftime(toc,tic,units = 'mins')),3),
                                Accuracy = sum(diag(confMatrix.SVM.lasso ))/nrow(Boston.test.SVM),
                                alpha = 1,
                                lambda = round(crossValidaton.lasso$lambda.min,2))

results.SVM.lasso

#### 3.3 SVM + elastic ####
tic <- Sys.time()
#we use cross-validation to find the hyper parameters that gives  us the least misclassification error
crossValidaton.elastic <- cv.sparseSVM(as.matrix(Boston.train.SVM[,-Y_index]),
                                       Boston.train.SVM[,Y_index],
                                       nfolds = 10,
                                       eval.metric = 'me')
estimates <- data.frame(Varibale <- names(Boston.train.SVM[,-1]))
IterResults <- data.frame(alpha = NA, Accuracy = NA)

#since cv.sparseSVM() doesn't support the tuning of alpha, we ru our model for different alpha values and select the one that gives the least misclassification error / max accuracy
for (alphaIter in seq(0,1,length=10)) {
  alphaIter <- round(alphaIter,1)
  model.SVM.elastic <- sparseSVM(as.matrix(Boston.train.SVM[,-Y_index]),
                                 Boston.train.SVM[,Y_index],
                                 alpha = alphaIter,
                                 lambda = crossValidaton.elastic$lambda.min)
  
  predictions<- predict(model.SVM.elastic,X = as.matrix(Boston.test.SVM[,-Y_index])) 
  confMatrix <- table(Predicted=predictions,Actual=Boston.test.SVM$CrimeRate)
  IterResults <- rbind.data.frame(IterResults,
                                  data.frame(alpha = alphaIter,
                                             Accuracy = sum(diag(confMatrix))/nrow(Boston.test.SVM)))
  estimateIter <- data.frame(col1= model.SVM.elastic$weights[-1])
  names(estimateIter) <- alphaIter
  estimates <- cbind.data.frame(estimates,estimateIter)
}
toc <- Sys.time()

estimates.SVM.elastic <- data.frame(SVM.elastic = estimates[,as.character(IterResults$alpha[which.max(IterResults$Accuracy)])])
paste('The best model parameters are alpha =',IterResults$alpha[which.max(IterResults$Accuracy)],'and lambda =',round(crossValidaton.elastic$lambda.min,2))
results.SVM.elastic <- data.frame(model = 'SVM.elastic',
                                  RMSE = NA,
                                  Rsquare = NA,
                                  runTime = round(as.numeric(difftime(toc,tic,units = 'mins')),3),
                                  Accuracy = max(IterResults$Accuracy,na.rm = T),
                                  alpha = IterResults$alpha[which.max(IterResults$Accuracy)],
                                  lambda = round(crossValidaton.elastic$lambda.min,2))

results.SVM.elastic

### Results ####
#combining individual model results and metrics into a single dataframe
modelComparision <- rbind.data.frame(results.lr,
                                     results.lr.lasso,
                                     results.lr.ridge,
                                     results.lr.elastic,
                                     results.nn,
                                     results.nn.lasso,
                                     results.nn.ridge,
                                     results.nn.elastic,
                                     results.SVM,
                                     results.SVM.lasso,
                                     results.SVM.ridge,
                                     results.SVM.elastic
                                     )

estimateComparision <- cbind.data.frame(estimates.lr,
                                        estimates.lr.lasso,
                                        estimates.lr.ridge,
                                        estimates.lr.elastic,
                                        estimates.nn,
                                        estimates.nn.lasso,
                                        estimates.nn.ridge,
                                        estimates.nn.elastic,
                                        estimates.SVM,
                                        estimates.SVM.lasso,
                                        estimates.SVM.ridge,
                                        estimates.SVM.elastic)


estimateComparision$Variable <- row.names(estimateComparision)
estimateComparision <- estimateComparision[,c(ncol(estimateComparision),1:(ncol(estimateComparision)-1))]


estimateComparision <- estimateComparision %>% mutate(lr.lasso = removeWeakEstimates(lr.lasso),
                                                      lr.ridge = removeWeakEstimates(lr.ridge),
                                                      lr.elastic = removeWeakEstimates(lr.elastic),
                                                      nn = removeWeakEstimates(nn),
                                                      nn.lasso = removeWeakEstimates(nn.lasso),
                                                      nn.ridge = removeWeakEstimates(nn.ridge),
                                                      nn.elastic = removeWeakEstimates(nn.elastic),
                                                      SVM = removeWeakEstimates(SVM),
                                                      SVM.lasso = removeWeakEstimates(SVM.lasso),
                                                      SVM.ridge = removeWeakEstimates(SVM.ridge),
                                                      SVM.elastic = removeWeakEstimates(SVM.elastic))




write_xlsx(list('Model Comparision' = modelComparision,
                'Data Correlation' = cbind.data.frame(Var = names(BostonNormalised),
                                                      data.frame(cor(BostonNormalised))),
                'Estimate Comparision' = estimateComparision,
                'variableImportance' = variableImportance.SVM),
           'Boston_Results.xlsx')
