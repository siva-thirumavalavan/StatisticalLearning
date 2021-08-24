rm(list = ls())
options(warn = -1)


#### libraries ####
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
  # paste('The significant variables are ',paste(significantVariables,collapse = ', '))
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
  return(as.numeric(gsub('0$',NA,as.character(round(iterCol,3)))))
}


#### Analysis of Census Data ####
#load the census dataset
censusData <- read.csv('Data/us_census_sampled.csv',stringsAsFactors = F)
summary(censusData)

#create a directory to save the  results
subDirectory <- 'US Census Results'
dir.create(subDirectory)
setwd(paste(workingDirectory,subDirectory,sep = '\\'))

#we are normalizing our x variables
censusNormalised <- censusData[,-c(2,3)]
names(censusNormalised)
censusNormalised[,-which(names(censusNormalised)=='IncomePerCap')] <- data.frame(sapply(censusNormalised[,-which(names(censusNormalised)=='IncomePerCap')] , minMaxScale))


#checking distribution of each column
png('US Census - Distribution.png',width = 1280, height = 720)
censusNormalised %>% gather() %>% ggplot(aes(value))+facet_wrap(~key,scales = 'free')+geom_density() 
dev.off()

png('US Census - Correlation.png',width = 1280, height = 720)
corrplot(cor(censusNormalised))
dev.off()

#converting the dataset to long format and plotting each column against per capita crime rate
CensusLongFormat <- gather(censusNormalised,key = 'var',value = 'value' ,-IncomePerCap)
png('US Census - Y vs X_1.png',width = 1280, height = 720, res = 90)
ggplot(CensusLongFormat[CensusLongFormat$var %in% names(censusNormalised)[1:9],],aes(x=value,y=IncomePerCap, col = var)) + geom_point() + geom_smooth(method = 'lm') + facet_grid(var ~. )
dev.off()

png('US Census - Y vs X_2.png',width = 1280, height = 720, res = 90)
ggplot(CensusLongFormat[CensusLongFormat$var %in% names(censusNormalised)[10:18],],aes(x=value,y=IncomePerCap, col = var)) + geom_point() + geom_smooth(method = 'lm') + facet_grid(var ~. )
dev.off()

png('US Census - Y vs X_3.png',width = 1280, height = 720, res = 90)
ggplot(CensusLongFormat[CensusLongFormat$var %in% names(censusNormalised)[19:27],],aes(x=value,y=IncomePerCap, col = var)) + geom_point() + geom_smooth(method = 'lm') + facet_grid(var ~. )
dev.off()

#### Data Split ####
set.seed(1234)
trainIndex <- createDataPartition(censusNormalised$IncomePerCap,p=0.8,list=FALSE) 
# we use cross validation techniques to split train and validation datasets
census.train <- censusNormalised[trainIndex,] 
census.test <- censusNormalised[-trainIndex,] 
ctrlParameters  <- trainControl(method  = "cv",
                                number  = 10,
                                savePredictions = T)
Y_index <- which(names(censusNormalised)=='IncomePerCap')

## 1. Multiple Linear Regression with all the variables ####
tic <- Sys.time()
model.lr <- train(IncomePerCap ~ .,
                  data = census.train, 
                  method = "lm",
                  metric ='RMSE',
                  trControl = ctrlParameters)
toc <- Sys.time()
summary(model.lr)
predictions.lr <- predict(model.lr,census.test)
estimates.lr <- data.frame(lr = coef(model.lr$finalModel, model.lr$bestTune$intercept)[-1])
results.lr <- modelParameters(predictions.lr,'lr',census.test$IncomePerCap)
results.lr$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.lr$Accuracy <- NA
results.lr$alpha <- NA
results.lr$lambda <- NA
results.lr

#we identify the significant variables from the summary and run another iteration with only those variables
model.lr.2 <- train(formula(paste('IncomePerCap ~',getSignificantEquation(model.lr,0.05))),
                    data = census.train, 
                    method = "lm",
                    metric ='RMSE',
                    trControl = ctrlParameters) 
summary(model.lr.2)
predictions.lr.2 <- predict(model.lr.2,census.test)
modelParameters(predictions.lr.2,'lr',census.test$IncomePerCap)

# 1.1 Linear Regression + Ridge ####
tic <- Sys.time()
# we tune for the hyper-parameters that gives the least RMSE value
model.lr.ridge <- train(IncomePerCap ~.,
                        data = census.train, 
                        method = "glmnet",
                        trControl = ctrlParameters,
                        tuneGrid = expand.grid(alpha=0,lambda=10^seq(5,-5,length=100)))
toc <- Sys.time()

png('US Census - LM Ridge.png',width = 1280, height = 720)
plot(model.lr.ridge)
dev.off()
estimates.lr.ridge <- data.frame(lr.ridge = coef(model.lr.ridge$finalModel, model.lr.ridge$bestTune$lambda)[-1,1])
predictions.lr.ridge <- predict(model.lr.ridge,as.matrix(census.test[,-which(names(census.test)=='IncomePerCap')]))
results.lr.ridge <-modelParameters(predictions.lr.ridge,'lr.ridge',census.test$IncomePerCap)
results.lr.ridge$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.lr.ridge$Accuracy <- NA
results.lr.ridge$alpha <- 0
results.lr.ridge$lambda <- round(model.lr.ridge$bestTune$lambda,3)
results.lr.ridge

# 1.2 Linear Regression + Lasso ####
tic <- Sys.time()
# we tune for the hyperprameters that gives the least RMSE value
model.lr.lasso <- train(IncomePerCap ~.,
                        data = census.train, 
                        method = "glmnet",
                        trControl = ctrlParameters,
                        tuneGrid = expand.grid(alpha=1,lambda=10^seq(5,-5,length=100)))

toc <- Sys.time()

png('US Census - LM Lasso.png',width = 1280, height = 720)
plot(model.lr.lasso)
dev.off()

estimates.lr.lasso <- data.frame(lr.lasso = coef(model.lr.lasso$finalModel, model.lr.lasso$bestTune$lambda)[-1,1])
predictions.lr.lasso <- predict(model.lr.lasso,as.matrix(census.test[,-which(names(census.test)=='IncomePerCap')]))
results.lr.lasso <-modelParameters(predictions.lr.lasso,'lr.lasso',census.test$IncomePerCap)
results.lr.lasso$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.lr.lasso$Accuracy <- NA
results.lr.lasso$alpha <- 1
results.lr.lasso$lambda <- round(model.lr.lasso$bestTune$lambda,3)
results.lr.lasso

# 1.3 Linear Regression + elastic ####
tic <- Sys.time()
# we tune for the hyperprameters that gives the least RMSE value
model.lr.elastic <- train(IncomePerCap ~.,
                          data = census.train, 
                          method = "glmnet",
                          trControl = ctrlParameters,
                          tuneGrid = expand.grid(alpha=c(0:10)/10,lambda=10^seq(5,-5,length=100)))
toc <- Sys.time()
png('US Census - LM elastic.png',width = 1280, height = 720)
plot(model.lr.elastic)
dev.off()

estimates.lr.elastic <- data.frame(lr.elastic = coef(model.lr.elastic$finalModel, model.lr.elastic$bestTune$lambda)[-1,1])
predictions.lr.elastic <- predict(model.lr.elastic,as.matrix(census.test[,-which(names(census.test)=='IncomePerCap')]))
results.lr.elastic <-modelParameters(predictions.lr.elastic,'lr.elastic',census.test$IncomePerCap)
results.lr.elastic$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.lr.elastic$Accuracy <- NA
results.lr.elastic$alpha <- round(model.lr.elastic$bestTune$alpha,3)
results.lr.elastic$lambda <- round(model.lr.elastic$bestTune$lambda,3)
results.lr.elastic

## 2. Neural Network ####
# a perceptron with single hidden layer emulates a simple linear regression. This will enable us to treat the weights as estimates
tic <- Sys.time()
model.nn <- neuralnetwork(as.matrix(census.train[,-which(names(census.test)=='IncomePerCap')]),
                          as.matrix(census.train[,'IncomePerCap']),
                          hidden.layers =1,
                          optim.type = 'adam',
                          n.epochs = 100,
                          regression = T,
                          learn.rates = 0.01, 
                          loss.type = 'squared', 
                          verbose = F,
                          activ.functions = 'relu',
                          L1 = 0,
                          L2 = 0)
toc <- Sys.time()
estimates.nn <- data.frame(nn = c(model.nn$Rcpp_ANN$getParams()$weights[[1]]))
predictions.nn <- predict(model.nn,as.matrix(census.test[,-which(names(census.test)=='IncomePerCap')]))
results.nn <- modelParameters(predictions.nn$predictions,'nn',census.test$IncomePerCap)
results.nn$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.nn$Accuracy <- NA
results.nn$alpha <- NA
results.nn$lambda <- NA
results.nn

# 2.1 Neural Network + lasso ####
tic <- Sys.time()
model.nn.lasso <- neuralnetwork(as.matrix(census.train[,-which(names(census.test)=='IncomePerCap')]),
                                as.matrix(census.train[,'IncomePerCap']),
                                hidden.layers =1,
                                optim.type = 'adam',
                                n.epochs = 100,
                                regression = T,
                                learn.rates = 0.01, 
                                loss.type = 'squared', 
                                verbose = F,
                                activ.functions = 'relu',
                                L1 = 1,
                                L2 = 0)

toc <- Sys.time()
estimates.nn.lasso <- data.frame(nn.lasso = c(model.nn.lasso$Rcpp_ANN$getParams()$weights[[1]]))
predictions.nn.lasso <- predict(model.nn.lasso,as.matrix(census.test[,-which(names(census.test)=='IncomePerCap')]))
results.nn.lasso <- modelParameters(predictions.nn.lasso$predictions,'nn.lasso',census.test$IncomePerCap)
results.nn.lasso$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.nn.lasso$Accuracy <- NA
results.nn.lasso$alpha <- 1
results.nn.lasso$lambda <- NA
results.nn.lasso
# 2.2 Neural Network + ridge ####
tic <- Sys.time()
model.nn.ridge <- neuralnetwork(as.matrix(census.train[,-which(names(census.test)=='IncomePerCap')]),
                                as.matrix(census.train[,'IncomePerCap']),
                                hidden.layers =1,
                                optim.type = 'adam',
                                n.epochs = 100,
                                regression = T,
                                learn.rates = 0.01, 
                                loss.type = 'squared', 
                                verbose = F,
                                activ.functions = 'relu',
                                L1 = 0,
                                L2 = 1)

toc <- Sys.time()
estimates.nn.ridge <- data.frame(nn.ridge = c(model.nn.ridge$Rcpp_ANN$getParams()$weights[[1]]))
predictions.nn.ridge <- predict(model.nn.ridge,as.matrix(census.test[,-which(names(census.test)=='IncomePerCap')]))
results.nn.ridge <- modelParameters(predictions.nn.ridge$predictions,'nn.ridge',census.test$IncomePerCap)
results.nn.ridge$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.nn.ridge$Accuracy <- NA
results.nn.ridge$alpha <- 0
results.nn.ridge$lambda <- NA
results.nn.ridge
# 2.3 Neural Network + elastic ####
tic <- Sys.time()
model.nn.elastic <- neuralnetwork(as.matrix(census.train[,-which(names(census.test)=='IncomePerCap')]),
                                  as.matrix(census.train[,'IncomePerCap']),
                                  hidden.layers =1,
                                  optim.type = 'adam',
                                  n.epochs = 100,
                                  regression = T,
                                  learn.rates = 0.01, 
                                  loss.type = 'squared', 
                                  verbose = F,
                                  activ.functions = 'relu',
                                  L1 = 1,
                                  L2 = 1)

toc <- Sys.time()
estimates.nn.elastic <-  data.frame(nn.elastic = c(model.nn.elastic$Rcpp_ANN$getParams()$weights[[1]]))

predictions.nn.elastic <- predict(model.nn.elastic,as.matrix(census.test[,-which(names(census.test)=='IncomePerCap')]))

results.nn.elastic <- modelParameters(predictions.nn.elastic$predictions,'nn.elastic',census.test$IncomePerCap)

results.nn.elastic$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.nn.elastic$Accuracy <- NA
results.nn.elastic$alpha <- NA
results.nn.elastic$lambda <- NA
results.nn.elastic

## 3. SVM ####

tic <- Sys.time()
model.SVM <- tune.svm(IncomePerCap~.,
                      data = census.train,
                      tunecontrol = tune.control(cross = 10),
                      kernel = 'linear')

toc <- Sys.time()
summary(model.SVM$best.model)
predictions.SVM <- predict(model.SVM$best.model,census.test)
estimates.SVM <- data.frame(SVM = coef(model.SVM$best.model)[-1])
results.SVM <- modelParameters(predictions.SVM,'SVM',census.test$IncomePerCap)
results.SVM$runTime <- round(difftime(toc,tic,units = 'mins'),3)
results.SVM$Accuracy <- NA
results.SVM$alpha <- NA
results.SVM$lambda <- NA
results.SVM



svmPredictor <- function(model,data){
  return(predict(model,data))
}

variableImportance.SVM <- data.frame(Var = names(census.train[,-Y_index]),
                                     Importance = round(Importance(model.SVM$best.model,census.train, PRED = svmPredictor)$imp[-1],2)
)


#### 3.1 SVM + Ridge ####
# we convert our target variable into a binary class to fit the data in SVM with regularization

censusNormalised.SVM <- censusNormalised
censusNormalised.SVM$IncomePerCap <- 'low'
censusNormalised.SVM$IncomePerCap[censusNormalised$IncomePerCap>=quantile(censusNormalised$IncomePerCap,0.5,names = F)] <- 'high'
census.train.SVM <- censusNormalised.SVM[trainIndex,] 
census.test.SVM <- censusNormalised.SVM[-trainIndex,] 

tic <- Sys.time()
crossValidaton.ridge <- cv.sparseSVM(as.matrix(census.train.SVM[,-which(names(census.train.SVM)=='IncomePerCap')]),
                                     census.train.SVM[,which(names(census.train.SVM)=='IncomePerCap')],
                                     nfolds = 10,
                                     eval.metric = 'me')

model.SVM.ridge <- sparseSVM(as.matrix(census.train.SVM[,-which(names(census.train.SVM)=='IncomePerCap')]),
                             census.train.SVM[,which(names(census.train.SVM)=='IncomePerCap')],
                             alpha = 0,
                             lambda = crossValidaton.ridge$lambda.min)
toc <- Sys.time()
#we use cross-validation to find the hyper parameters that gives  us the least misclassification error
print(model.SVM.ridge)


estimates.SVM.ridge <- data.frame(SVM.ridge = model.SVM.ridge$weights[-1])
predictions.SVM.ridge<- predict(model.SVM.ridge,X = as.matrix(census.test.SVM[,-which(names(census.train.SVM)=='IncomePerCap')])) 
confMatrix.SVM.ridge  <- table(Predicted=predictions.SVM.ridge,Actual=census.test.SVM$IncomePerCap)
results.SVM.ridge <- data.frame(model= 'SVM.ridge',
                                RMSE = NA,
                                Rsquare = NA,
                                runTime = round(as.numeric(difftime(toc,tic,units = 'mins')),3),
                                Accuracy = sum(diag(confMatrix.SVM.ridge ))/nrow(census.test.SVM),
                                alpha = 0,
                                lambda = round(crossValidaton.ridge$lambda.min,3))
results.SVM.ridge


#### 3.2 SVM + Lasso ####
tic <- Sys.time()
#we use cross-validation to find the hyper parameters that gives  us the least misclassification error
crossValidaton.lasso <- cv.sparseSVM(as.matrix(census.train.SVM[,-which(names(census.train.SVM)=='IncomePerCap')]),
                                     census.train.SVM[,which(names(census.train.SVM)=='IncomePerCap')],
                                     nfolds = 10,
                                     eval.metric = 'me')

model.SVM.lasso <- sparseSVM(as.matrix(census.train.SVM[,-which(names(census.train.SVM)=='IncomePerCap')]),
                             census.train.SVM[,which(names(census.train.SVM)=='IncomePerCap')],
                             alpha = 1,
                             lambda = crossValidaton.lasso$lambda.min)
toc <- Sys.time()
print(model.SVM.lasso)

estimates.SVM.lasso <- data.frame(SVM.lasso = model.SVM.lasso$weights[-1])
predictions.SVM.lasso<- predict(model.SVM.lasso,X = as.matrix(census.test.SVM[,-which(names(census.train.SVM)=='IncomePerCap')])) 
confMatrix.SVM.lasso  <- table(Predicted=predictions.SVM.lasso,Actual=census.test.SVM$IncomePerCap)
results.SVM.lasso <- data.frame(model= 'SVM.lasso',
                                RMSE = NA,
                                Rsquare = NA,
                                runTime = round(as.numeric(difftime(toc,tic,units = 'mins')),3),
                                Accuracy = sum(diag(confMatrix.SVM.lasso ))/nrow(census.test.SVM),
                                alpha = 1,
                                lambda = round(crossValidaton.lasso$lambda.min,3))

results.SVM.lasso

#### 3.3 SVM + elastic ####
tic <- Sys.time()
#we use cross-validation to find the hyper parameters that gives  us the least misclassification error
crossValidaton.elastic <- cv.sparseSVM(as.matrix(census.train.SVM[,-which(names(census.train.SVM)=='IncomePerCap')]),
                                       census.train.SVM[,which(names(census.train.SVM)=='IncomePerCap')],
                                       nfolds = 10,
                                       eval.metric = 'me')
estimates <- data.frame(Varibale = names(census.train.SVM[,-which(names(census.train.SVM)=='IncomePerCap')]))
IterResults <- data.frame(alpha = NA, Accuracy = NA)

#since cv.sparseSVM() doesn't support the tuning of alpha, we ru our model for different alpha values and select the one that gives the least misclassification error / max accuracy
for (alphaIter in seq(0,1,length=10)) {
  alphaIter <- round(alphaIter,1)
  model.SVM.elastic <- sparseSVM(as.matrix(census.train.SVM[,-which(names(census.train.SVM)=='IncomePerCap')]),
                                 census.train.SVM[,which(names(census.train.SVM)=='IncomePerCap')],
                                 alpha = alphaIter,
                                 lambda = crossValidaton.elastic$lambda.min)
  
  predictions<- predict(model.SVM.elastic,X = as.matrix(census.test.SVM[,-which(names(census.train.SVM)=='IncomePerCap')])) 
  confMatrix <- table(Predicted=predictions,Actual=census.test.SVM$IncomePerCap)
  IterResults <- rbind.data.frame(IterResults,
                                  data.frame(alpha = alphaIter,
                                             Accuracy = sum(diag(confMatrix))/nrow(census.test.SVM)))
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
                                  lambda = round(crossValidaton.elastic$lambda.min,3))

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
                'Data Correlation' = cbind.data.frame(Var = names(censusNormalised),
                                                      data.frame(cor(censusNormalised))),
                'Estimate Comparision' = estimateComparision,
                'variableImportance' = variableImportance.SVM),
           'US Census Results.xlsx')

