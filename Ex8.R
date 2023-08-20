###Support Vector Machines###
library(e1071) #used for svm
#Below: generating observations and checking whether the classes are lin. separable
set.seed(1)
x = matrix(rnorm(20*2), ncol = 2)
y = c(rep(-1, 10), rep (1, 10))
x[y == 1, ] = x[y == 1, ] + 1
plot(x, col = (3 - y)) #we see they're not separable
#Below: kernel = "linear" is used to fit a support vector clissifier
dat = data.frame(x = x, y = as.factor(y))
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 10, scale = FALSE)
#Above: scale = FALSE is used to not scale each feature to have mean zero or stndrd deviation one
plot(svmfit, dat) # x - support vectors; o - remaining observations
svmfit$index #to determine their identities
summary(svmfit)
#Trying a smaller value of the cost
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 0.1, scale = FALSE)
plot(svmfit, dat) # margin is wider => more support vectors in this situation
svmfit$index
#tune() - performs 10-fold CV on a set of models of interest
set.seed(1)
tune.out = tune(svm, y ~ ., data = dat, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 1000)))
summary(tune.out) #used to access the Cv errros for each of these models
#Below: accessing the best model obtained
bestmod = tune.out$best.model
summary(bestmod)
#Below: generating a test data set
xtest = matrix(rnorm(20 * 2), ncol = 2)
ytest = sample(c(-1, 1), 20, rep = TRUE)
xtest[ytest == 1, ] = xtest[ytest == 1, ] + 1
testdata = data.frame(x = xtest, y = as.factor(ytest))
#Below: predict() - used to predict the class labels of thee test observations
ypred = predict(bestmod, testdata)
table(predict = ypred, truth = testdata$y)
#Below: trying a cost value = 0.01
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 0.01, scale = FALSE)
ypred = predict(svmfit, testdata)
table(predict = ypred, truth = testdata$y) #it performed worse
#Below: using a separating hyperplane
##first we make data linearly separable
x[y == 1, ] = x[y == 1, ] + 0.5
plot(x, col = (y + 5)/2, pch = 19)
#Below: Fitting support vector class. and plotting the hyperplane
dat = data.frame(x = x, y = as.factor(y))
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 1e5)
summary(svmfit)
plot(svmfit, dat)
#Below: trying cost = 1
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 1)
summary(svmfit)
plot(svmfit, dat)

##Support Vector Machine##
set.seed(1)
x = matrix(rnorm(200*2), ncol = 2)
x[1:100, ] = x[1:100, ] + 2
x[101:150, ] = x[101:150, ] - 2
y = c(rep(1, 150), rep(2, 50))
dat = data.frame(x = x, y = as.factor(y))
plot(x, col = y)
#Below: randomly splitting into test and training; using kernel = "radial" because of the results above
##gamma - positive const
train = sample(200, 100)
svmfit = svm(y ~ ., data = dat[train, ], kernel = "radial", gamma = 1, cost = 1)
plot(svmfit, dat[train, ])
summary(svmfit) #gives information about SVM
#Below: increasing the value of cost to reduce the number of training errors
svmfit = svm(y ~ ., data = dat[train, ], kernel = "radial", gamma = 1, cost = 1e5)
plot(svmfit, dat[train, ])
#Below: tune() - used to select the best choice of gamma and cost
set.seed(1)
tune.out = tune(svm, y ~ ., data = dat[train, ], kernel = "radial", ranges = list(cost = c(0.1, 1, 10, 100, 1000), gamma = c(0.5, 1, 2, 3, 4)))
summary(tune.out) #shows that best result is cost = 1 and gamma = 0.5
table(true = dat[-train, "y"], pred = predict(tune.out$best.model, newdata = dat[-train, ]))

##Roc Curves##
library(ROCR) #needed for ROC curves
rocplot = function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf, ...)
}
#Below: decision.values = TRUE is used to obtain the fitted values for a given SVM
svmfit.opt = svm(y ~ ., data = dat[train, ], kernel = "radial", gamma = 2, cost = 1, decision.values = T)
fitted = attributes(predict(svmfit.opt, dat[train, ], decision.values = TRUE))$decision.values
#Below: plotting ROC
par(mfrow = c(1, 2))
rocplot(-fitted, dat[train, "y"], main = "Training Data")
#Below: improving accuracy by inreasing gamma
svmfit.flex = svm(y ~ ., data = dat[train, ], kernel = "radial", gamma = 50, cost = 1, decision.values = T)
fitted = attributes(predict(svmfit.flex, dat[train, ], decision.values = TRUE))$decision.values
rocplot(-fitted, dat[train, "y"], add = T, col = "red")
#Below: now plotting ROC on test data; gamma = 2 appears to give the most accurate results
fitted = attributes(predict(svmfit.opt, dat[-train, ], decision.values = TRUE))$decision.values
rocplot(-fitted, dat[-train, "y"], main = "Test Data")
fitted = attributes(predict(svmfit.flex, dat[-train, ], decision.values = TRUE))$decision.values
rocplot(-fitted, dat[-train, "y"], add = T, col = "red")

##Application to Gene Expression Data##
library(ISLR2)
names(Khan) #names are "xtrain" "xtest"  "ytrain" "ytest"
#Below: Examining the dimensions of the data
dim(Khan$xtrain)
dim(Khan$xtest)
length(Khan$ytrain)
length(Khan$ytest)
table(Khan$ytrain)
table(Khan$ytest)
#Below: using SV approach to predict cancer subtype using gene espression measurements
##Data set has a large # of features = > using kernel = "linear" because flexability is unnecessary
dat = data.frame(x = Khan$xtrain, y = as.factor(Khan$ytrain))
out = svm(y ~ ., data = dat, kernel = "linear", cost = 10)
summary(out)
table(out$fitted, dat$y)  #shows 0 errors
#Below: trying it on the test set
dat.te = data.frame(x = Khan$xtest, y = as.factor(Khan$ytest))
pred.te = predict(out, newdata = dat.te)
table(pred.te, dat.te$y) #shows 2 errors