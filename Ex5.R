###Subset Selection Methods###
##Best Subset Selection##
library(ISLR2)
View(Hitters) #opens a spreadsheet
names(Hitters)
dim(Hitters)
#Below: is.na() - used to identify the missing obseravtions; output - number of missing
sum(is.na(Hitters$Salary))
#Below: na.omit() - removes all rows with missing values
Hitters = na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))
#Below: regsubsets() - performs best subset selection (best quantified using RSS)
library(leaps) #this library needed to preform it
regfit.full = regsubsets(Salary ~ ., Hitters)
summary(regfit.full) # more * is better (I think)
regfit.full = regsubsets(Salary ~ ., data = Hitters, nvmax = 19) #we got 19 from the command above
reg.summary = summary(regfit.full) 
names(reg.summary) #summary() also Returns R^2, RSS, adjusted R^2, C_p, and BIC which we'll examine
reg.summary$rsq #checking R^2; values increase more variables it gets
#Below: plotting RSS, adjusted R^2, C_p, and BIC at once
##type = "l" tells R to connect the plotted points with lines
par(mfrow = c(2,2))
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
#Below: points() - puts point on an existing/generated plot
which.max(reg.summary$adjr2) #gives the location of the max point of a vector
points(11, reg.summary$adjr2[11], col = "red", cex = 2, pch = 20) #indicates the model with the largest adjusted R^2 statistic
#Below: doing the same for C_p, and BIC statistics
##which.min() - used to indicate the model with the smallest statistic
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
which.min(reg.summary$cp) #gives the location of the min point of a vector
points(10, reg.summary$cp[10], col = "red", cex = 2, pch = 20)
which.min(reg.summary$bic)
points(reg.summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
points(6, reg.summary$cp[6], col = "red", cex = 2, pch = 20)
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")
#Below: the lowest BIC is the six-variable model
coef(regfit.full, 6) #checking the coefficient estimates associated with this model
##Forward and Backward Stepwise Selection##
#Below: Forward
regfit.fwd = regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "forward")
summary(regfit.fwd)
#Below: backward
regfit.bwd = regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "backward")
summary(regfit.bwd)
coef(regfit.full, 7) #7-variable models is identified by fwd, bwd, and best sub. selection are different
coef(regfit.fwd, 7)
coef(regfit.bwd, 7)
##Choosing Among Models Using the Validation-Set Approach and Cross-Validation##
#to yield accurate estimates of the test error - only use the training observations
#Below: splitting the observations into training and test sets
set.seed(1)
train = sample(c(TRUE, FALSE), nrow(Hitters), replace = TRUE) #True - training set
test = (!train)
regfit.best = regsubsets(Salary ~ ., data = Hitters[train, ], nvmax = 19) #performing best sub. selection
#Above: Hitters[train, ] - gets only the training subset of data
test.mat = model.matrix(Salary ~ ., data = Hitters[test, ]) #Creating an "X" matrix from data
val.errors = rep(NA, 19)
for(i in 1:19){
  coefi = coef(regfit.best, id = i)
  pred = test.mat[, names(coefi)] %*% coefi
  val.errors[i] = mean((Hitters$Salary[test] - pred)^2)
}
#Above: extract coef. from regfit.best for the best model of that size; multpl. them into appropriate columns of the test model matrix
##to form the predictors; compute the test MSE
val.errors
which.min(val.errors)
coef(regfit.best, 7)
#Above: we find that best model contains 7 variables
#Below: writing our own predict method
predict.regsubets = function(object, newdata, id, ...){
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  xvars = names(coefi)
  mat[, xvars] %*% coefi
}
#Above: this function mimics the one above
#Below: performing best subset selection
regfit.best = regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
coef(regfit.best, 7)
#Below: using Cross-Validation method with 10 folds
k = 10
n = nrow(Hitters)
set.seed(1)
folds = sample(rep(1:k, length = n))
cv.errors = matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))
#Below: for loop that performs Cross-Validation:
#first, writing own predict.regsubsets (mentioned above)
#second, cross-validation itself
for(j in 1:k){
  best.fit <- regsubsets(Salary ~ ., data = Hitters[folds != j, ], nvmax = 19)
  for(i in 1:19){
    pred <- predict.regsubsets(best.fit, newdata = Hitters[folds == j, ], id = i)
    cv.errors[j, i] = mean((Hitters$Salary[folds == j] - pred)^2)
  }
}
#Below: apply() - average over the columns of this matrix; vector for which ith element is the
##Cross-Validation error fro the i-variable model
mean.cv.errors = apply(cv.errors, 2, mean)
mean.cv.errors
par(mfrow = c(1, 1))
plot(mean.cv.errors, type = "b") #from this wee see that 10-var. model is the best
reg.best = regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
coef(reg.best, 10)

###Ridge Regression And The Lasso###
#Below: making sure missing values are removed
x = model.matrix(Salary ~ ., Hitters)[, -1] #model.matrix() is also good because it automatically qualit. to quant
y = Hitters$Salary
##Ridge Regression##
library(glmnet) #needed for regression models, lasso, and more
grid = 10^seq(10, -2, length = 100) #lambda in range 10^10 to 10^-2; full range
ridge.mod = glmnet(x, y, alpha = 0, lambda = grid)
#Above: if you don't want glmnet() to set var. on the same scale, use standardize = FALSE
dim(coef(ridge.mod))
ridge.mod$lambda[50] #lambda = 11,498
coef(ridge.mod)[, 50]
sqrt(sum(coef(ridge.mod)[-1, 50]^2))

ridge.mod$lambda[60] #lambda = 705
coef(ridge.mod)[, 60]
sqrt(sum(coef(ridge.mod)[-1, 60]^2))
#Below: using predict() to obtain the ridge regression coef. for a new val. of lambda, 50
predict(ridge.mod, s = 50, type = "coefficients")[1:20, ]
#Below: splitting it into training and test sets
set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
y.test = y[test]
#Below: fit RRM on the training set, evaluate MSE, lambda = 4
ridge.mod = glmnet(x[train, ], y[train], alpha = 0, lambda = grid, thresh = 1e-12)
ridge.pred = predict(ridge.mod, s = 4, newx = x[test, ])
mean((ridge.pred - y.test)^2)
mean((mean(y[train])- y.test)^2)
#Below: fitting RRM with a very large lambda
ridge.pred = predict(ridge.mod, s = 1e10, newx = x[test, ])
mean((ridge.pred - y.test)^2)

ridge.pred = predict(ridge.mod, s = 0, newx = x[test, ], exact = T, x = x[train, ], y = y[train])
mean((ridge.pred - y.test)^2)
lm(y ~ x, subset = train) #lm() - used to fit a (unpenalized) LSM; gives use useful outputs such as stnd errors and p-values
predict(ridge.mod, s = 0, exact = T, type = "coefficients", x = x[train, ], y = y[train])[1:20, ]
#Below: cv.glmnet() - using Cross-Validation for lambda tuning; by default this function performs 10-fold CV
set.seed(1)
cv.out = cv.glmnet(x[train, ], y[train], alpha = 0)
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam #gives value of lambda
#Below: test MSE associated with the value of lambda
ridge.pred = predict(ridge.mod, s = bestlam, newx = x[test, ])
mean((ridge.pred - y.test)^2)
#Below: refitting RRM on the full data set using lambda chosen by Cross-Validation
out = glmnet(x, y, alpha = 0)
predict(out, type = "coefficients", s = bestlam)[1:20, ]
##The Lasso##
#Below: using glmnet() for the lasso but this time alpha = 1
lasso.mod = glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)
#Below: performing Cross-Validation and computing associated test error
set.seed(1)
cv.out = cv.glmmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)
bestlam = cv.out$lambda.min
lasso.pred = predict(lasso.mod, s = bestlam, newx = x[test, ])
mean((lasso.pred - y.test)^2)

out = glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef = predict(out, type = "coefficients", s = bestlam)[1:20, ]
lasso.coef

###PCR and PLS Regression###
library(pls) #needed for Principal Component Regression
set.seed(2)
pcr.fit = pcr(Salary ~ ., data = Hitters, scale = TRUE, validation = "CV")
#Above: scale = TRUE - standardizing each predictor; validation = "CV" - computes ten-fold 
##Cross-Validation error for each possible value of M
#Below: resulting fit can be examined by using summary()
summary(pcr.fit)
#Above: it gives Root Mean Squared Error; to obtain the usual MSE => RMSE^2 = MSE
#Below: validationplot() - used to plot CV scores; val.type = "MSEP" will plot CV MSE 
validationplot(pcr.fit, val.type = "MSEP")
#Below: performing PCR on the training data
set.seed(1)
pcr.fit = pcr(Salary ~ ., data = Hitters, subset = train, scale = TRUE, validation = "CV")
validationplot(pcr.fit, val.type = "MSEP")
#Below: computing MSE; lowest CV error occurs when M = 5 (from the plot)
pcr.pred = predict(pcr.fit, x[test, ], ncomp = 5)
mean((pcr.pred - y.test)^2)
#Below: fitting PCR on the full data set; M = 5
pcr.fit = pcr(y ~ x, scale = TRUE, ncomp = 5)
summary(pcr.fit)
##Partial Least Squares##
set.seed(1)
pls.fit = plsr(Salary ~ ., data = Hitters, subset = train, scale = TRUE, validation = "CV")
summary(pls.fit)
validationplot(pls.fit, val.type = "MSEP")
#Below: M = 1 which was found from the plot above
pls.pred = predict(pls.fit, x[test, ], ncomp = 1)
mean((pls.pred - y.test)^2) #gives test MSE #; higher is worse
pls.fit = plsr(Salary ~ ., data = Hitters, scale = TRUE, ncomp = 1)
summary(pls.fit)
