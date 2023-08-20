###The Validation Set Approach###
library(ISLR2)
set.seed(1) #this is important to to set a random seed to be able to reproduce the result later
train = sample(392, 196) # splitting the set of observations into two halves
#Above: 392 - overall amount, 196 - random subset of 196 observations
#Below: fitting a linear regression using only the observations from the training set
lm.fit = lm(mpg ~ horsepower, data = Auto, subset = train)
attach(Auto)
#Below: predict() - estimates the response for all observations; mean() - calculate MSE of the observations in the validation set
mean((mpg - predict(lm.fit, Auto))[-train]^2)
#Below: poly() - used to estimate the test error for the quadratic and cubic regression
lm.fit2 = lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train) #quadratic
mean((mpg - predict(lm.fit2, Auto))[-train]^2)
lm.fit3 = lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train) #cubic
mean((mpg - predict(lm.fit3, Auto))[-train]^2)
#Below: choosing a different seed
set.seed(2)
train = sample(392, 196)
lm.fit = lm(mpg ~ horsepower, data = Auto, subset = train)
mean((mpg - predict(lm.fit, Auto))[-train]^2)
lm.fit2 = lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train) 
mean((mpg - predict(lm.fit2, Auto))[-train]^2)
lm.fit3 = lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train) 
mean((mpg - predict(lm.fit3, Auto))[-train]^2)
#Above: from comparing results from seed(1) and seed(2) we can see that results are consistent
##and we can tell that using a quadratic function performs better than linear function
##and little evidence in favor of cubic function

###Leave-One-Out Cross-Validation###
glm.fit = glm(mpg ~ horsepower, data = Auto)
coef(glm.fit)
lm.fit = lm(mpg ~ horsepower, data = Auto)
coef(lm.fit)
#Above: this shows that glm() function can act like lm() if not passed in the family argument
library(boot) #cv.glm() is part of this library
glm.fit = glm(mpg ~ horsepower, data = Auto)
cv.err = cv.glm(Auto, glm.fit)
cv.err$delta
#Below: repeating the procedure for complex poly. using a for loop
cv.error = rep(0,10)
for (i in 1:10){
+ glm.fit = glm(mpg ~ poly(horsepower, i), data = Auto)
+ cv.error[i] = cv.glm(Auto, glm.fit)$delta[1]
+ }
cv.error
plot(cv.error) #not part of the lab; did it out of curiosity 

###k-Fold Cross-Validation###
set.seed(17)
cv.error.10 = rep(0, 10)
for (i in 1:10){
+ glm.fit = glm(mpg ~ poly(horsepower, i), data = Auto)
+ cv.error.10[i] = cv.glm(Auto, glm.fit, K = 10)$delta[1]
+ }
cv.error.10
#Somewhat above: (using cv.err$delta)First delta value - k-fold Cv estimate
##Second delta value - bias-corrected version

###The Bootstrap###
#Below: alpha.fn() - takes as input (X, Y) data + vector indication which observations should be used to estimate alpha
alpha.fn = function(data, index){
+ X = data$X[index]
+ Y = data$Y[index]
+ (var(Y) - cov(X, Y))/(var(X) + var(Y) - 2*cov(X, Y))
}
alpha.fn(Portfolio, 1:100) #returns an estimate for alpha
set.seed(7)
alpha.fn(Portfolio, sample(100, 100, replace = T))
#Below: boot() - produces bootstrap estimates for alpha and computes the std. deviation
boot(Portfolio, alpha.fn, R = 1000)
#Below: boot.fn() - takes in data set as well as set of indices for the observations
## and returns the intercept and slope estimates for linear regression
boot.fn = function(data, index)
+ coef(lm(mpg ~ horsepower, data = data, subset = index))
boot.fn(Auto, 1:392)
#Below: using boot.fn() to create estimates for the intercept and slope by randomly sampling from among
##the observations with replacement
set.seed(1)
boot.fn(Auto, sample(392, 392, replace = T)) #do it multiple times
boot(Auto, boot.fn, 1000) #computing standard errors of 1000 bootstrap estimates
#t1*  std.error =>SE(Betta.hat0) ( t2* is Betta.hat1 respectively)
#Below: another method of calculating those using summary()
summary(lm(mpg ~ horsepower, data = Auto))$coef
#Above: results between bootstrap and summary are slightly different
##this is because summary depends on noise variance (sigma^2)
##and standard formula assumes that x_i is fixed. 
##Bootstrap doesn't rely on these assumption => likely more accurate
#Below: Calculatingbootstrap std. error est. and std. lr estimates that result from fitting the qm to the data
boot.fn = function(data, index)
+ coef(lm(mpg ~ horsepower + I(horsepower^2), data = data, subset = index))
set.seed(1)
boot(Auto, boot.fn, 1000)
summary(lm(mpg ~ horsepower + I(horsepower^2), data = Auto))$coef