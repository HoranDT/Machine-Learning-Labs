###Libraries###
#installing/using libraries
library(MASS)
library(ISLR2)

###Simple Linear Regression###
#loading data;to find more information type ?Boston
head(Boston)
#fitting a simple linear regression where medv is a response and lstat is a predictor
#basic notation: lm(x~y, data)
#lm.fit = lm(medv~lstat) - gives an error; missing "data" to load from
#loading data
lm.fit = lm(medv~lstat, data = Boston)
attach(Boston)
lm.fit = lm(medv~lstat)
#to get some basic information about the model
lm.fit
#for detailed information
summary(lm.fit)
#for additional information
names(lm.fit)        
#to acces information
coef(lm.fit)
#to obtain confidence interval
confint(lm.fit)
#to produce confidence intervals and prediction intervals for the prediction of medv for a given value of lstat
predict(lm.fit, data.frame(lstat = (c(5, 10, 15))), interval = "confidence")
predict(lm.fit, data.frame(lstat = (c(5, 10, 15))), interval = "prediction")
#more here: prediction 95% confidence is interval (24.47, 25.63) or 2nd row lwr+upr. Same rules for prediction
#plotting medv and lstate along with the LSR (least square regression)
plot(lstat, medv)
#abline(lm.fit) #this draws a line but it doesn't fit in this situation
#trying different settings of the function abline()
abline(lm.fit, lwd = 3) #increases the width of the function by a factor of 3
abline(lm.fit, lwd = 3, col = "red")
plot(lstat, medv, col = "red")
plot(lstat, medv, pch = 20) #pch - plot character
plot(lstat, medv, pch = "+")
plot(1:20, 1:20, pch = 1:20)
#to get all four plots together
par(mfrow = c(2, 2)) #divides plotting region into 2x2 grid of panels
plot(lm.fit)
#computing residuals
plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit)) #returns students residuals
#computing leverage statistics for any number of predictors
plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit))

###Multiple Linear Regressions###
#fitting multiple linear regression model using least squares; lm(y~x1+x2+x3)
lm.fit = lm(medv~lstat + age, data = Boston)
summary(lm.fit)
#this can be used instead
lm.fit = lm(medv~., data = Boston)
summary(lm.fit)
## ?summary.lm shows what's available
## summary(lm.fit)$r.sq gives us the R^(2)
## summary(lm.fit)$sigma gives us the RSE
#computing variance inflation factors
library(car)
vif(lm.fit)
#performing regression using all variables but one
lm.fit1 = lm(medv~., -age, data = Boston)
summary(lm.fit1)

###Interaction Terms###
#to do so use lm()
#lstat:black syntax tells R to include an interaction term between them
#lstate*age simultaneously includes lstate,age; the interaction term lstat x age as a predictor
#it's a shorthand for lstat+age+lstat:age
summary(lm(medv~lstat*age, data = Boston))

###Non-Linear Interaction Terms###
#lm()  can accommodate non-linear transform. of predictors
#EX: from X to X^(2) use I(X^2)
#performing regression
lm.fit2 = lm(medv~lstat + I(lstat^2), data = Boston) #added data = Boston myself because of an error
summary(lm.fit2)
lm.fit2 = lm(medv)
#checking the extent to which the quadratic fit is superior to the linear fit
lm.fit = lm(medv ~ lstat, data = Boston) #added data myself
anova(lm.fit, lm.fit2) #runs a hypothesis test comparing two models
#to check the terms in graphs
par(mfrow = c(2,2))
plot(lm.fit2)
#creating a polynomial function
lm.fit5 = lm(medv ~ poly(lstat, 5), data = Boston)
summary(lm.fit5)
##for raw polynomials, use argument raw = TRUE
#trying log transformation
summary(lm(medv ~ log(rm), data = Boston))

###Qualitative Predictors###
head(Carseats)
lm.fit = lm(Sales ~ . + Income:Advertising + Price:Age, data = Carseats)
summary(lm.fit)
#returning the coding that R uses for dummy variables
attach(Carseats)
contrasts(ShelveLoc) #specifically this function does it
##?contrasts to learn about other contrasts and how to set them

###Writing Function###
#how to write own functions:
#when you write in console, if you hit ENTER after {, R will automatically add + on the next line
LoadLibraries = function(){
+ library(ISLR2)
+ library(MASS)
+ print("The libraries have been loaded.")
+}
LoadLibraries #this will give information about the function
LoadLibraries() #loading libraries to be used
