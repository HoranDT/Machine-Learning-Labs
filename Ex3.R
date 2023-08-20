###Stock market data###
library(ISLR2)
names(Smarket)
dim(Smarket) #gives dimensions
summary(Smarket)
pairs(Smarket)#returns a plot matrix
cor(Smarket) #creates a matrix of the pairwise correalations among the predictors in a data set
cor(Smarket[, -9]) #since Directions is qualitative, we use this command
attach(Smarket)
plot(Volume)

###Logistic Regression###
#glm() cna be used to fit many types of generalized linear models (GLM)
glm.fits = glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, data = Smarket, family = binomial)
summary(glm.fits)
coef(glm.fits) #to access just the coefficients
summary(glm.fits)$coef #to access particular aspects
summary(glm.fits)$coef[, 4] #to access particular aspects
glm.probs = predict(glm.fits, type = "response") #Probability that the market goes up; P(Y = 1|X)
glm.probs[1:10] #in range 1-10
#To get predictions for Up/Down on a particular day, convert predicted probabilities into class labels Up or Down
glm.pred = rep("Down", 1250) #creates a vector of 1,250 Down elements
glm.pred[glm.probs > .5] = "Up" #if > 0.5 then it's Up
table(glm.pred, Direction) #generates a confusion table
(507 + 145)/1250 #calculating mean
mean(glm.pred == Direction) #calculates mean for you
train = (Year < 2005) #train is a vector of 1250 elements; before 2005 TRue and False for during
Smarket.2005 = Smarket[!train, ]#this will choose only for train is False
dim(Smarket.2005)
Direction.2005 = Direction[!train]
#training - data before 2005; testing - in 2005
glm.fits = glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, data = Smarket, family = binomial, subset = train)
glm.probs = predict(glm.fits, Smarket.2005, type = "response")
glm.pred = rep("Down", 252)
glm.pred[glm.probs > .5] = "Up"
table(glm.pred, Direction.2005)
mean(glm.pred == Direction.2005)
mean(glm.pred != Direction.2005)#test set error rate
#Below we're refitting the logistic regression using Lag1 and Lag2
glm.fits = glm(Direction ~ Lag1 + Lag2, data = Smarket, family = binomial, subset = train)
glm.probs = predict(glm.fits, Smarket.2005, type = "response")
glm.pred = rep("Down", 252)
glm.pred[glm.probs > .5] = "Up"
table(glm.pred, Direction.2005)
mean(glm.pred == Direction.2005) #checking the accuracy of predictions of the daily movements
106/(106+76)#checking the accuracy rate of predictions of an increase in market
#Below predicting the returns associated with particular values of Lag1 and Lag2
#to be specific, we're predicting Direction using information (values) from two different days
#Lag1 = (1.2, 1.5) and Lag2 = (1.1, -0.8); LagN = (Day1, Day2)
predict(glm.fits, newdata = data.frame(Lag1 = c(1.2, 1.5), Lag2 = c(1.1, -0.8)), typre = "response")

###Linear Discriminant Analysis###
library(MASS)
lda.fit = lda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
lda.fit
plot(lda.fit)
lda.pred = predict(lda.fit, Smarket.2005)
names(lda.pred)
lda.class = lda.pred$class
table(lda.class, Direction.2005)
mean(lda.class == Direction.2005)
sum(lda.pred$posterior[, 1] >= .5) #applying 50% threshold
sum(lda.pred$posterior[, 1] < .5)#applying 50% threshold
lda.pred$posterior[1:20, 1] #predictions based on the threshold (num values)
lda.class[1:20] #predictions based on the threshold (up/down)
sum(lda.pred$posterior[, 1] > .9) #changing the threshold to 90% just to test

###Quadratic Discriminant Analysis###
qda.fit = qda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
qda.fit
qda.class = predict(qda.fit, Smarket.2005)$class
table(qda.class, Direction.2005)
mean(qda.class == Direction.2005)

###Naive Bayes###
library(e1071) #needed for naive bayes
nb.fit = naiveBayes(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
nb.fit #first value in the matrix of each lag - mean, second value - standard deviation
mean(Lag1[train][Direction[train] == "Down"]) #finding mean of Lag1 "Down" manually
sd(Lag1[train][Direction[train] == "Down"]) #finding standard deviation of Lag1 "Down" manually
nb.class = predict(nb.fit, Smarket.2005)
table(nb.class, Direction.2005)
mean(nb.class == Direction.2005)
#Below Generating estimate of the probability that each observation belongs to a particular class
nb.preds = predict(nb.fit, Smarket.2005, type = "raw")
nb.preds[1:5, ]

###K-nearest Neighbors###
library(class) #needed for k-nearest
#K-nearest requires 4 inputs to perform
train.X = cbind(Lag1, Lag2)[train, ] #Matrix of predictors associated with the training data
test.X = cbind(Lag1, Lag2)[!train, ] #Matrix of predictors associated with the data for which we wish to make predictors
train.Direction = Direction[train] #Vector containign the class labels for the training observations
#4th value - the number of nearest neighbors to be used by the classifier
set.seed(1)
knn.pred = knn(train.X, test.X, train.Direction, k = 1)
table(knn.pred, Direction.2005)
(83+43)/252
#our results were bad, so we're changing value of k
knn.pred = knn(train.X, test.X, train.Direction, k = 3)
table(knn.pred, Direction.2005)
mean(knn.pred == Direction.2005)
#another example for knn
dim(Caravan)
attach(Caravan)
348/5822
standardized.X = scale(Caravan[, -86]) #standardize data: scale()
var(Caravan[, 1]) #diff vaue
var(Caravan[, 2]) #diff vaue
var(standardized.X[, 1]) #now it's 1
var(standardized.X[, 2]) #now it's 1
#Now
test = 1:1000 #split the observation into a test set of a 1000
train.X = standardized.X[-test, ] #yields the submatrix of the data containing observations not in the range 1-1000
test.X = standardized.X[test, ] #yields the submatrix of the data containing observations in the range 1-1000
train.Y = Purchase[-test]
test.Y = Purchase[test]
set.seed(1)
knn.pred = knn(train.X, test.X, train.Y, k = 1)
mean(test.Y != knn.pred)
mean(test.Y != "No")
table(knn.pred, test.Y)
9/(68+9)
summary(Purchase)
#testing k = 3
knn.pred = knn(train.X, test.X, train.Y, k = 3)
table(knn.pred, test.Y)
5/26
#testing k = 5
knn.pred = knn(train.X, test.X, train.Y, k = 5)
table(knn.pred, test.Y)
4 / 15
#now, we set a threshold of 50 to see the accuracy
glm.fits = glm(Purchase ~ ., data = Caravan, family = binomial, subset = -test)
glm.probs = predict(glm.fits, Caravan[test, ], type = "response")
glm.pred = rep("No", 1000)
glm.pred[glm.probs > .5] = "Yes"
table(glm.pred, test.Y)
#50 turned out to be too large; using 25 instead
glm.fits = glm(Purchase ~ ., data = Caravan, family = binomial, subset = -test)
glm.probs = predict(glm.fits, Caravan[test, ], type = "response")
glm.pred = rep("No", 1000)
glm.pred[glm.probs > .25] = "Yes"
table(glm.pred, test.Y)
11/(22+11)

###Poisson Regression###
attach(Bikeshare)
dim(Bikeshare)
names(Bikeshare)
#below we're fitting the linear regression model
mod.lm = lm(bikers ~ mnth + hr + workingday + temp + weathersit, data = Bikeshare)
summary(mod.lm)
#Updating negative values
contrasts(Bikeshare$hr) = contr.sum(24) #sets 24 hours in a day (so no negatives)
contrasts(Bikeshare$mnth) = contr.sum(12) #sets 12 months in a year (so no negatives)
mod.lm2 = lm(bikers ~ mnth + hr + workingday + temp + weathersit, data = Bikeshare)
summary(mod.lm2)
#Below: Checking the difference between lm and lm2
sum((predict(mod.lm) - predict(mod.lm2))^2) #result showed that the change in code does not change the result
all.equal(predict(mod.lm), predict(mod.lm2)) #this also proves the statement above
#Plotting
coef.months = c(coef(mod.lm2)[2:12], -sum(coef(mod.lm2)[2:12]))
plot(coef.months, xlab = "Month", ylab = "Coefficient", xaxt = "n", col = "blue", pch = 19, type = "o") #generates a plot and labels
axis(side = 1, at = 1:12, labels = c("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")) #marks month on x-axis
coef.hours = c(coef(mod.lm2)[13:35], -sum(coef(mod.lm2)[13:35]))
plot(coef.hours, xlab = "Hour", ylab = "Coefficient", col = "blue", pch = 19, type = "o")
#Below: fitting a Poisson regression to the Bikershare data
mod.pois = glm(bikers ~ mnth + hr + workingday + temp + weathersit, data = Bikeshare, family = poisson)
summary(mod.pois)
#Below: plotting coefficients associated with mnth and hr
coef.mnth = c(coef(mod.pois)[2:12], -sum(coef(mod.pois)[2:12]))
plot(coef.mnth, xlab = "Month", ylab = "Coefficient", xaxt = "n", col = "blue", pch = 19, type = "o")
axis(side = 1, at = 1:12, labels = c("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"))
coef.hours = c(coef(mod.pois)[13:35], -sum(coef(mod.pois)[13:35]))
plot(coef.hours, xlab = "Hour", ylab = "Coefficient", col = "blue", pch = 19, type = "o")
#Below: Obtaining the fitted values from the Poisson regression above; outputting the exp(Betta0 + Betta1X1 +...) 
plot(predict(mod.lm2), predict(mod.pois, type = "response"))
abline(0, 1, col = 2, lwd = 3)