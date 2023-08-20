###Decision Trees###
##Fitting Classification Trees##
library(tree) #needed for classification and regression trees
library(ISLR2)
attach(Carseats)
High = factor(ifelse(Sales <= 8, "No", "Yes")) #Var. High takes "No" if less or equal to 8 and "Yes" otherwise
#Below: data.frame() is used to merge High with the rest of the Carseats data
Carseats = data.frame(Carseats, High)
#Below: tree() to fit classification tree to predict High
tree.carseats = tree(High ~ . - Sales, Carseats)
summary(tree.carseats)
plot(tree.carseats) #shows tree structure
text(tree.carseats, pretty = 0) #displays nodes; pretty = 0 includes the cat. names
#Above: first branch shows it as the most important indicator
tree.carseats #otputs trees as text showing: observations, the deviance, the overall prediction of the branch
#Below: predict() is used to split observ. into training and test sets; build trees using train. set; evaluate its performance
##type = "class" will return the actual class prediction
set.seed(2)
train = sample(1:nrow(Carseats), 200)
Carseats.test = Carseats[-train, ]
High.test = High[-train]
tree.carseats = tree(High ~ . -Sales, Carseats, subset = train)
tree.pred = predict(tree.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
#Above: (104 + 50)/200 = 0.77 or 77% accuracy of predictions
#Below: cv.tree() is used to perform Cross-Validation to determine the optimal level of tree complexity
##FUN = prune.misclass is used to indicate that we want the classf. error rate to guide the CV and pruning process
set.seed(7)
cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
cv.carseats
#Above: output "dev" corresponds to the number of CV errors
par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")
#Below: prune.misclass() is used to prune the tree to obtain the nine-node tree
prune.carseats = prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty = 0)
#Below: predict() is used to test the pruned tree performance on the test data
tree.pred = predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
#Above: (97 + 58)/200 = 0.775 or 77.5% accuracy
##if we play around and increase the value of "best" then the accuracy will drop

##Fitting Regression Trees##
#Below: we fit a regression tree to the Boston data set
## we create a training set, and fit the tree to the training data
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston = tree(medv ~ ., Boston, subset = train)
summary(tree.boston)
#Below: plotting the tree
plot(tree.boston)
text(tree.boston, pretty = 0)
#Below: trying to see if pruning improves the performance by using CV
cv.boston = cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = "b")
#pruning the tree
prune.boston = prune.tree(tree.boston, best = 5)
plot(prune.boston)
text(prune.boston, pretty = 0)
#using unpruned tree to make predictions on the test set
yhat = predict(tree.boston, newdata = Boston[-train, ])
boston.test = Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0, 1)
mean((yhat - boston.test)^2)
#Above: MSE associated with the regression tree is 35.29 => sqrt(MSE) = 5.941
##this means that pridictions are within $5,941 of the true median home value

##Bagging and Random Forest##
library(randomForest) #needed for Bagging and rand. Forests; alternatively I could've used "ranger" or "h2o"
set.seed(1)
bag.boston = randomForest(medv ~ ., data = Boston, subset = train, mtry = 12, importance = TRUE)
bag.boston
#Above: mtry = 12 - all 12 predictors should be considered for each split of the tree
#Below: checking the performance of bagging
yhat.bag = predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0, 1)
mean((yhat.bag - boston.test)^2)
#Above: MSE of the bagged regression = 23.42; 2/3 obtained using am optimaly-pruned single tree
#Below: ntree - used to change the number of pruned trees
bag.boston = randomForest(medv ~ ., data =Boston, subset = train, mtry = 12, ntree = 25)
yhat.bag = predict(bag.boston, newdata = Boston[-train, ])
mean((yhat.bag - boston.test)^2)
#Below: to grow a random forest, we reduce the value of mtry; by default, randomForest() uses p/3 var.
##when building a rand. forest for regression and sqrt(p) for classification; using mtry = 6
set.seed(1)
rf.boston = randomForest(medv ~ ., data = Boston, subset = train, mtry = 6, importance = TRUE)
yhat.rf = predict(rf.boston, newdata = Boston[-train, ])
mean((yhat.rf - boston.test)^2)
#Above: test set MSE = 20.07 which is improvement over bagging
importance(rf.boston) #shows the importance of each var.
varImpPlot(rf.boston) #Plotting the importance measures

##Boosting##
library(gbm) #needed for boosting
set.seed(1)
boost.boston = gbm(medv ~ ., data = Boston[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
#Above: using distribution = "gaussian" because it's a regression problem
##if it was a class. problem, we wouldhave used distribution = "bernoulli"
summary(boost.boston) #gives us a relative influence plot+outputs relative influence statistics
#Below: using rm and lstat to produce partial dependence plots
plot(boost.boston, i = "rm")
plot(boost.boston, i ="lstat")
#Below: using the boosted model to predict medv
yhat.boost = predict(boost.boston, newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test)^2)
#Above: test MSE = 18.39 which is superior ro other test we've done so far
#Below: changing the shrinkage parameter; change from default = 0.001 to 0.2
boost.boston = gbm(medv ~ ., data = Boston[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = F)
yhat.boost = predict(boost.boston, newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test)^2)
#Above; now test MSE = 16.55

###Bayesian Additive Regression Trees###
library(BART) #needed to perform BART
x = Boston[, 1:12]
y = Boston[, "medv"]
xtrain = x[train, ]
ytrain = y[train]
xtest = x[-train, ]
ytest = y[-train]
set.seed(1)
bartfit = gbart(xtrain, ytrain, x.test = xtest)
#Below: computing test error
yhat.bart = bartfit$yhat.test.mean
mean((ytest - yhat.bart)^2)
#Above: the test MSE = 15.95
#Below: checking how many times each var. appeared in the collection of trees
ord = order(bartfit$varcount.mean, decreasing = T)
bartfit$varcount.mean[ord]