library(ISLR2)
attach(Wage)
###Polynomial Regression and Step Function###
fit = lm(wage ~ poly(age, 4), data = Wage) #fourth degree polynomial; age, age^2,...
coef(summary(fit))
fit2 = lm(wage ~ poly(age, 4, raw = TRUE), data = Wage)
coef(summary(fit2))
#Below: different ways of fitting
fit2a = lm(wage ~ age + I(age^2) + I(age^3) + I(age^4), data = Wage) #I() - here it's a wrapper function
coef(fit2a)
fit2b = lm(wage ~ cbind(age, age^2, age^3, age^4), data = Wage)
#Creating a grid for age
agelims = range(age)
age.grid = seq(from = agelims[1], to = agelims[2])
preds = predict(fit, newdata = list(age = age.grid), se = TRUE)
se.bands = cbind(preds$fit + 2*preds$se.fit, preds&fit - 2*preds$se.fit)
#Plot data; add the fit from the degree-4 polynomial
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 1, 1), oma = c(0, 0, 4, 0)) #control margins
plot(age, wage, xlim = agelims, cex =.5, col = "darkgrey") 
title("Degree-4 Polynomial", outer = T) #give it a title
lines(age.grid, preds$fit, lwd = 2, col = "blue")
matlines(age.grid, se.bands, lwd = 1, col = "blue", lty = 3)
#Below: checking values from fit2
preds2 = predict(fit2, newdata = list(age = age.grid), se = TRUE)
max(abs(preds$fit - preds2$fit))
#Below: fit different models to see the simplest model to explain the relationship between age and wage
##anove() - Analysis of VAriance; tests the null hypothesis
fit.1 = lm(wage ~ age, data = Wage)
fit.2 = lm(wage ~ poly(age,2), data = Wage)
fit.3 = lm(wage ~ poly(age,3), data = Wage)
fit.4 = lm(wage ~ poly(age,4), data = Wage)
fit.5 = lm(wage ~ poly(age,5), data = Wage)
anova(fit.1, fit.2, fit.3, fit.4, fit.5)
B#Below; another way of obtaining the p-values
coef(summary(fit.5)) 
#Above: comparing t values with p-values (prev. method), we see that t^2 = p
#Below: using ANOVA to compare three methods:
fit.1 = lm(wage ~ education + age, data = Wage)
fit.2 = lm(wage ~ education + poly(age, 2), data = Wage)
fit.3 = lm(wage ~ education + poly(age, 3), data = Wage)
anova(fit.1, fit.2, fit.3)
#Below: Predicting if an individual earns >$250,000 per year; use family = "binomial" to fit a polynomial LRM
fit = glm(I(wage > 250) ~ poly(age, 4), data = Wage, family = binomial)
preds = predict(fit, newdata = list(age = age.grid), se =T)
#Below: calculating confidence intevals: Pr(Y = 1|X) = (exp(X*Betta))/(1+exp(X*Betta))
pfit = exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit +2*preds$se.fit, preds$fit - 2*preds$se.fit)
se.bands = exp(se.bands.logit) / (1 + exp(se.bands.logit))

preds = predict(fit, newdata = list(age = age.grid), type = "response", se = T)

plot(age, I(wage > 250), xlim = agelims, type = "n", ylim = c(0, .2))
points(jitter(age), I((wage > 250)/5), cex = .5, pch = "|", col = "darkgrey") #jitter() - helps with readability; rug plot
lines(age.grid, pfit, lwd = 2, col = "blue")
matlines(age.grid, se.bands, lwd = 1, col = "blue", lty = 3)
#Below: step function
table(cut(age, 4)) #cut() - picks cutpoints  
fit = lm(wage ~ cut(age, 4), data = Wage) #lm() - creates a set of dummy variables to use in the regression
coef(summary(fit))

###Splines###
library(splines)
#Below: bs() - generates the entire matrix basis functions for splines with the specified set of knots
fit = lm(wage ~ bs(age, knots = c(25, 40, 60)), data = Wage)
pred = predict(fit, newdata = list(age = age.grid), se = T)
plot(age, wage, col = "gray")
lines(age.grid, pred$fit, lwd = 2)
lines(age.grid, pred$fit + 2*pred$se, lty = "dashed")
lines(age.grid, pred$fit - 2*pred$se, lty = "dashed")
#Below: Knots
dim(bs(age, knots = c(25, 40, 60)))
dim(bs(age, df = 6)) #df() - produces a spline of knots at uniform quantities; 6 is from previos line, second value (columns maybe)
attr(bs(age, df = 6), "knots") #chooses knots
#Below: fitting a natural spline using ns()
fit2 = lm(wage ~ ns(age, df = 4), data = Wage)
pred2 = predict(fit2, newdata = list(age = age.grid), se = TE)
lines(age.grid, pred2$fit, col = "red", lwd = 2)
#Below: fitting a smoothing spline
plot(age, wage, xlim = agelims, cex = .5, col = "darkgrey")
title("Smoothing Splines")
fit = smooth.spline(age, wage, df = 16) #df - degrees of freedom
fit2 = smooth.spline(age, wage, cv = TRUE)
fit2$df #gives df of the second fit
lines(fit, col = "red", lwd = 2)
lines(fit2, col = "blue", lwd = 2)
legend("topright", legend = c("16 DF", "6.8 DF"), col = c("red", "blue"), lty = 1, lwd = 2, cex = .8)
#Below: loess() - needed to perform local regression
plot(age, wage, xlim = agelims, cex = .5, col = "darkgrey")
title("Local Regression")
fit = loess(wage ~ age, span = .2, data = Wage)
fit2 = loess(wage ~ age, span = .5, data = Wage)
lines(age.grid, predict(fit, data.frame(age = age.grid)), col = "red", lwd = 2)
lines(age.grid, predict(fit2, data.frame(age = age.grid)), col = "blue", lwd = 2)
legend("topright", legend = c("Span = 0.2", "Span = 0.5"), col = c("red", "blue"), lty = 1, lwd = 2, cex = 0.8)

##GAMs##
gam1 = lm(wage ~ ns(year, 4) + ns(age, 5) + education, data = Wage)
library(gam)
gam.m3 = gam(wage ~ s(year, 4) + s(age, 5) + education, data = Wage)
par(mfrow = c(1, 3))
plot(gam.m3, se = TRUE, col = "blue") #gives plots
#Below: since gam.m3 is an object of class GAM => use plot.Gam()
plot.Gam(gam1, se = TRUE, col = "red")
#Below: Testing 3 methods (GAM without year; GAM with a linear function of year; GAM with a spline function of year)
gam.m1 = gam(wage ~ s(age, 5) + education, data = Wage)
gam.m2 = gam(wage ~ year + s(age, 5) + education, data = Wage)
gam.m3 = gam(wage ~ s(age, 4) + s(age, 5) + education, data = Wage)
anova(gam.m1, gam.m2, gam.m3, test = "F")
summary(gam.m3)
#Below: making predictions for Gam
preds = predict(gam.m2, newdata = Wage)
#Below: lo() - used for local regression
gam.lo = gam(wage ~ s(year, df = 4) + lo(age, span = 0.7) + education, data = Wage)
plot.Gam(gam.lo, se =TRUE, col = "green")
#Below: using lo() for interaction between year and age
gam.lo.i = gam(wage ~ lo(year, age, span = 0.5) + education, data = Wage)
library(akima) #used to plot a 2D surface
plot(gam.lo.i)
#Below: fitting logistic regression GAM
gam.lr = gam(I(wage > 250) ~ year + s(age, df = 5) + education, family = binomial, data = Wage)
par(mfrow = c(1, 3)) #parameters
plot(gam.lr, se = TRUE, col = "green")
table(education, I(wage > 250)) #gives a T/F table based on info from education
#Below: fitting logistic regression GAM without category (<HS Grad) from education list; that class had 0 True
gam.lr.s = gam(I(wage > 250) ~ year + s(age, df = 5) + education, family = binomial, data = Wage, subset = (education != "1. < HS Grad"))
plot(gam.lr.s, se = TRUE, col = "green")
