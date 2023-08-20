###Survival Analysis###
##Brain Cancer Data##
library(ISLR2)
names(BrainCancer)
#Below: Examining data
attach(BrainCancer)
table(sex)
table(diagnosis)
table(status) #status = 1 is uncensored information and 0 otherwise
#Below: survfit() is used to create the Kaplan-Meier curve
library(survival)
fit.surv = survfit(Surv(time, status) ~ 1)
plot(fit.surv, xlab = "Months", ylab = "Estimated Probability of Survival")
#Below: creating Kaplan-Meier survival curves that are stratified by sex
fit.sex = survfit(Surv(time, status) ~ sex)
plot(fit.sex, xlab = "Months", ylab = "Estimated Probability of Survival", col = c(2, 4))
legend("bottomleft", levels(sex), col = c(2, 4), lty = 1)
#Below: survdiff() is used to perform log-rank test to compare the survival of males to females
logrank.test = survdiff(Surv(time, status) ~ sex)
logrank.test #we got p = 0.2 indicating no evidence of a difference in survival between the two sexes
#Below: coxph() is used for fitting Cox proportional hazards model; considering a model that uses sex as the only predictor
fit.cox = coxph(Surv(time, status) ~ sex)
summary(fit.cox)
#Below: displaying non-rounded values
summary(fit.cox)$logtest[1]
summary(fit.cox)$waldtest[1]
summary(fit.cox)$sctest[1]
logrank.test$chisq
#Below: fitting a model that makes use of additional predictors
fit.all = coxph(
  Surv(time, status) ~ sex + diagnosis + loc + ki + gtv + stereo)
fit.all
#Below: creating a data frame; survfit() will produce a curve for each of the rows; plot() will diplay them all in the same plot
modaldata = data.frame(
  diagnosis = levels(diagnosis),
  sex = rep("Female", 4),
  loc = rep("Supratentorial", 4),
  ki = rep(mean(ki), 4),
  gtv = rep(mean(gtv), 4),
  stereo = rep("SRT", 4)
)
survplots = survfit(fit.all, newdata = modaldata)
plot(survplots, xlab = "Months",
     ylab = "Survival Probability", col = 2:5)
legend("bottomleft", levels(diagnosis), col = 2:5, lty = 1)

##Publication Data##
#Below: plotting Kaplan-Meier on the posres variable (it records whether the study had a pos or neg result)
fit.posres = survfit(Surv(time, status) ~ posres, data = Publication)
plot(fit.posres, xlab = "Months", ylab = "Probability of Not Being Published", col = 3:4)
legend("topright", c("Negative Result", "Positive Result"), col = 3:4, lty = 1)
#Below: fitting Cox proportional hazrds 
fit.pub = coxph(Surv(time, status) ~ posres, data = Publication)
fit.pub
#Below: performing a log-rank test
logrank.test = survdiff(Surv(time, status) ~ posres, data = Publication)
#Below: fitting model (funding mech. var. is excluded)
fit.pub2 = coxph(Surv(time, status) ~ . -mech, data = Publication)
fit.pub2

##call Center Data##
set.seed(4)
N = 2000
Operators = sample(5:15, N, replace = T)
Center = sample(c("A", "B", "C"), N, replace = T)
Time = sample(c("Morn.", "After.", "Even."), N, replace = T)
X = model.matrix(~ Operators + Center + Time)[, -1]
X[1:5, ] #gives information about the matrix
#Below: specifying the coefficients and the hazard function
true.beta = c(0.04, -0.3, 0, 0.2, -0.2) #c(Operators, Center = B,Center = A,...)
h.fn = function(x) return(0.00001 * x)
#Below: generating data under Cox prop. hazards model; simsurvdata() allows us to specify the max possible failure time
library(rms)
library(coxed)
queuing = sim.survdata(N = N, T = 1000, X = X, beta = true.beta, hazard.fun = h.fn)
names(queuing)
head(queuing$data)
mean(queuing$data$failed) #gives 0.89 => almost 90% of calls answered
#Below: plotting Kaplan-Meier survival curves
par(mfrow = c(1, 2))
fit.Center = survfit(Surv(y, failed) ~ Center, data = queuing$data)
plot(fit.Center, xlab = "Seconds", ylab = "Probability of Still Being on Hold",
     col = c(2, 4, 5))
legend("topright", c("Call Center A", "Call Center B", "Call Center C"),
       col = c(2, 4, 5), lty = 1) #we see that Call center B is the longest to be answered
#Below: stratifying by Time
fit.Time = survfit(Surv(y, failed) ~ Time, data = queuing$data)
plot(fit.Time, xlab = "Seconds", ylab = "Probability of Still Being on Hold",
     col = c(2, 4, 5))
legend("topright", c("Morning", "Afternoon", "Evening"),
       col = c(5, 2, 4), lty = 1) #we see that Morning has the longest wait time
#Below: using log-rank test to see if those differences are statistically significant
survdiff(Surv(y, failed) ~ Center, data = queuing$data)
survdiff(Surv(y, failed) ~ Time, data = queuing$data)
#Below; fitting Cox's prop. hazards model to the data
fit.queuing = coxph(Surv(y, failed) ~ . ,data = queuing$data)
fit.queuing