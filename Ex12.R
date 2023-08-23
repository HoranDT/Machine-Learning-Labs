###Multiple Testing###
##Review of Hypothesis Test##
#Below: we create 100 var. with 10 obs. each; 50 var. have mean 0.5 and var. 1, other 50 have mean 0 and var 1
set.seed(6)
x = matrix(rnorm(10*100), 10, 100)
x[, 1:50] = x[, 1:50] + 0.5
#Below: t.test() is used to perform a one-sample t-test
t.test(x[, 1], mu = 0)
#Below: we compute th 100 p-val., and then construct a vector recording whether the jth p-val. is less or equal to 0.05 to check if we reject the null hypothesis
p.values = rep(0, 100)
for(i in 1:100)
  p.values[i] = t.test(x[, i], mu = 0)$p.value
decision = rep("Do not reject H0", 100)
decision[p.values <= 0.05] = "Reject h0"
#Below: creating a 2x2 table to analize data
table(decision,
      c(rep("H0 is False", 50), rep("H0 is True", 50)))
#Below: testing with mean 1 instead of mean 0.5; gives a better performance
x = matrix(rnorm(10*100), 10, 100)
x[, 1:50] = x[, 1:50] + 1
for(i in 1:100)
  p.values[i] = t.test(x[, i], mu = 0)$p.value
decision = rep("Do not reject H0", 100)
decision[p.values <= 0.05] = "Reject h0"
table(decision,
      c(rep("H0 is False", 50), rep("H0 is True", 50)))

##The Family-Wise Error Rate##
#Below: computing FWER for m = 1,...,500 and alpha = 0.05, 0.01, and 0.001
m = 1:500
fwe1 = 1 - (1 - 0.05)^m
fwe2 = 1 - (1 - 0.01)^m
fwe3 = 1 - (1 - 0.001)^m
#Below: plotting these vectors
par(mfrow = c(1, 1))
plot(m, fwe1, type = "l", log = "x", ylim = c(0, 1), col = 2,
     ylab = "Family-Wise Error Rate",
     xlab = "Number of Hypotheses")
lines(m, fwe2, col = 4)
lines(m, fwe3, col = 3)
abline(h = 0.05, lty = 2)
#Below: performing one-sample t-test for each of the first 5 managers
library(ISLR2)
fund.mini = Fund[, 1:5]
t.test(fund.mini[, 1], mu = 0)
fund.pvalue = rep(0, 5)
for(i in 1:5)
  fund.pvalue[i] = t.test(fund.mini[, i], mu = 0)$p.value
fund.pvalue
#Below: p.adjust() is used to perform Bonferroni's
p.adjust(fund.pvalue, method = "bonferroni")
pmin(fund.pvalue*5, 1)
#Below: p.adjust() is used to perform Holm's
p.adjust(fund.pvalue, method = "holm")
apply(fund.mini, 2, mean) #calculates the mean average
#Below: checking if there's a meaningful difference in performance between managers 1&2 using paired t-test
t.test(fund.mini[, 1], fund.mini[, 2], paired = T)
#Below: TukeyHSD() is used to adjust for multiple testing
returns = as.vector(as.matrix(fund.mini))
manager = rep(c("1", "2", "3", "4", "5"), rep(50, 5))
a1 = aov(returns ~ manager)
TukeyHSD(x = a1)
plot(TukeyHSD(x = a1)) #plotting the confidence intervals for the pairwise comparison

##The False Discovery Rate##
#Below: performing the hypothesis test for all 2,000 fund managers
fund.pvalues = rep(0, 2000)
for(i in 1:2000)
  fund.pvalues[i] = t.test(Fund[, i], mu = 0)$p.value
#Below: p.adjust() is used to perform the Benjamini-Hochberg procedure
q.values.BH = p.adjust(fund.pvalues, method = "BH")
q.values.BH[1:10]
sum(q.values.BH <= 0.1) #checks how many managers have a q-value less than 0.1
#Below: the Benjamini-Hochberg procedure performed manually
ps = sort(fund.pvalues) 
m = length(fund.pvalues)
q = 0.1
wh.ps = which(ps < q * (1:m)/m)
if (length(wh.ps) > 0){
  wh = 1:max(wh.ps)
}else{
  wh = numeric(0)
}
#Below: plotting
plot(ps, log = "xy", ylim = c(4e-6, 1), ylab = "P-value",
     xlab = "Index", main = "")
points(wh, ps[wh], col = 4)
abline(a = 0, b = (q/m), col = 2, untf = TRUE)
abline(h = 0.1 / 2000, col = 3)

##A Re-Sampling Approach##
attach(Khan)
#Below: merging the training and testing data
x = rbind(xtrain, xtest)
y = c(as.numeric(ytrain), as.numeric(ytest))
dim(x)
table(y)
#Below: comparing the mean expression in the second class to the mean expression in the fourth class
x = as.matrix(x)
x1 = x[which(y == 2), ]
x2 = x[which(y == 4), ]
n1 = nrow(x1)
n2 = nrow(x2)
#Below: performing a two-sample t-test on the 11th gene
t.out = t.test(x1[, 11], x2[, 11], var.equal = TRUE)
TT = t.out$statistic
TT
#Below: computing the fraction of the time that our observed test statistic exceeds the test statistics obtained via re-sampling
set.seed(1)
B = 10000
Tbs = rep(NA, B)
for (b in 1:B){
  dat = sample(c(x1[, 11], x2[, 11]))
  Tbs[b] = t.test(dat[1:n1], dat[(n1 + 1):(n1 + n2)],
                  var.equal = TRUE)$statistic
}
mean((abs(Tbs) >= abs(TT)))
#Below: plotting the histogram of the re-sampling-based test statistic 
hist(Tbs, breaks = 100, xlim = c(-4.2, 4.2), main = "",
     xlab = "Null Distribution of Test Statistic", col = 7)
lines(seq(-4.2, 4.2, len = 1000),
      dt(seq(-4.2, 4.2, len = 1000),
         df = (n1 + n2 - 2))*1000, col = 2, lwd = 3)
abline(v = TT, col = 4, lwd = 2)
text(TT + 0.5, 350, paste("T = ", round(TT, 4), sep = ""), col = 4)
#Below: implementing the plug-in re-sampling FDR approach
m = 100
set.seed(1)
index = sample(ncol(x1), m)
Ts = rep(NA, m)
Ts.star = matrix(NA, ncol = m, nrow = B) #set B = 500 or so to make it run faster
for (j in 1:m){
  k = index[j]
  Ts[j] = t.test(x1[, k], x2[, k], var.equal = TRUE)$statistic
  for (b in 1:B){
    dat = sample(c(x1[, k], x2[, k]))
    Ts.star[b, j] = t.test(dat[1:n1],
                           dat[(n1 + 1):(n1 + n2)], var.equal = TRUE)$statistic
  }
  
}
#Below: computing the #-rejected null hypothesis R, estimated #-false positives Vhat, and estimated FDR for a range of threshold val. c
cs = sort(abs(Ts))
FDRs = Rs = Vs = rep(NA, m)
for(j in 1:m){
  R = sum(abs(Ts) >= cs[j])
  V = sum(abs(Ts.star) >= cs[j]) / B
  Rs[j] = R
  Vs[j] = V
  FDRs = V/R
}
#Below: finding genes that will be rejected
max(Rs[FDRs <= 0.1]) #with the FDR at 0.1, we reject 15/100 null hypothesis
sort(index[abs(Ts) >= min(cs[FDRs < 0.1])])
max(Rs[FDRs <= 0.2]) #with the FDR at 0.1, we reject 28/100 null hypothesis
sort(index[abs(Ts) >= min(cs[FDRs < 0.2])])
plot(Rs, FDRs, xlab = "Number of rejections", type = "l",
     ylab = "False Discovery Rate", col = 4, lwd = 3)
