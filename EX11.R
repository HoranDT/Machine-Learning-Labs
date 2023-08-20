###Unsupervised Learning###
##Principal Components Analysis##
states = row.names(USArrests) #gives rows of the data set (States in alph. order)
states
names(USArrests) #gives names of variables
apply(USArrests, 2, mean) #gives means of each; apply() applies the mean function to each row/column (1 for rows and 2 otherwise)
apply(USArrests, 2, var) #gives variences of each
#Below: prcomp() is used to perform PCA
pr.out = prcomp(USArrests, scale = TRUE) #scale = TRUE is used to scale allvariables to have std. deviation one
names(pr.out)
pr.out$center
pr.out$scale
pr.out$rotation #provides the principal components loadings
dim(pr.out$x)
biplot(pr.out, scale = 0) #plot first two principal components; scale = 0 ensures that the arrows are scaled to represent the loadings
#Below: making a few small changes
pr.out$rotation = -pr.out$rotation
pr.out$x = -pr.out$x
biplot(pr.out, scale = 0)

pr.out$sdev #prints std. deviation of each principal component
pr.var = pr.out$sdev^2 #getting variances
pr.var
pve = pr.var/sum(pr.var) #computes proportion of varience explain
pve #each number represents principal component explain of the varience in the data (62%, 24%,...)
#Below: plotting PVE explained by each component and cumulative PVE
par(mfrow = c(1, 2))
plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0, 1), type = "b")
plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative of Variance Explained", ylim = c(0, 1), type = "b")
#Below: cumsum() is used to compute the cumulative sum of elements of a numeric vector; example:
a = c(1, 2, 8, -3)
cumsum(a)

##Matrix Completion##
#Below: turn the data frame into a matrix, after centering and scaling each column to have mean zero and varience one
X = data.matrix(scale(USArrests))
pcob = prcomp(X)
summary(pcob)
sX = svd(X) #svd() returns three components
names(sX)
round(sX$v, 3)
pcob$rotation
#Below: recovering score vectors using the output of svd(); svd() is not necessary (prcomp() could be used instead)
t(sX$d * t(sX$u))
pcob$x
#Beow: omitting 20 entries in the 50x2 data matrix at random
##We select 20 rows at random; select 1 of the 4 entries in each row at random
nomit = 20
set.seed(15)
ina = sample(seq(50), nomit)
inb = sample(1:4, nomit, replace = TRUE)
Xna = X
index.na = cbind(ina, inb)
Xna[index.na] = NA
#Below: implementing the algorithm:
##writing a function that takes in a matrix, and returns an approximation to the matrix using svd()
fit.svd = function(X, M =1){
  svdob = svd(X)
  with(svdob,                                    #with() is used to make it easier to index elements
       u[, 1:M, drop = FALSE]%*%
         (d[1:M]*t(v[, 1:M, drop = FALSE]))
  )
}
##Initializing Xhat by replacing the missing values with the col. means of the non-missing entries
Xhat = Xna
xbar = colMeans(Xna, na.rm = TRUE)
Xhat[index.na] = xbar[inb]
##setting ourselves up to measure the progress of our iterations
thresh = 1e-7
rel_err = 1
iter = 0
ismiss = is.na(Xna) #new logical matrix with same dimensions as Xna; element is TRUE if the correspond. element is missing
mssold = mean((scale(Xna, xbar, FALSE)[!ismiss])^2) #storing the mean squared error of the non-missing elements of the old version of Xhat
mss0 = mean(Xna[!ismiss]^2) #storing mean of the squared non-missing elements 
##Iterating until relative error falls below thresh:
##approximating Xhat using fit.svd() (Xapp in the code; step 2(a) of algorithm) 
##using Xapp to update the estimates for elements in Xhat that are missing in Xna (step 2(b) of algorithm)
##computing the relative error (step 2(c) of algorithm)
while(rel_err > thresh){
  iter = iter + 1
  #2(a)
  Xapp = fit.svd(Xhat, M = 1)
  #2(b)
  Xhat[ismiss] = Xapp[ismiss]
  #2(c)
  mss = mean(((Xna - Xapp)[!ismiss])^2)
  rel_err = (mssold - mss) / mss0
  mssold = mss
  cat("Iter: ", iter, "MSS: ", mss, "Rel. Err: ", rel_err, "\n")
}
#Below: computing the correlation between the 20 imputed values and the actual values
cor(Xapp[ismiss], X[ismiss])

##Clustering##
#K-Means Clustering#
set.seed(2)
x = matrix(rnorm(50*2), ncol = 2) #creates 100 rnd numbers from a standard normal distribution (mean = 0, stand. deviation = 1)
x[1:25, 1] = x[1:25, 1] + 3 #shifts 25 values along x-axis
x[1:25, 2] = x[1:25, 2] - 4 #shifts 25 values along y-axisdownward
#Below: kmeans() is used to perform K-means clustering (in our case, K = 2)
km.out = kmeans(x, 2, nstart = 20)
km.out$cluster #contains the cluster assignments of the 50 observations
#Below: plotting the data
par(mfrow = c(1, 2))
plot(x, col = (km.out$cluster + 1), 
     main = "K-means Clustering Results with K = 2",
     xlab = "", ylab = "", pch = 20, cex = 2)
#Below: testing K-means clustering where K = 3
set.seed(4)
km.out = kmeans(x, 3, nstart = 20)
km.out
plot(x, col = (km.out$cluster + 1),
     main = "K-means Clustering Results with K = 3",
     xlab = "", ylab = "", pch = 20, cex = 2)
#Below: nstart is used for multiple initial cluster assignment
set.seed(4)
km.out = kmeans(x, 3, nstart = 1) #it's recommended to run it with large number like nstart = 20 or 50
km.out$tot.withinss #the total within-cluster sum of squares
km.out = kmeans(x, 3, nstart = 20)
km.out$tot.withinss

#Hierarchal Clustering#
#Below: hclust() implements hierarchal clustering
##We begin by clustering obser. using complete, average, and single linkage 
##dist() is used to cumpute 50x50 inter-observation Euclidean distance matrix
hc.complete = hclust(dist(x), method = "complete")
hc.average = hclust(dist(x), method = "average")
hc.single = hclust(dist(x), method = "single")
#Below: plotting the dendrograms
par(mfrow = c(1, 3))
plot(hc.complete, main = "Complete Linkage", 
     xlab = "", sub = "", cex = .9)
plot(hc.average, main = "Average Linkage", 
     xlab = "", sub = "", cex = .9)
plot(hc.single, main = "Single Linkage", 
     xlab = "", sub = "", cex = .9)
#Below: cutree() is used to get the cluster labels for each observation associated with a given cut of the dendrogram
cutree(hc.complete, 2) #second argument is the number of clusters we wish to obtain
cutree(hc.average, 2)
cutree(hc.single, 2)
#Below: getting a more sensible result for hc.single with argument 4
cutree(hc.single, 4)
#Below: scale() is used toscale the variables before performing HC of the observations
xsc = scale(x)
plot(hclust(dist(xsc), metho = "complete"),
     main = "Hierarchal Clustering with Scaled features")
#Below: clustering a 3D data set; as.dist() is used to compute the correlation-based distance
x = matrix(rnorm(30*3), ncol = 3)
dd = as.dist(1 - cor(t(x)))
plot(hclust(dd, method = "complete"),
     main = "Complete Linkage with Correlation-Based Distance",
     xlab ="", sub ="")

##NCI60 Data example##
library(ISLR2)
nci.labs = NCI60$labs
nci.data = NCI60$data
dim(nci.data)
#Below: examining the cancer types for the cell lines
nci.labs[1:4]
table(nci.labs)
#PCA on the NCI60 Data#
#Below: performing PCA on the data after scaling the variables to hace std. deviation 1
pr.out = prcomp(nci.data, scale = TRUE)
#Below: plotting the first few principal component score vectors
Cols = function(vec){
  cols = rainbow(length(unique(vec)))
  return(cols[as.numeric(as.factor(vec))])
}
#Below: plotting the principal component score vectors
par(mfrow = c(1, 2))
plot(pr.out$x[, 1:2], col = Cols(nci.labs), pch = 19,
     xlab = "Z1", ylab = "Z2")
plot(pr.out$x[, c(1, 3)], col = Cols(nci.labs), pch = 19,
     xlab = "Z1", ylab = "Z3")
#Below: summary() is used to obtain the summary of the proportion of varience explained (PVE) of the first few principal components
summary(pr.out)
plot(pr.out)
#Below: scree plots of PVE and cumulative PVE of each principal component
pve = 100*pr.out$sdev^2 / sum(pr.out$sdev^2)
par(mfrow = c(1, 2))
plot(pve, type = "o", ylab = "PVE",
     xlab = "Principal Component", col = "blue")
plot(cumsum(pve), type = "o", ylab = "Cumulative PVE",
     xlab = "Principal Component", col = "brown3")

#Clustering the Observations of the NCI60 Data#
sd.data = scale(nci.data) #standardizing the variables to have mean zero and stnd. dev. one
#Below: performing hierarchal clustering of the observations using all three linkages
par(mfrow = c(1, 3))
data.dist = dist(sd.data)
plot(hclust(data.dist), xlab = "", sub = "", ylab = "",
     labels = nci.labs, main = "Complete Linkage")
plot(hclust(data.dist, method = "average"), xlab = "", sub = "", ylab = "",
     labels = nci.labs, main = "Average Linkage")
plot(hclust(data.dist, method = "single"), xlab = "", sub = "", ylab = "",
     labels = nci.labs, main = "Single Linkage")
#Below: we will use complete linkage for the analysis; cutting the dendrogram at height 4
hc.out = hclust(dist(sd.data))
hc.clusters = cutree(hc.out, 4)
table(hc.clusters, nci.labs)
#Below: plotting the cut on the dendrogram that produces 4 clusters
par(mfrow = c(1, 1))
plot(hc.out, labels = nci.labs)
abline(h = 139, col = "red")#abline() draws a straight line on top of any plot; h = 139 is the hieght of the cut
hc.out
#Below: performing HC on the first few princial component score vectors
hc.out = hclust(dist(pr.out$x[, 1:5]))
plot(hc.out, labels = nci.labs,
     main = "Hier. Clust. on First Five Score Vectors")
table(cutree(hc.out, 4), nci.labs)