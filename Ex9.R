###Deep Learning###
library(reticulate) #needed to connect to env.
use_condaenv("tensorflow_env", required = TRUE) #connects to tensorflow env.
##A Single Layer Network on The Hitters Data##
library(ISLR2)
Gitters = na.omit(Hitters)
n = nrow(Gitters)
set.seed(13)
ntest = trunc(n/3)
testid = sample(1:n, ntest)
lfit = lm(Salary ~ ., data = Gitters[-testid, ])
lpred = predict(lfit, Gitters[testid, ])
with(Gitters[testid, ], mean(abs(lpred - Salary))) #with(dataframe, mean absolute predictor)
#Below: writing formulas to use glmnet()
x = scale(model.matrix(Salary ~ . -1, data = Gitters)) # -1 omits the intercept; converts factors to dummy var.
y = Gitters$Salary
library(glmnet) #used for lasso
cvfit = cv.glmnet(x[-testid, ], y[-testid], type.measure = "mae")
cpred = predict(cvfit, x[testid, ], s = "lambda.min")
mean(abs(y[testid] - cpred))
#Below: setting up a model structure
library(keras)
modnn = keras_model_sequential() %>% layer_dense(units = 50, activation = "relu", input_shape = ncol(x)) %>%
  layer_dropout(rate = 0.4) %>% layer_dense(units = 1)
#Above: %>% is the pipe operator that passes the previous term at the first argument to the next function
x = scale(model.matrix(Salary ~ . -1, data = Gitters))
##or x = model.matrix(Salary ~ . -1, data = Gitters) %>% scale()
#Below: adding details to modnn
modnn %>% compile(loss = "mse", optimizer = optimizer_rmsprop(), metrics = list("mean_absolute_error"))
#Below: history is used to display the mean absolute error for both training and test data
history = modnn %>% fit(x[-testid, ], y[-testid], epochs = 1500, batch_size = 32, validation_data = list(x[testid, ], y[testid]))
#Above: rerunning fit() will make the fitting process to pick up from where it stopped 
library(ggplot2) #used for best aesthetics
plot(history)
npred = predict(modnn, x[testid, ])
mean(abs(y[testid] - npred))

##A Multilayer Network on the MNSIT Digit Data##
mnist = dataset_mnist() #accessing MNIST dataset
x_train = mnist$train$x
g_train = mnist$train$y
x_test = mnist$test$x
g_test = mnist$test$y
dim(x_train)
dim(x_test)
#Below: reshaping images into matrixes
x_train = array_reshape(x_train, c(nrow(x_train), 784))
x_test = array_reshape(x_test, c(nrow(x_test), 784))
y_train = to_categorical(g_train, 10)
y_test = to_categorical(g_test, 10)
#Below: rescaling images because they're 8-bit or 2^8 => 0 - 255 values; NN are somewhat sensitive to the scale
x_train = x_train/255
x_test = x_test/255
#Below: fitting our NN
modelnn = keras_model_sequential()
modelnn %>% layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")
summary(modelnn)
#Below: adding details to the model to specify the fitting algorithm
modelnn %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(), metrics = c("accuracy"))
#Below: supplying training data and fitting the model
system.time(history <- modelnn %>% fit(x_train, y_train, epochs = 30, batch_size = 128, validation_split = 0.2))
plot(history, smooth = FALSE)
#Below: obtaining the test error; accuracy() compares predicted and true class labels, and then evaluates our predictions
accuracy <- function(pred, truth)
  mean(drop(pred) == drop(truth))
modelnn %>% predict_classes(x_test) %>% accuracy(g_test)
#Below: using keras for multiclass logistic regression
modellr = keras_model_sequential() %>% 
  layer_dense(input_shape = 784, units = 10, activation = "softmax") # 784 = 28x28
summary(modellr)
#Below: fitting the model
modellr %>% compile(loss = "categorical_crossentropy",
                    optimizer = optimizer_rmsprop(), metrics = c("accuracy"))
modellr %>% fit(x_train, y_train, epochs = 30, batch_size = 128, validation_split = 0.2)
modellr %>% predict_classes(x_test) %>% accuracy(g_test)

##Convolutional Neural Networks##
cifar100 = dataset_cifar100()
names(cifar100)
x_train = cifar100$train$x
g_train = cifar100$train$y
x_test = cifar100$test$x
g_test = cifar100$test$y
dim(x_train)
range(x_train[1,,, 1]) #first 1 - first image of the set; ,,, - at least 4D; last 1- color channel (typically Red)
#Below: one-hot encode the response factor to produce a 100-col. binary matrix
x_train = x_train/255
x_test = x_test/255
y_train = to_categorical(g_train, 100)
dim(y_train)
#Below: taking a look at some of the training images using jpeg
library(jpeg)
par(mar = c(0, 0, 0, 0), mfrow = c(5, 5))
index = sample(seq(50000), 25)
for (i in index) plot(as.raster(x_train[i,,, ])) #as.raster() - converts the feature map to be plotted as a color image
#Below: specifying a moderately-sized CNN
model = keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
  padding = "same", activation = "relu", input_shape = c(32, 32, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
  padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3),
  padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 256, kernel_size = c(3, 3),
  padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 100, activation = "softmax")
#Above: padding = "same" ensures that output channels have the same dimensions as the input channels
##We used a 3x3 convolution filter for each channel; each convolution is followed by a max-pooling layer over 2x2
summary(model)
#Below: specifying the fitting algorithm and fitting the model
model %>% compile(loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(), metrics = c("accuracy"))
history = model %>% fit(x_train, y_train, epochs = 30, 
    batch_size = 128, validation_split = 0.2) #took ~10 minutes to run
model %>% predict_classes(x_test) %>% accuracy(g_test) #we got accuracy of 46%
#This can be done using these commands:
##accuracy <- base::mean(as.integer(predicted_classes) == as.integer(g_test))
##summary(accuracy)

##Using Pretrained CNN Models##
getwd() #checking my directory
list.files() #checking what files I have
setwd("C:\\Users\\xopan\\Documents\\book_images") #changing work directory
list.files(pattern = "\\.jpg$") #checking what .jpeg files I have in that directory

image_names = list.files(pattern = "\\.jpg$") #retrieving image names from directory
num_images = length(image_names)
#Below: Had to run these commands to make it work:
##library(reticulate)
##py_install("Pillow")
##py_install("Pillow", pip = TRUE)
x = array(dim = c(num_images, 224, 224, 3))
for (i in 1:num_images){
  img_path = image_names[i]
  img = image_load(img_path, target_size = c(224, 224))
  x[i,,, ] = image_to_array(img)
}
x = imagenet_preprocess_input(x)
#Below: loading the trained network with 50 models
model = application_resnet50(weights = "imagenet")
summary(model)
#Below: classifying 5 images and returning the top 3 class choices in terms of predicted prob.
pred6 = model %>% predict(x) %>%
  imagenet_decode_predictions(top = 3)
names(pred6) = image_names
print(pred6)

##IMDb Document Classification##
max_features = 10000
imdb = dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb # shortcut for unpacking the list of lists
#provides 12 words from the review matching the dictionary word (0-999)
x_train[[1]][1:12]
#decode_review provides a simple interface to the dictionary
word_index = dataset_imdb_word_index()
decode_review = function(text, word_index){
  word = names(word_index)
  idx = unlist(word_index, use.names = FALSE)
  word = c("<PAD>", "<START>", "<UNK>", "<UNUSED>", word)
  idx = c(0:3, idx + 3)
  words = word[match(text, idx, 2)]
  paste(words, collapse = " ")
}
decode_review(x_train[[1]][1:12], word_index)
#Below: writing function to one-hot encode each doc. in a list of doc.; return binary matrix in sparse-matrix format
library(Matrix)
one_hot = function(sequences, dimension){
  seqlen = sapply(sequences, length)
  n = length(seqlen)
  rowind = rep(1:n, seqlen)
  colind = unlist(sequences)
  sparseMatrix(i = rowind, j = colind, dims = c(n, dimension))
}
x_train_1h = one_hot(x_train, 10000)
x_test_1h = one_hot(x_test, 10000)
dim(x_train_1h)
nnzero(x_train_1h)/(25000*10000) #gives the amount of non zero entries (1.3%)
#Below: creating a validation set of size 2,000 leaving 23,000 for training
set.seed(3)
ival = sample(seq(along = y_train), 2000)
#Below; fitting a lasso logistic regression on the training data
fitlm = glmnet(x_train_1h[-ival, ], y_train[-ival],
    family = "binomial", standardized = FALSE)
classlmv = predict(fitlm, x_train_1h[ival, ]) > 0
acclmv = apply(classlmv, 2, accuracy, y_train[ival] > 0) #Using accuracy function that we wrote earlier
#adjusting the plot window and plotting
par(mar = c(4, 4, 4, 4), mfrow = c(1, 1))
plot(-log(fitlm$lambda), acclmv)
#Below: fitting a fully connected NN with two hidden layers each with 16 units
model = keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu",
              input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy", metrics = c("accuracy"))
history = model %>% fit(x_train_1h[-ival, ], y_train[-ival],
                        epochs = 20, batch_size = 512,
                        validation_data = list(x_train_1h[ival, ], y_train[ival]))
#Below: computing the test accuracy by rerunning the above sequences and replacing the last line with this one
history = model %>% fit(x_train_1h[-ival, ], y_train[-ival], epochs = 20,
                        batch_size = 512, validation_data = list(x_test_1h, y_test))

##Recurrent Neural Networks##
#Sequential Models for Document Classification#
#Below: calculating the length of the documents
wc = sapply(x_train, length)
median(wc)
sum(wc <= 500) / length (wc) #we get 0.91568 => 91.6% of the documents have fewer than 500 words
#Below: restricting the doc. lengths to the last 500 words; pad the beginning of the shorter ones with blanks
maxlen = 500
x_train = pad_sequences(x_train, maxlen = maxlen)
x_test = pad_sequences(x_test, maxlen = maxlen)
dim(x_train)
dim(x_test)
x_train[1, 490:500] #shows the last few words in the doc
#Below: Encoding each doc as matrix of 500x10,000; maps 10,000 dimensions down to 32
model = keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32) %>%
  layer_lstm(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")
#Below: fitting the model
model %>% compile(optimizer = "rmsprop",
                  loss ="binary_crossentropy", metrics = c("acc"))
history = model %>% fit(x_train, y_train, epochs = 10,
                        batch_size = 128, validation_data = list(x_test, y_test))
plot(history)
predy = predict(model, x_test) > 0.5
mean(abs(y_test == as.numeric(predy))) #shows accuracy which is 87.3%
#Time Series Prediction#
#Below: setting up the data, and standardizing each of the elements
#library(ISLR2) is needed; been used above
xdata = data.matrix(NYSE[, c("DJ_return", "log_volume", "log_volatility")])
istrain = NYSE[, "train"] #contains T for each year that is in the train. set, and F otherwise
xdata = scale(xdata)
#Below: creating a function to create lagged versions of the three time series
lagm = function(x, k = 1){
  n = nrow(x)
  pad = matrix(NA, k, ncol(x))
  rbind(pad, x[1:(n - k), ])
}
#Below: creating a data frame with all the required lags
arframe = data.frame(log_volume = xdata[, "log_volume"],
                     L1 = lagm(xdata, 1), L2 = lagm(xdata, 2),
                     L3 = lagm(xdata, 3), L4 = lagm(xdata, 4),
                     L5 = lagm(xdata, 5))
#Below: removing rows with missing val.; adjusting istrain 
arframe = arframe[-(1:5), ]
istrain = istrain[-(1:5)]
#Below: fitting linear AR model to the train. data using lm(); predicting the test data
arfit = lm(log_volume ~ ., data = arframe[istrain, ])
arpred = predict(arfit, arframe[!istrain, ])
V0 = var(arframe[!istrain, "log_volume"])
1 - mean((arpred - arframe[!istrain, "log_volume"])^2)/V0 #last two lines compute R^2 which is 0.41
#Below: refitting this model including the factor day_of_week
arframed = data.frame(day = NYSE[-(1:5), "day_of_week"], arframe)
arfitd = lm(log_volume ~ ., data = arframed[istrain, ])
arpred = predict(arfitd, arframed[!istrain, ])
1 - mean((arpred - arframe[!istrain, "log_volume"])^2)/V0 #now it's 0.46
#Below; reshaping data since it expects a sequence of L = 5
n = nrow(arframe)
xrnn = data.matrix(arframe[, -1]) #extracts the nx15 matrix of lagged versions of the three predictors from arframe
xrnn = array(xrnn, c(n, 3, 5)) #converts matrix into nx3x5 array
xrnn = xrnn[,, 5:1] #reversing the order of lagged var. to be 5-1 (was 1-5)
xrnn = aperm(xrnn, c(1, 3, 2)) #rearranges the coords of the array into the format that RNN in keras expects
dim(xrnn)
#Below: proceeding with the RNN thatuses 12 hidden units
model = keras_model_sequential() %>%
  layer_simple_rnn(units = 12, input_shape = list(5, 3),
                   dropout = 0.1, recurrent_dropout = 0.1) %>%
  layer_dense(units = 1)
model %>% compile(optimizer = optimizer_rmsprop(), loss = "mse")
#Below: fitting the model
history = model %>% fit(
  xrnn[istrain,, ], arframe[istrain, "log_volume"],
  batch_size = 64, epochs = 200,
  validation_data = 
    list(xrnn[!istrain,, ], arframe[!istrain, "log_volume"])
)
kpred = predict(model, xrnn[!istrain,, ])
1 - mean((kpred - arframe[!istrain, "log_volume"])^2)/V0 #0.41
#Below: performing a nonlinear AR model
x = model.matrix(log_volume ~ . -1, data = arframed) # -1 avoids creation of a col. of ones for the intercept
arnnd = keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
               input_shape = ncol(x)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)
arnnd %>% compile(loss = "mse", 
                  optimizer = optimizer_rmsprop())
history = arnnd %>% fit(
  x[istrain, ], arframe[istrain, "log_volume"], epochs = 100,
  batch_size = 32, validation_data =
    list(x[!istrain, ], arframe[!istrain, "log_volume"])
)
plot(history)
npred = predict(arnnd, x[!istrain, ])
1 - mean((arframe[!istrain, "log_volume"] - npred)^2)/V0 #0.47
colnames(x)