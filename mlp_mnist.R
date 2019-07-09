# mlp_mnist.R

#install.packages("keras")

library(keras)

# Data preparation

batch_size  <- 128
num_classes <- 10
epochs      <- 30

# The data, shuffled and split between train and test sets
mnist   <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y
dim(x_train)  <- c(nrow(x_train),784)

dim(x_test)  <- c(nrow(x_test),784)

# Transform RGB values info [0,1] range
x_train <- x_train/255
x_test <- x_test/255

cat(nrow(x_train),'train sample \n')
cat(nrow(x_test),'test sample \n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define model

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')
  
model %>%compile(  
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Training & Evaluation

# Fit model to data
history <- model %>% fit(
    x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_split = 0.2
)



