
#install.packages("keras")

library(keras)

# Data preparation

batch_size  <- 128
num_classes <- 10
epochs      <- 12

# Input images dimensions
img_rows <- 28
img_cols <- 28

# The data, shuffled and split between train and test sets
mnist   <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y

dim(x_train)  <- c(nrow(x_train),784)
dim(x_test)  <- c(nrow(x_test),784)

# Redefine dimension of train/test inputs
dim(x_train) <- c(nrow(x_train), img_rows, img_cols,1)
dim(x_test) <- c(nrow(x_test), img_rows, img_cols,1)
input_shape <- c(img_rows, img_cols,1)

# Transform RGB values info [0,1] range
x_train <- x_train/255
x_test <- x_test/255

cat('x_train_shape',dim(x_train), '\n')
cat(nrow(x_train),'train sample \n')
cat(nrow(x_test),'test sample \n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define model

model <- keras_model_sequential()
model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                activation = 'relu', input_shape = input_shape) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')

model %>%compile(  
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Training & Evaluation

# Fit model to data
history <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(x_test, y_test)
)


