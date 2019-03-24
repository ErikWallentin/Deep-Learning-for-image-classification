######################################################################################################
# The goal of this project is to train a neural network model from scratch which classifies whether an
# image contains a dog or not. In order to do this, a datafile from the Kaggle user "chetanimravan" must
# be imported. The file contains 10,000 images where 5000 images are of dogs and 5000 of cats. 
# Link to the file: https://www.kaggle.com/chetankv/dogs-cats-images
# (To be able download the data set, you must first create a cost-free account on Kaggle)

# The data material is divided into a training, validation and test set of 7000, 1000 and 2000 images
# where each set contains as many dogs as cat pictures.
# The data is divided into a training and validation set so that you in real time can see how accurate
# the model predicts images it never seen before during the training phase. This is important so that 
# you e.g. can get an idea of how many epochs you should train the final model on. The final test set
# is then used to evaluate the model. Because of limited computer performance, only two models are 
# created with predetermined numbers of epochs, 30 and 100. However, the model trained on 100 epochs
# uses two methods that prevent overfitting.

# A popular package used in many depp learning projects is keras, and that's the package we will use
# in this image classification-project. For keras to work properly, the program "Anaconda" is needed
# on the user's computer. The project starts by setting up keras properly, and the user can follow 
# the instructions in the R console if encountering problems.

# When keras is successfully installed, the next step is to enable R to read the 10,000 images that
# are included in the project before it's time for training and evaluation of how good the models are
# in classifying an image as a dog or non-dog.
# The project concludes with a brief reflection on how you could create a model that classifies an
# image as a dog or non-dog with an even higher accuracy.
######################################################################################################


# Install the keras package.
install.packages("keras")
library(keras)
install_keras() # Denna rad behöver endast köras första gången keras-paketet hämtas.

# Enable R to find the images contained in the file "dataset".
# In row 35, enter the searchpath to where you saved the file "dataset".
dir <- "C:/.../.../.../dataset"

train_dir <- file.path(dir, "training")

validation_dir <- file.path(dir, "validation")

test_dir <- file.path(dir, "test")

train_cats_dir <- file.path(train_dir, "cats")

train_dogs_dir <- file.path(train_dir, "dogs")

validation_cats_dir <- file.path(validation_dir, "cats")

validation_dogs_dir <- file.path(validation_dir, "dogs")

test_cats_dir <- file.path(test_dir, "cats")

test_dogs_dir <- file.path(test_dir, "dogs")


fnames <- paste0("cat.", 1:500, ".jpg" )
file.copy(file.path(dir, fnames), file.path(validation_cats_dir))

fnames <- paste0("dog.", 1:500, ".jpg" )
file.copy(file.path(dir, fnames), file.path(validation_dogs_dir))

length(list.files(validation_cats_dir)) # If everything went as it should, the value shown is 500.
length(list.files(validation_dogs_dir)) # If everything went as it should, the value shown is 500.


fnames <- paste0("cat.", 501:4000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(train_cats_dir))

fnames <- paste0("dog.", 501:4000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(train_dogs_dir))

length(list.files(train_cats_dir)) # If everything went as it should, the value shown is 3500.
length(list.files(train_dogs_dir)) # If everything went as it should, the value shown is 3500.

fnames <- paste0("cat.", 4001:5000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(test_cats_dir))

fnames <- paste0("dog.", 4001:5000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(test_dogs_dir))

length(list.files(test_cats_dir)) # If everything went as it should, the value shown is 1000.
length(list.files(test_dogs_dir)) # If everything went as it should, the value shown is 1000.


# A visual test that R can find the images in the file "dataset" is performed below.
# Picture number 17 in the test dataset containing dogs is visualized if everything goes as expected.
fnames <- list.files(test_dogs_dir, full.names = T)
img_path <- fnames[[17]]
img <- image_load(img_path, target_size = c(150,150))
img_tensor<- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1,150,150,3))
img_tensor <- img_tensor / 255
dim(img_tensor)
plot(as.raster(img_tensor[1,,,]))


# Create the first neural network model.
# The model expects its input to be of the format image height: 150, image width: 150, image channels: 3.
# The image entered as input is converted to a feature map, where each feature map is modified in
# several layers before it's finally estimated as a dog or non-dog.
model1 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(150,150,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# Report a summary of the newly created model.
# Using "layer_max_pooling_2d", the size of the feature map is reduced by its half, and its depth 
# assumes the value specified in the "filters" parameter.
# Because the final prediction is binary, dog / non dog, the output from the model becomes a 
# probability that assumes a value between 0-1.
summary(model1)

# The model is compiled. "loss" is set to "binary_crossentropy" because the outcome is binary.
model1 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)


# Before we train our model, we need to prepare the data, which in our case is images in the JPEG
# format. We start by reducing the size of all images by 1/255.
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

# Our training, validation and test data are transformed into a format that R can handle. This is done 
# using the keras function "flow_images_from_directory" which transforms our JPEG images into tensors
# that are approved as input to our newly created neural network model. 
# (One dimensional tensor = vector, two dimensional tensor = matrix, etc.).
# It's worth noting that "batch_size" splits the data into x parts. After a batch, the weights in our
# neural network model are altered and an epoch is completed when all the splits of the data are
# iterated once.
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150,150),
  batch_size = 70, 
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150,150),
  batch_size = 20,
  class_mode = "binary"
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150,150),
  batch_size = 40,
  class_mode = "binary"
)


# The first model is trained. For a regular laptop, the training took about 3.5 hours.
# If you don't want to wait so long, you can instead in row 202 import a pre-trained version of the
# model down below.

######################################################################################################
######################################################################################################

# Train the model.
history1 <- model1 %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

# Save the model.
model1 %>% save_model_hdf5("CatAndDog1.h5")

# Present the validation accuracy and the validation loss for each epoch.
plot(history1)

# Find out which epoch gave the highest validation accuracy.
which.max(history1$metrics$val_acc)

# Find out which epoch gave the lowest validation loss.
which.min(history1$metrics$val_loss)

######################################################################################################
######################################################################################################


# Import the first pre-trained model.
# In order for the import to work, you must save the file "CatAndDog1.h5" in your working directory.
# The "getwd()" function displays the searchpath to the working directory currently used.
model1 <- load_model_hdf5("CatAndDog1.h5")

# Predict the test dataset.
model1 %>% evaluate_generator(test_generator, steps = 50) # loss: 0.42-0.46   acc: approximately 0.805

# Estimate the probability that dog number 17 in the test dataset is a dog using the first model.
(model1 %>% predict(img_tensor))[,1]


####################################################################################################
# The first model predicts whether an image contains a dog or not correctly in about 80% of the cases.
# We now wish to increase the likelihood that a picture is correctly predicted by training a new model
# on several more epochs. This can lead to overfitting, i.e. that the model to a large extent follows
# the patterns learned from its training observations, and thus predicts pictures they never seen
# before poorly. However, in this model we will introduce two techniques that prevent overfitting,
# dropout and data augmentation.
# Dropout means that some output features randomly sets to zero during the training of the model and
# data augmentation creates even more training observations by modifying randomly selected training
# observations that are then included in the training set. 
# The more observations we use in our training set, the less risk of a model that overfits.
####################################################################################################


# Create the second neural network model.
model2 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(150,150,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# The model is compiled.
model2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

# Data augmentation is perfomed. We randomly modify selected orginal images.
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = T
)

# Include our modified images in the training set.
train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150,150),
  batch_size = 70,
  class_mode = "binary"
)

# The second model is trained. For a regular laptop, the training took about 26 hours.
# If you don't want to wait so long, you can instead in row 301 import a pre-trained version of the
# model down below.

######################################################################################################
######################################################################################################

# Train the model.
history2 <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

# Save the model.
model2 %>% save_model_hdf5("CatAndDog2.h5")

# Present the validation accuracy and the validation loss for each epoch.
plot(history2)

# Find out which epoch gave the highest validation accuracy.
which.max(history2$metrics$val_acc)

# Find out which epoch gave the lowest validation loss.
which.min(history2$metrics$val_loss)

######################################################################################################
######################################################################################################


# Import the second pre-trained model. 
# As earlier stated, note that the file "CatAndDog2.h5" must be saved in the current working directory.
model2 <- load_model_hdf5("CatAndDog2.h5")

# Predict the test dataset.
model2 %>% evaluate_generator(test_generator, steps = 50) # loss: 0.24-0.29  acc: approximately 0.885

# Estimate the probability that dog number 17 in the test dataset is a dog using the second model.
(model2 %>% predict(img_tensor))[,1]


####################################################################################################
# The second model predicts whether an image contains a dog or not correctly in about 88% of the 
# cases and is thus a better model than the first.
# If an even higher percentage of correct classifications is desired, the first step should be to 
# adjust the number of layers in the model, modify its parameters "filters" and "kernel_size" and 
# test training the model on even more epochs. Due to limited computer performance, this is not done
# in this project.
# Our model is created from scratch, however, it's common to use a large, already trained model that
# is then fine-tuned to optimize the classification ability.
# A natural second step is therefore to use a large pre-trained model and tune that model, so it e.g.
# classifies whether an image contains a dog or not in a good way.
####################################################################################################