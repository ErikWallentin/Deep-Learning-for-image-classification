######################################################################################################
# M�let med detta projekt �r att fr�n grunden tr�na en neural network-modell som p� ett bra s�tt 
# klassificerar om en bild inneh�ller en hund eller inte. F�r att lyckas med detta m�ste vi fr�n 
# Kaggle-anv�ndaren "chetanimravan" importerat en fil inneh�llandes 10 000 bilder d�r 5000 bilder �r 
# p� hundar och 5000 p� katter. Importera filen: https://www.kaggle.com/chetankv/dogs-cats-images
# (F�r att ladda ner data-setet m�ste man f�rst skapa ett kostnadsfrit konto p� Kaggle).

# Datamaterialet �r uppdelat i ett tr�nings-, validerings- samt test-set p� 7000, 1000 och 2000 bilder 
# d�r respektive set inneh�ller lika m�nga hund- som katt-bilder.
# Datamaterialet �r uppdelat i ett tr�nings- samt validerings-set s� man i realtid kan se hur v�l 
# modellen predikterar bilder den aldrig sett f�rut under tr�ningsfasen. Detta �r viktigt s� man t.ex.
# kan f� en ide om hur m�nga epocher man ska tr�na den slutgiltiga modellen som man sedan utv�rderar 
# med hj�lp av test-setet. P.g.a. prestandask�l skapas tv� modeller med f�rbest�mt antal epocher, 
# 30 samt 100. Modellen som tr�nas i 100 epocher anv�nder dock tv� metoder som f�rebygger overfiting. 

# Ett popul�rt paket som anv�nds vid m�nga depp learning-projekt �r keras, och det �r just det paketet 
# som anv�nds i detta bild-igenk�nnes-projekt. F�r att keras ska fungera som det ska beh�vs bland annat
# programmet "Anaconda" installerat p� anv�ndarens dator. Projektet b�rjar med att s�tta upp keras p� 
# ett korrekt s�tt, och anv�ndaren kan f�lja anvisningarna i R-konsolen om denne st�ter p� problem.

# N�r keras installerats �r n�sta steg att g�ra det m�jligt f�r R att l�sa av de 10 000 bilder som 
# inkluderas i projektet innan det �r dags f�r tr�ning samt utv�rdering av hur bra modellerna �r p� att
# klassificera en bild som hund eller icke hund. 
# Projektet avslutas med en kort reflektion �ver hur man skulle kunna skapa en modell som klassificerar
# en bild som hund alternativt icke hund med en �nnu h�gre tr�ffs�kerhet.
######################################################################################################


# Installera paketet "keras".
install.packages("keras")
library(keras)
install_keras() # Denna rad beh�ver endast k�ras f�rsta g�ngen keras-paketet h�mtas.

# G�r s� att R kan hitta bilderna som finns i filen "dataset".
# I kodrad 35 anger du s�kv�gen till vart du sparat filen "dataset".
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

length(list.files(validation_cats_dir)) # Om allt gick som det ska redovisas v�rdet 500.
length(list.files(validation_dogs_dir)) # Om allt gick som det ska redovisas v�rdet 500.


fnames <- paste0("cat.", 501:4000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(train_cats_dir))

fnames <- paste0("dog.", 501:4000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(train_dogs_dir))

length(list.files(train_cats_dir)) # Om allt gick som det ska redovisas v�rdet 3500.
length(list.files(train_dogs_dir)) # Om allt gick som det ska redovisas v�rdet 3500.

fnames <- paste0("cat.", 4001:5000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(test_cats_dir))

fnames <- paste0("dog.", 4001:5000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(test_dogs_dir))

length(list.files(test_cats_dir)) # Om allt gick som det ska redovisas v�rdet 1000.
length(list.files(test_dogs_dir)) # Om allt gick som det ska redovisas v�rdet 1000.


# Ett visuellt test att R kan hitta bilderna i filen "dataset" utf�rs nedan. 
# Bild nummer 17 i test-datasetet inneh�llandes hundar visualiseras om allt g�r som f�rv�ntat.
fnames <- list.files(test_dogs_dir, full.names = T)
img_path <- fnames[[17]]
img <- image_load(img_path, target_size = c(150,150))
img_tensor<- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1,150,150,3))
img_tensor <- img_tensor / 255
dim(img_tensor)
plot(as.raster(img_tensor[1,,,]))


# Skappa den f�rsta neural network-modellen.
# Modellen f�rv�ntar sig att dess input-data �r av formen bildens h�jd: 150, bildens bredd: 150, 
# bildens channels: 3. 
# Bilden man anger som input f�rv�ndlas till en "feature map", d�r varje feature map modifieras i flera
# lager innan den slutligen skattas som hund eller icke hund.
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

# Redovisa en summering av den nyskapade modellen. 
# Med hj�lp av "layer_max_pooling_2d" minskas stoleken p� feature mapen med h�lften, och dess djup
# antar v�rdet angivet i parametern "filters".
# Eftersom vi �r ute efter en bin�r slutprediktion, hund/icke hund, blir outputen fr�n modellen en 
# sannolikhet som antar ett v�rde mellan 0-1.
summary(model1)

# Modellen compileras. "loss" s�tts till "binary_crossentropy" eftersom utfallet �r bin�rt.
model1 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)


# Innan vi tr�nar v�r modell beh�ver vi f�rbereda datamaterialet som i v�rt fall �r bilder i formatet
# JPEG. B�rja med att f�rminska alla bilder med 1/255.
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

# V�rt tr�nings-, validerings- samt test-data formateras till ett format som R kan hantera. 
# Detta g�rs med hj�lp av keras-funktionen "flow_images_from_directory" som transformerar v�ra
# JPEG-bilder till tensorer som �r godk�nda som input till v�r nyskapade neural network-modell. 
# (Endimensionell tensor = vektor, tv�dimensionell tensor = matris osv.).
# V�rt att notera �r att "batch_size" delar upp datamaterialet i x delar. Efter en batch korrigeras
# vikterna i v�r neural network-modell och en epoch slutf�rs n�r alla uppdelningar av datamaterialet 
# itererats igenom en g�ng.
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


# Nedan tr�nas den f�rsta modellen. F�r en vanlig laptop tog tr�ningen ungef�r 3.5 timme.
# Vill du inte v�nta s� l�nge kan du ist�llet p� kodrad 202 importera en f�rtr�nad version av 
# nedanst�ende modell.

######################################################################################################
######################################################################################################

# Tr�na modellen.
history1 <- model1 %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

# Spara modellen.
model1 %>% save_model_hdf5("CatAndDog1.h5")

# Redovisa validation accuracy samt validation loss f�r varje epoch.
plot(history1)

# Ta reda p� vilken epoch som gav h�gst validation accuracy.
which.max(history1$metrics$val_acc)

# Ta reda p� vilken epoch som gav l�gst validation loss.
which.min(history1$metrics$val_loss)

######################################################################################################
######################################################################################################


# Importera den f�rsta f�rtr�nade modellen.
# F�r att importeringen ska fungera m�ste du spara den bifogade filen "CatAndDog1.h5" i din 
# working directory. Funktionen getwd()" visar s�kv�gen till den working directory som anv�nds.
model1 <- load_model_hdf5("CatAndDog1.h5")

# Prediktera test-datasetet.
model1 %>% evaluate_generator(test_generator, steps = 50) # loss: 0.42-0.46   acc: ungef�r 0.805

# Skatta sannolikheten att hund nummer 17 i test-datasetet �r en hund med hj�lp av den f�rsta modellen.
(model1 %>% predict(img_tensor))[,1]


####################################################################################################
# Den f�rsta modellen predikterar om en bild f�rest�ller en hund eller inte r�tt i ungef�r 80% av 
# fallen. 
# Vi �nskar nu f�rs�ka �ka sannolikheten att en bild predikteras r�tt genom att tr�na en ny modell i 
# fler epocher. Detta kan leda till att modellen overfitas, dvs att modellen till en f�r stor grad 
# f�ljer de m�nster de l�rt sig av tr�ningsobservationerna och s�ledes predikterar bilder de aldrig 
# sett tidigare p� ett d�ligt s�tt. Vi ska dock anv�nda oss utav tv� tekniker som f�rebygger 
# overfitting, dropout samt data augmentation. 
# Dropout inneb�r att man slumpm�ssigt s�tter vissa outputfeatures till noll under tr�ningen av 
# modellen och data augmentation handlar om att skapa �nnu fler tr�ningsobservationer genom att 
# modifiera slumpm�ssigt utvalda tr�ningsobservationer som sedan inkluderas i tr�nings-setet. Ju fler 
# observationer vi anv�nder i v�rt tr�nings-set, desto mindre risk att vi overfitar modellen.
####################################################################################################


# Skappa den andra neural network-modellen.
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

# Modellen compileras.
model2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

# Data augmentation utf�rs. Vi modifierar allts� n�gra slumpm�ssigt utvlada ursprungsbilder.
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = T
)

# Inkludera v�ra modifierade bilder i tr�nings-setet.
train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150,150),
  batch_size = 70,
  class_mode = "binary"
)

# Nedan tr�nas den andra modellen. F�r en vanlig laptop tog tr�ningen ungef�r 26 timmar.
# Vill du inte v�nta s� l�nge kan du ist�llet p� kodrad 302 importera en f�rtr�nad version av 
# nedanst�ende modell.

######################################################################################################
######################################################################################################

# Tr�na modellen.
history2 <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

# Spara  modellen.
model2 %>% save_model_hdf5("CatAndDog2.h5")

# Redovisa validation accuracy samt validation loss f�r varje epoch.
plot(history2)

# Ta reda p� vilken epoch som gav h�gst validation accuracy.
which.max(history2$metrics$val_acc)

# Ta reda p� vilken epoch som gav l�gst validation loss.
which.min(history2$metrics$val_loss)

######################################################################################################
######################################################################################################


# Importera den andra f�rtr�nade modellen. Notera som sagt att den bifogade filen "CatAndDog2.h5"
# m�ste sparas i nuvarnde working directory.
model2 <- load_model_hdf5("CatAndDog2.h5")

# Prediktera test-datasetet.
model2 %>% evaluate_generator(test_generator, steps = 50) # loss: 0.24-0.29  acc: ungef�r 0.885

# Skatta sannolikheten att hund nummer 17 i test-datasetet �r en hund med hj�lp av den andra modellen.
(model2 %>% predict(img_tensor))[,1]


####################################################################################################
# Den andra modellen predikterar om en bild f�rest�ller en hund eller inte r�tt i ungef�r 88% av 
# fallen och �r s�ledes en b�ttre modell �n den f�rsta.
# �nskas �nnu h�gre pricks�kerhet �r f�rsta steget att justera antalet layers i modellen, 
# modifiera dess parametrar "filters" samt "kernel_size" samt testa att tr�na modellen i �nnu fler
# epocher. P� grund av prestandask�l g�rs inte detta i projektet.
# V�r modell �r skapad fr�n grunden, det �r dock vanligt att man anv�nder sig av en  stor redan 
# f�rdigtr�nad modell som sedan finjusteras f�r att optimera klassifikationsf�rm�gan ifr�ga. 
# Ett naturligt andra steg �r d�rf�r att anv�nda grunderna en stor f�rdigtr�nad modell l�rt sig,
# och anpassa modellen s� att den p� ett bra s�tt klassificerar om en bild inneh�ller en hund eller ej.
####################################################################################################