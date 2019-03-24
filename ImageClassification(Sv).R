######################################################################################################
# Målet med detta projekt är att från grunden träna en neural network-modell som på ett bra sätt 
# klassificerar om en bild innehåller en hund eller inte. För att lyckas med detta måste vi från 
# Kaggle-användaren "chetanimravan" importerat en fil innehållandes 10 000 bilder där 5000 bilder är 
# på hundar och 5000 på katter. Importera filen: https://www.kaggle.com/chetankv/dogs-cats-images
# (För att ladda ner data-setet måste man först skapa ett kostnadsfrit konto på Kaggle).

# Datamaterialet är uppdelat i ett tränings-, validerings- samt test-set på 7000, 1000 och 2000 bilder 
# där respektive set innehåller lika många hund- som katt-bilder.
# Datamaterialet är uppdelat i ett tränings- samt validerings-set så man i realtid kan se hur väl 
# modellen predikterar bilder den aldrig sett förut under träningsfasen. Detta är viktigt så man t.ex.
# kan få en ide om hur många epocher man ska träna den slutgiltiga modellen som man sedan utvärderar 
# med hjälp av test-setet. P.g.a. prestandaskäl skapas två modeller med förbestämt antal epocher, 
# 30 samt 100. Modellen som tränas i 100 epocher använder dock två metoder som förebygger overfiting. 

# Ett populärt paket som används vid många depp learning-projekt är keras, och det är just det paketet 
# som används i detta bild-igenkännes-projekt. För att keras ska fungera som det ska behövs bland annat
# programmet "Anaconda" installerat på användarens dator. Projektet börjar med att sätta upp keras på 
# ett korrekt sätt, och användaren kan följa anvisningarna i R-konsolen om denne stöter på problem.

# När keras installerats är nästa steg att göra det möjligt för R att läsa av de 10 000 bilder som 
# inkluderas i projektet innan det är dags för träning samt utvärdering av hur bra modellerna är på att
# klassificera en bild som hund eller icke hund. 
# Projektet avslutas med en kort reflektion över hur man skulle kunna skapa en modell som klassificerar
# en bild som hund alternativt icke hund med en ännu högre träffsäkerhet.
######################################################################################################


# Installera paketet "keras".
install.packages("keras")
library(keras)
install_keras() # Denna rad behöver endast köras första gången keras-paketet hämtas.

# Gör så att R kan hitta bilderna som finns i filen "dataset".
# I kodrad 35 anger du sökvägen till vart du sparat filen "dataset".
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

length(list.files(validation_cats_dir)) # Om allt gick som det ska redovisas värdet 500.
length(list.files(validation_dogs_dir)) # Om allt gick som det ska redovisas värdet 500.


fnames <- paste0("cat.", 501:4000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(train_cats_dir))

fnames <- paste0("dog.", 501:4000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(train_dogs_dir))

length(list.files(train_cats_dir)) # Om allt gick som det ska redovisas värdet 3500.
length(list.files(train_dogs_dir)) # Om allt gick som det ska redovisas värdet 3500.

fnames <- paste0("cat.", 4001:5000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(test_cats_dir))

fnames <- paste0("dog.", 4001:5000, ".jpg" )
file.copy(file.path(dir, fnames), file.path(test_dogs_dir))

length(list.files(test_cats_dir)) # Om allt gick som det ska redovisas värdet 1000.
length(list.files(test_dogs_dir)) # Om allt gick som det ska redovisas värdet 1000.


# Ett visuellt test att R kan hitta bilderna i filen "dataset" utförs nedan. 
# Bild nummer 17 i test-datasetet innehållandes hundar visualiseras om allt går som förväntat.
fnames <- list.files(test_dogs_dir, full.names = T)
img_path <- fnames[[17]]
img <- image_load(img_path, target_size = c(150,150))
img_tensor<- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1,150,150,3))
img_tensor <- img_tensor / 255
dim(img_tensor)
plot(as.raster(img_tensor[1,,,]))


# Skappa den första neural network-modellen.
# Modellen förväntar sig att dess input-data är av formen bildens höjd: 150, bildens bredd: 150, 
# bildens channels: 3. 
# Bilden man anger som input förvändlas till en "feature map", där varje feature map modifieras i flera
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
# Med hjälp av "layer_max_pooling_2d" minskas stoleken på feature mapen med hälften, och dess djup
# antar värdet angivet i parametern "filters".
# Eftersom vi är ute efter en binär slutprediktion, hund/icke hund, blir outputen från modellen en 
# sannolikhet som antar ett värde mellan 0-1.
summary(model1)

# Modellen compileras. "loss" sätts till "binary_crossentropy" eftersom utfallet är binärt.
model1 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)


# Innan vi tränar vår modell behöver vi förbereda datamaterialet som i vårt fall är bilder i formatet
# JPEG. Börja med att förminska alla bilder med 1/255.
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

# Vårt tränings-, validerings- samt test-data formateras till ett format som R kan hantera. 
# Detta görs med hjälp av keras-funktionen "flow_images_from_directory" som transformerar våra
# JPEG-bilder till tensorer som är godkända som input till vår nyskapade neural network-modell. 
# (Endimensionell tensor = vektor, tvådimensionell tensor = matris osv.).
# Värt att notera är att "batch_size" delar upp datamaterialet i x delar. Efter en batch korrigeras
# vikterna i vår neural network-modell och en epoch slutförs när alla uppdelningar av datamaterialet 
# itererats igenom en gång.
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


# Nedan tränas den första modellen. För en vanlig laptop tog träningen ungefär 3.5 timme.
# Vill du inte vänta så länge kan du istället på kodrad 202 importera en förtränad version av 
# nedanstående modell.

######################################################################################################
######################################################################################################

# Träna modellen.
history1 <- model1 %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

# Spara modellen.
model1 %>% save_model_hdf5("CatAndDog1.h5")

# Redovisa validation accuracy samt validation loss för varje epoch.
plot(history1)

# Ta reda på vilken epoch som gav högst validation accuracy.
which.max(history1$metrics$val_acc)

# Ta reda på vilken epoch som gav lägst validation loss.
which.min(history1$metrics$val_loss)

######################################################################################################
######################################################################################################


# Importera den första förtränade modellen.
# För att importeringen ska fungera måste du spara den bifogade filen "CatAndDog1.h5" i din 
# working directory. Funktionen getwd()" visar sökvägen till den working directory som används.
model1 <- load_model_hdf5("CatAndDog1.h5")

# Prediktera test-datasetet.
model1 %>% evaluate_generator(test_generator, steps = 50) # loss: 0.42-0.46   acc: ungefär 0.805

# Skatta sannolikheten att hund nummer 17 i test-datasetet är en hund med hjälp av den första modellen.
(model1 %>% predict(img_tensor))[,1]


####################################################################################################
# Den första modellen predikterar om en bild föreställer en hund eller inte rätt i ungefär 80% av 
# fallen. 
# Vi önskar nu försöka öka sannolikheten att en bild predikteras rätt genom att träna en ny modell i 
# fler epocher. Detta kan leda till att modellen overfitas, dvs att modellen till en för stor grad 
# följer de mönster de lärt sig av träningsobservationerna och således predikterar bilder de aldrig 
# sett tidigare på ett dåligt sätt. Vi ska dock använda oss utav två tekniker som förebygger 
# overfitting, dropout samt data augmentation. 
# Dropout innebär att man slumpmässigt sätter vissa outputfeatures till noll under träningen av 
# modellen och data augmentation handlar om att skapa ännu fler träningsobservationer genom att 
# modifiera slumpmässigt utvalda träningsobservationer som sedan inkluderas i tränings-setet. Ju fler 
# observationer vi använder i vårt tränings-set, desto mindre risk att vi overfitar modellen.
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

# Data augmentation utförs. Vi modifierar alltså några slumpmässigt utvlada ursprungsbilder.
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = T
)

# Inkludera våra modifierade bilder i tränings-setet.
train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150,150),
  batch_size = 70,
  class_mode = "binary"
)

# Nedan tränas den andra modellen. För en vanlig laptop tog träningen ungefär 26 timmar.
# Vill du inte vänta så länge kan du istället på kodrad 302 importera en förtränad version av 
# nedanstående modell.

######################################################################################################
######################################################################################################

# Träna modellen.
history2 <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

# Spara  modellen.
model2 %>% save_model_hdf5("CatAndDog2.h5")

# Redovisa validation accuracy samt validation loss för varje epoch.
plot(history2)

# Ta reda på vilken epoch som gav högst validation accuracy.
which.max(history2$metrics$val_acc)

# Ta reda på vilken epoch som gav lägst validation loss.
which.min(history2$metrics$val_loss)

######################################################################################################
######################################################################################################


# Importera den andra förtränade modellen. Notera som sagt att den bifogade filen "CatAndDog2.h5"
# måste sparas i nuvarnde working directory.
model2 <- load_model_hdf5("CatAndDog2.h5")

# Prediktera test-datasetet.
model2 %>% evaluate_generator(test_generator, steps = 50) # loss: 0.24-0.29  acc: ungefär 0.885

# Skatta sannolikheten att hund nummer 17 i test-datasetet är en hund med hjälp av den andra modellen.
(model2 %>% predict(img_tensor))[,1]


####################################################################################################
# Den andra modellen predikterar om en bild föreställer en hund eller inte rätt i ungefär 88% av 
# fallen och är således en bättre modell än den första.
# Önskas ännu högre pricksäkerhet är första steget att justera antalet layers i modellen, 
# modifiera dess parametrar "filters" samt "kernel_size" samt testa att träna modellen i ännu fler
# epocher. På grund av prestandaskäl görs inte detta i projektet.
# Vår modell är skapad från grunden, det är dock vanligt att man använder sig av en  stor redan 
# färdigtränad modell som sedan finjusteras för att optimera klassifikationsförmågan ifråga. 
# Ett naturligt andra steg är därför att använda grunderna en stor färdigtränad modell lärt sig,
# och anpassa modellen så att den på ett bra sätt klassificerar om en bild innehåller en hund eller ej.
####################################################################################################