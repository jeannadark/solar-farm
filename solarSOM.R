# sources to use are:
# 1. https://predictivehacks.com/how-to-build-stacked-ensemble-models-in-r/
# 2. https://www.polarmicrobes.org/tutorial-self-organizing-maps-in-r/
# 3. https://www.kaggle.com/sohelranaccselab/solar-radiation-prediction-using-ai
# 4. https://www.r-bloggers.com/2014/02/self-organising-maps-for-customer-segmentation-using-r/
# 5. https://iamciera.github.io/SOMexample/html/SOM_RNAseq_tutorial_part2a_SOM.html

library("dplyr")
library("psych")
library("kohonen")
library("tidyverse")
library("mlbench")
library("caret")
library("caretEnsemble")
library("ggplot2")
library("stats")
library("ggplot2") 
# ------read in the dataset--------- #
solar_df = read.csv("SolarPrediction.csv")
head(solar_df)
# sort in ascending order of date
solar_df = solar_df[order(as.Date(solar_df$Data, format = "%m/%d/%Y")),]
drop <- c("UNIXTime")
solar_df = solar_df[,!(names(solar_df) %in% drop)]
sum(is.na(solar_df))
# ------ perform engineering on time data ----- #
solar_df$Month = as.numeric(strftime(as.Date(solar_df$Data, "%m/%d/%Y"), "%m"))
solar_df$Day = as.numeric(strftime(as.Date(solar_df$Data, "%m/%d/%Y"), "%d"))
solar_df$Hour = as.numeric(format(strptime(solar_df$Time,"%H:%M:%S"),'%H'))
solar_df$Minute = as.numeric(format(strptime(solar_df$Time,"%H:%M:%S"),'%M'))
solar_df$Second = as.numeric(format(strptime(solar_df$Time,"%H:%M:%S"),'%S'))
solar_df$SunriseMinute = as.numeric(format(strptime(solar_df$TimeSunRise,"%H:%M:%S"),'%M'))
solar_df$SunriseHour = as.numeric(format(strptime(solar_df$TimeSunRise,"%H:%M:%S"),'%H'))
solar_df$SunsetMinute = as.numeric(format(strptime(solar_df$TimeSunSet,"%H:%M:%S"),'%M'))
solar_df$SunsetHour = as.numeric(format(strptime(solar_df$TimeSunSet,"%H:%M:%S"),'%H'))
drop <- c("Time", "Data", "TimeSunRise", "TimeSunSet")
solar_df = solar_df[,!(names(solar_df) %in% drop)]
solar_df$SunDuration = solar_df$SunsetHour*60 + solar_df$SunsetMinute - solar_df$SunriseHour*60 - solar_df$SunriseMinute
drop <- c("SunriseMinute", "SunriseHour", "SunsetMinute", "SunsetHour")
solar_df = solar_df[,!(names(solar_df) %in% drop)]
summary(solar_df)

# ------ remove outliers --------- #

boxplot(solar_df[c(1, 2, 3, 4, 5, 6, 12)], boxwex=0.4, main='Boxplot', ylab='value', col=c('green'), las=2,
        names = c('Radiation', 'Temp', 'Pressure', 'Humidity',
                  'Wind Dir.', 'Speed', 'Sun Dur.'))

outliers_radiation <- boxplot(solar_df$Radiation, plot = FALSE)$out
outliers_wind_dir <- boxplot(solar_df$WindDirection.Degrees., plot = FALSE)$out
transformed_df = solar_df[-c(which(solar_df$Radiation %in% outliers_radiation)), ]
transformed_df = transformed_df[-c(which(transformed_df$WindDirection.Degrees. %in% outliers_wind_dir)), ]

# ---- normalize the dataset using min-max scaler ---- #
norm_minmax <- function(x){
  (x- min(x)) /(max(x)-min(x))
}

norm_df <- as.data.frame(lapply(transformed_df, norm_minmax))
head(norm_df)

# ------ plot pairwise correlations for analysis ------ #
library("corrplot")
correlations <- cor(norm_df)
corrplot(correlations, method="circle")


# ----- split into train / test -------- #
norm_df.nrows <- nrow(norm_df)
norm_df.sample <- 0.7
norm_df.train.index <- c(1:as.integer(norm_df.sample*norm_df.nrows))
norm_df.train <- norm_df[norm_df.train.index,]
norm_df.test <- norm_df[-norm_df.train.index,]
truth = norm_df[-norm_df.train.index,]
norm_df.test <-norm_df.test[,-grep("Radiation",colnames(norm_df.test))]
nrow(norm_df.train)
nrow(norm_df.test)

# ------ perform SOM clustering on whole dataset ---- #
data_train_matrix <- as.matrix(norm_df.train)
som_grid <- somgrid(xdim = 5, ydim=5, topo="hexagonal")
som_model <- som(data_train_matrix, 
                 grid=som_grid, 
                 rlen=100, 
                 alpha=c(0.05,0.01), 
                 keep.data = TRUE)
plot(som_model, type="changes")
# visualise the count of how many samples are mapped to each node on the map. 
# This metric can be used as a measure of map quality
# ideally the sample distribution is relatively uniform
plot(som_model, type="count")
# areas of low neighbour distance indicate groups of nodes that are similar. 
# Areas with large distances indicate the nodes are much more dissimilar
plot(som_model, type="dist.neighbours")
# individual fan representations of the magnitude of each variable in the weight vector is shown for each node
plot(som_model, type="codes")
var <- 1 #define the variable to plot 
var_unscaled <- aggregate(as.numeric(norm_df[,var]), by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,1] 
plot(som_model, type = "property", property=var_unscaled, main=names(norm_df)[var])

# determine the optimal no of clusters
mydata <- som_model$codes 
mydata <- mydata[[1]]
par(mar = c(5,5,3,1))
library("factoextra")
fviz_nbclust(mydata, kmeans, method = "wss") +
  labs(subtitle = "Elbow method")


# ----- till here --- #
# train the LSTM on each cluster or on train with cluster info
# predict using som
# add cluster info to test set after prediction
# predict using lstm for cluster
# analyze the results (rmse, mape)

# use hierarchical clustering to cluster the code block vectors
som_cluster <- cutree(hclust(dist(som_model$codes[[1]])), 3)
plot(som_model, type="mapping", bgcol = som_cluster, main = "Clusters") 
add.cluster.boundaries(som_model, som_cluster) 
som_clusterKey <- data.frame(som_cluster)
som_clusterKey$unit.classif <- c(1:25)
norm_df.train <- cbind(norm_df.train, som_model$unit.classif,som_model$distances)
names(norm_df.train)[13] <- "unit.classif"
norm_df.train <- merge(norm_df.train, som_clusterKey, by.x = "unit.classif")
drop <- c("som_model$distances", "unit.classif")
norm_df.train = norm_df.train[,!(names(norm_df.train) %in% drop)]
# out <- split(norm_df, f = norm_df$som_cluster)


# ----- build a stacked ensemble model ------ #
fitControl <- trainControl(method = "repeatedcv",   
                           number = 3,     # number of folds
                           repeats = 5)    # repeated five times
model.cv <- train(Radiation ~ .,
                  data = norm_df.train,
                  method = "gbm",
                  trControl = fitControl) 
model.cv
# ----- view the results ------ #
predictions <- predict(model.cv, norm_df.test)
norm_df.test$predictedRadiation = as.factor(predictions)
RMSE(predictions, truth$Radiation)
library("MLmetrics")
MAPE(predictions, truth$Radiation)


