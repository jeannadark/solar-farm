library("dplyr")
library("psych")
library("kohonen")
library("tidyverse")
library("mlbench")
library("caret")
library("caretEnsemble")
library("ggplot2")
library("stats")
library("gmodels")
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
norm_df.test <-norm_df.test[,-grep("Radiation",colnames(norm_df.test))]
truth = norm_df[-norm_df.train.index,]
nrow(norm_df.train)
nrow(norm_df.test)

# ------ perform SOM clustering on train dataset excluding dependent variable ---- #
set.seed(3100)
data_train_matrix <- as.matrix(norm_df.train[,c(2:12)])
som_grid <- somgrid(xdim = 4, ydim=4, topo="hexagonal")
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

# determine the optimal no of clusters
mydata <- som_model$codes 
mydata <- mydata[[1]]
par(mar = c(5,5,3,1))
library("factoextra")
fviz_nbclust(mydata, kmeans, method = "wss") +
  labs(subtitle = "Elbow method")

# use hierarchical clustering to cluster the code book vectors
pretty_palette <- c("#1f77b4", '#ff7f0e', '#2ca02c', '#655f60')
som_cluster <- cutree(hclust(dist(som_model$codes[[1]])), 3)
plot(som_model, type="mapping", bgcol = pretty_palette[som_cluster], main = "Clusters") 
add.cluster.boundaries(som_model, som_cluster) 
som_clusterKey <- data.frame(som_cluster)
som_clusterKey$unit.classif <- c(1:16)
norm_df.train <- cbind(norm_df.train, som_model$unit.classif,som_model$distances)
names(norm_df.train)[13] <- "unit.classif"
norm_df.train <- merge(norm_df.train, som_clusterKey, by.x = "unit.classif")
drop <- c("som_model$distances", "unit.classif")
norm_df.train = norm_df.train[,!(names(norm_df.train) %in% drop)]

# ----- train SVR model on each cluster --- #
library("e1071")
fits<-list(
  svm(Radiation ~ Temperature + Humidity + Minute + Second + SunDuration,  
      data = norm_df.train, subset = som_cluster==1, scale=FALSE),
  svm(Radiation ~ Temperature + WindDirection.Degrees. + Month + SunDuration,  
      data = norm_df.train, subset = som_cluster==2, scale=FALSE),
  svm(Radiation ~ Temperature + WindDirection.Degrees. + Month + SunDuration,  
      data = norm_df.train, subset = som_cluster==3, scale=FALSE)
)

# ---- predict SOM clusters for test data ---- #
som.prediction <- predict(som_model, newdata = as.matrix(norm_df.test))
preds <- som.prediction$unit.classif
norm_df.test <- cbind(norm_df.test, preds)
names(norm_df.test)[12] <- "unit.classif"
norm_df.test <- merge(norm_df.test, som_clusterKey, by.x = "unit.classif")
drop <- c("unit.classif")
norm_df.test = norm_df.test[,!(names(norm_df.test) %in% drop)]
head(norm_df.test)
truth <- cbind(truth, norm_df.test$som_cluster)
names(truth)[13] <- "som_cluster"

# --- predict radiation using SVR for test data ------ #

cluster_one_test = subset(norm_df.test, som_cluster==1)
cluster_two_test = subset(norm_df.test, som_cluster==2)
cluster_three_test = subset(norm_df.test, som_cluster==3)


cluster_one_truth = subset(truth, som_cluster==1)
cluster_two_truth = subset(truth, som_cluster==2)
cluster_three_truth = subset(truth, som_cluster==3)


cluster_one_preds = predict(fits[1], cluster_one_test[,c(1, 3, 9,10,11)])
cluster_two_preds = predict(fits[2], cluster_two_test[,c(1, 4, 6,11)])
cluster_three_preds = predict(fits[3], cluster_three_test[,c(1,4,6,11)])


cluster_one_test$predicted = cluster_one_preds[[1]]
cluster_two_test$predicted = cluster_two_preds[[1]]
cluster_three_test$predicted = cluster_three_preds[[1]]


RMSE(cluster_one_test$predicted, cluster_one_truth$Radiation)
RMSE(cluster_two_test$predicted, cluster_two_truth$Radiation)
RMSE(cluster_three_test$predicted, cluster_three_truth$Radiation)

nrow(subset(norm_df.train, som_cluster==1))
nrow(subset(norm_df.train, som_cluster==2))
nrow(subset(norm_df.train, som_cluster==3))

