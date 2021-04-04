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
# ------read in the dataset--------- #
solar_df = read.csv("SolarPrediction.csv")
head(solar_df)
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
solar_df$SunDuration = solar_df$SunsetHour + solar_df$SunsetMinute - solar_df$SunriseHour - solar_df$SunriseMinute
drop <- c("SunriseMinute", "SunriseHour", "SunsetMinute", "SunsetHour")
solar_df = solar_df[,!(names(solar_df) %in% drop)]
summary(solar_df)
# ---- normalize the dataset using min-max scaler ---- #
norm_minmax <- function(x){
  (x- min(x)) /(max(x)-min(x))
}

norm_df <- as.data.frame(lapply(solar_df, norm_minmax))
head(norm_df)

# ------ plot pairwise correlations for analysis ------ #
pairs(norm_df[, c(2, 3, 4, 5, 6, 12)], col='blue', pch=18, lower.panel = NULL, labels = c("Temp",
                                                                                          "Pressure", "Humidity", "Wind", "Speed", "SunTime"))
pairs.panels(norm_df[, c(2, 3, 4, 5, 6, 12)], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)
# ------ split data into train/test ------ #
norm_df.nrows <- nrow(norm_df)
norm_df.sample <- 0.8
norm_df.train.index <- sample(norm_df.nrows, norm_df.sample*norm_df.nrows)
norm_df.train <- norm_df[norm_df.train.index,]
norm_df.test <- norm_df[-norm_df.train.index,]
truth = norm_df[-norm_df.train.index,]
norm_df.test <-norm_df.test[,-grep("Radiation",colnames(norm_df.test))]
nrow(norm_df.train)
nrow(norm_df.test)

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
# rmse is worth without SOM

