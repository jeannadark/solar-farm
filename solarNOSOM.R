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

# ----- train SVR model --- #
library("e1071")
fits<- svm(Radiation ~ Temperature + Pressure + Humidity + WindDirection.Degrees. + Speed + Month + Day + Hour + Minute + Second + SunDuration,  
      data = norm_df.train, scale=FALSE)


# --- predict radiation using SVR for test data ------ #

preds = predict(fits, norm_df.test[,c(1:11)])

norm_df.test$predicted = preds[[1]]

RMSE(norm_df.test$predicted, truth$Radiation)
