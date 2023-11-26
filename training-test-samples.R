#PAQUETES...
require(e1071)
require(randomForest)
require(sp)
require(raster)
require(rgdal)
require(epicalc)
require(caret)
require(MLmetrics)
library(kernlab); #for spam data

library(caret)
library(e1071)
library(klaR)
library(rgdal)
library(raster)
library(caret)
library(plyr)

library(data.table)
library(mapview)
################################
library(tidyverse)
library(rgee)
library(sf)

###################################
img_2010 <- brick("C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM/Tubul-post.tif")

plot(img_2010, axes=FALSE)
datos_raster<-as.data.frame(img_2010)
names(img_2010) <- c(paste0("B", c(1:7)))  
#shapefile 
trainData10 <- shapefile("C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM/shapes/entrena-migra-test")
responseCol <- "class"

dfAll = data.frame(matrix(vector(), nrow = 0, ncol = length(names(img_2010)) + 1))   
for (i in 1:length(unique(trainData10[[responseCol]]))){
  category <- unique(trainData10[[responseCol]])[i]
  categorymap <- trainData10[trainData10[[responseCol]] == category,]
  dataSet <- raster::extract(img_2010, categorymap)
  if(is(trainData10, "SpatialPointsDataFrame")){
    dataSet <- cbind(dataSet, class = as.numeric(rep(category, nrow(dataSet))))
    dfAll <- rbind(dfAll, dataSet[complete.cases(dataSet),])
  }
  if(is(trainData10, "SpatialPolygonsDataFrame")){
    dataSet <- dataSet[!unlist(lapply(dataSet, is.null))]
    dataSet <- lapply(dataSet, function(x){cbind(x, class = as.numeric(rep(category, nrow(x))))})
    df <- do.call("rbind", dataSet)
    dfAll <- rbind(dfAll, df)
  }
}
new_df<- na.omit(dfAll)
#new_df <- subset(dfAll, dfAll[,8] != 0) 
new_df<-new_df[,-c(6)]

nsamples <- 1300
training <- new_df[sample(1:nrow(new_df), nsamples), ]
train_index <- createDataPartition(training$class,
                                   p = 0.08,
                                   list = FALSE,
                                   times = 1)

dat_train <- training[train_index,]
test_total <- training[-train_index,]

########################################################################
##################################################################
colnames(dat_train)<-c("B1","B2","B3","B4","B5","B6","Class")
dat_train$class <- as.factor(dat_train$Class)
dat_train$class <- mapvalues(dat_train$Class,
                             from = c(1,2,3,4,5),
                             to = c("claseA", "claseB", "claseC", "claseD","claseE"))

colnames(test_total)<-c("B1","B2","B3","B4","B5","B6","Class")
test_total$Class <- as.factor(test_total$Class)
test_total$Class <- mapvalues(test_total$Class,
                              from = c(1,2,3,4,5),
                              to = c("claseA", "claseB", "claseC", "claseD","claseE"))

dat_train<-dat_train[,-7]
#################################################
tam=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
train_in=list()
h=0
for (i in tam) {
  h=h+1
  train_index <- createDataPartition(dat_train$class,
                                     p = i,
                                     list = FALSE,
                                     times = 100)
  train_in[[h]]=train_index
}


entrena=list()
test=list()
dd=list()
ee=list()
hh=0
lk=data.frame(matrix(NA))
data_total = data.frame()
for (j in seq(1,length(train_in))) {
  hh=hh+1
  hhh=0
  for (jj in seq(1,ncol(train_index))) {
    hhh=hhh+1
    dt_train <- dat_train[train_in[[j]][,jj],]
    dt_test <- dat_train[-train_in[[j]][,jj],]
    df <- data.frame(matrix(unlist(dt_train), nrow=nrow(dt_train), byrow=F))
    colnames(df)<-c("B1","B2","B3","B4","B5","B6","Class")
    df$B1<-as.numeric(df$B1)
    df$B2<-as.numeric(df$B2)
    df$B3<-as.numeric(df$B3)
    df$B4<-as.numeric(df$B4)
    df$B5<-as.numeric(df$B5)
    df$B6<-as.numeric(df$B6)
    df$Class <- as.factor(df$Class)
    #    df$Class <- mapvalues(df$Class,
    #                          from = c(1,2,3,4,5),
    #                          to = c("claseA", "claseB", "claseC", "claseD", "claseE"))
    dd[[hhh]]=df
    
  }
  entrena[[hh]]=dd
  test[[hh]]=ee  
}
test_total2<-test_total
save(test_total2,file = "test_pos_migra2.RData")
