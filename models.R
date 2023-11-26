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
load("C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/test2_ndvi_part4.RData")
load("C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/test_real_ndvi4.RData")
load("D:/test2_ndvi_part3.RData")
load("D:/test_real_ndvi2.RData")
#####################################################
n_folds <- 10
#set.seed(321)
folds <- createFolds(1:length(entrena[[1]]), k = n_folds)


fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
######################################
#####################################
##############################
##RF
# define a grid of parameter options to try
rf_grid <- expand.grid(mtry = c(2))
#rf_grid <- expand.grid(mtry = c(1,2,3,4,5))

mod_rf_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_rf1<-list()
cf_rf11<-list()
ll=0
pb=txtProgressBar(min=0,max = 20,style = 2)
for (l in seq(1,10)) {
  Sys.sleep(0.1)
  ll=ll+1
  mod_rf_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_rf<-list()
  cf_rf1<-list()
  kkk=0
  for (kk in seq(1,10)) {
    kkk=kkk+1
    model_rf <- caret::train(Class ~ . , method = 'rf', data = entrena[[l]][[kk]],
                             allowParallel = TRUE,
                             tuneGrid = rf_grid)
    #    op<-which(model_svm$results["mtry"]==as.numeric(model_svm$bestTune))
    accu<-model_rf$results[as.numeric(rownames(model_rf$bestTune)),"Accuracy"]
    
    mod_rf_fin[[kkk]]=model_rf
    param_opti[[kkk]]<-as.numeric(model_rf$bestTune)
    accu_opti1[[kkk]]<-accu
    prediccion<-confusionMatrix(predict( model_rf,test_total10),as.factor(test_total10$Class))
    prec<-prediccion$overall[1]
    prec_rf[[kkk]]<-prec
    cf_rf<-prediccion$table
    cf_rf1[[kkk]]<-cf_rf
    
    #    accu_opti<-rbind(accu_opti,accu)
  }
  mod_rf_fin1[[ll]]=mod_rf_fin
  param_opti1[[ll]]=param_opti
  accu_opti11[[ll]]=accu_opti1
  df_accu_opti<-cbind(df_accu_opti,accu_opti1)
  prec_rf1[[ll]]=prec_rf
  cf_rf11[[ll]]=cf_rf1
  setTxtProgressBar(pb,l)
}
close(pb)


precision<-rbindlist(prec_rf1)
precision2<-data.frame(t(precision))
promedio2<-data.frame(matrix(NA))
stde2<-data.frame(matrix(NA))
maxii<-data.frame(matrix(NA))
#param_max<-data.frame(matrix(NA))
for (s in seq(1,ncol(precision2))) {
  prom<-mean(precision2[,s])
  sd<-sd(precision2[,s])
  maxi<-which.max(precision2[,s])
  maxii<-cbind(maxii,maxi)
  #p_max<-parametro2[maxi,s]
  promedio2<-cbind(promedio2,prom)
  stde2<-cbind(stde2,sd)
  # param_max<-cbind(param_max,p_max)
}
promedio2<-promedio2[-1]
max_total<-which.max(promedio2)
maxii<-maxii[-1]

prom_rf<-promedio2
std_rf<-stde2[-1]
cm_rf <- confusionMatrix(data = predict(mod_rf_fin1[[10]][[as.numeric(2)]], newdata =test_total10),
                         test_total10$Class)

mod_rf<-mod_rf_fin1[[10]][[as.numeric(2)]]
save(mod_rf_fin1,mod_rf,prom_rf,std_rf,cm_rf,file = "C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/mod_rf2_ndvi_part4.RData")

############################################

svm_gridR <- expand.grid(sigma = c(0.25),C = c(2^3))

mod_svm_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_svm1<-list()
cf_svm11<-list()
ll=0
pb=txtProgressBar(min=0,max = 20,style = 2)
for (l in seq(1,10)) {
  Sys.sleep(0.1)
  ll=ll+1
  mod_svm_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_svm<-list()
  cf_svm1<-list()
  kkk=0
  for (kk in seq(1,5)) {
    kkk=kkk+1
    model_svm <- caret::train(Class ~ . , method = 'svmRadial', data = entrena[[l]][[kk]],
                              allowParallel = TRUE,
                              tuneGrid = svm_gridR)
    #    op<-which(model_svm$results["mtry"]==as.numeric(model_svm$bestTune))
    accu<-model_svm$results[as.numeric(rownames(model_svm$bestTune)),"Accuracy"]
    
    mod_svm_fin[[kkk]]=model_svm
    param_opti[[kkk]]<-as.numeric(model_svm$bestTune)
    accu_opti1[[kkk]]<-accu
    prediccion<-confusionMatrix(predict( model_svm,test_total10),as.factor(test_total10$Class))
    prec<-prediccion$overall[1]
    prec_svm[[kkk]]<-prec
    cf_svm<-prediccion$table
    cf_svm1[[kkk]]<-cf_svm
    
    #    accu_opti<-rbind(accu_opti,accu)
  }
  mod_svm_fin1[[ll]]=mod_svm_fin
  param_opti1[[ll]]=param_opti
  accu_opti11[[ll]]=accu_opti1
  df_accu_opti<-cbind(df_accu_opti,accu_opti1)
  prec_svm1[[ll]]=prec_svm
  cf_svm11[[ll]]=cf_svm1
  setTxtProgressBar(pb,l)
}
close(pb)

precision<-rbindlist(prec_svm1)
precision2<-data.frame(t(precision))
promedio2<-data.frame(matrix(NA))
stde2<-data.frame(matrix(NA))
maxii<-data.frame(matrix(NA))
#param_max<-data.frame(matrix(NA))
for (s in seq(1,ncol(precision2))) {
  prom<-mean(precision2[,s])
  sd<-sd(precision2[,s])
  maxi<-which.max(precision2[,s])
  maxii<-cbind(maxii,maxi)
  #p_max<-parametro2[maxi,s]
  promedio2<-cbind(promedio2,prom)
  stde2<-cbind(stde2,sd)
  # param_max<-cbind(param_max,p_max)
}
promedio2<-promedio2[-1]
max_total<-which.max(promedio2)
maxii<-maxii[-1]

prom_svm<-promedio2
std_svm<-stde2[-1]
cm_svm <- confusionMatrix(data = predict(mod_svm_fin1[[2]][[as.numeric(4)]], newdata =test_total10),
                          test_total10$Class)

mod_svm<-mod_svm_fin1[[2]][[as.numeric(4)]]
save(cm_svm,mod_svm,mod_svm_fin1,prom_svm,std_svm,file = "mod_svm_part3.RData")
################################################

########KNN
# define a grid of parameter options to try

# Hiperparámetros
#hiperparametros <- data.frame(k = c(1,2,3,4,5,6,7,8,9,10))
hiperparametros <- data.frame(k = c(2))

mod_knn_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_knn1<-list()
cf_knn11<-list()
ll=0
pb <- txtProgressBar(min = 0, max = 20, style = 3)
for (l in seq(1,10)) {
  Sys.sleep(0.1)
  ll=ll+1
  mod_knn_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_knn<-list()
  cf_knn1<-list()
  kkk=0
  for (kk in seq(1,5)) {
    kkk=kkk+1
    model_knn <- caret::train(Class ~ ., data = entrena[[l]][[kk]],
                              method = "knn",
                              tuneGrid = hiperparametros,
                              metric = "Accuracy")
    op<-which(model_knn$results["k"]==as.numeric(model_knn$bestTune))
    accu<-model_knn$results[op,"Accuracy"]
    prediccion<-confusionMatrix(predict( model_knn,test_total10),as.factor(test_total10$Class))
    prec<-prediccion$overall[1]
    prec_knn[[kkk]]<-prec
    cf_knn<-prediccion$table
    cf_knn1[[kkk]]<-cf_knn
    mod_knn_fin[[kkk]]=model_knn
    param_opti[[kkk]]<-as.numeric(model_knn$bestTune)
    accu_opti1[[kkk]]<-accu
    #    accu_opti<-rbind(accu_opti,accu)
  }
  mod_knn_fin1[[ll]]=mod_knn_fin
  param_opti1[[ll]]=param_opti
  accu_opti11[[ll]]=accu_opti1
  df_accu_opti<-cbind(df_accu_opti,accu_opti1)
  prec_knn1[[ll]]=prec_knn
  cf_knn11[[ll]]=cf_knn1
  
  setTxtProgressBar(pb, l)
}
close(pb)

precision<-rbindlist(prec_knn1)
precision2<-data.frame(t(precision))
promedio2<-data.frame(matrix(NA))
stde2<-data.frame(matrix(NA))
maxii<-data.frame(matrix(NA))
#param_max<-data.frame(matrix(NA))
for (s in seq(1,ncol(precision2))) {
  prom<-mean(precision2[,s])
  sd<-sd(precision2[,s])
  maxi<-which.max(precision2[,s])
  maxii<-cbind(maxii,maxi)
  #p_max<-parametro2[maxi,s]
  promedio2<-cbind(promedio2,prom)
  stde2<-cbind(stde2,sd)
  # param_max<-cbind(param_max,p_max)
}
promedio2<-promedio2[-1]
max_total<-which.max(promedio2)
maxii<-maxii[-1]

cm_knn <- confusionMatrix(data = predict(mod_knn_fin1[[10]][[as.numeric(2)]], newdata =test_total10),
                          test_total10$Class)


prom_knn<-promedio2
std_knn<-stde2[-1]
mod_knn<-mod_knn_fin1[[7]][[as.numeric(2)]]
save(mod_knn,mod_knn_fin1,prom_knn,std_knn,file = "mod_knn2_ndvi_part2.RData")

##############################################################
######DNN

point <- entrena[[10]][[3]]
point$Class<- as.factor(point$Class)
test_total10$Class<- as.factor(test_total10$Class)

point$Class <- mapvalues(point$Class,
                         from = c("claseA", "claseB"),
                         to = c(1,2))

test_total10$Class <- mapvalues(test_total10$Class,
                                from = c("claseA", "claseB"),
                                to = c(1,2))

##inicializar el cluster h2o
library(h2o)
localH2o <- h2o.init(nthreads = -1, max_mem_size = "50G")
####
df<- as.h2o(point)
df$Class<-as.factor(df$Class)
df_val<- as.h2o(test_total10)
df_val$Class<-as.factor(df_val$Class)


hyper_params<- list(
  activation = "RectifierWithdfDropout",
  hidden = c(100,100),
  epochs = c(200),
  l1=1e-6, # L1/L2 regularization, improve generalization
  l2=1e-6,
  rho = c(0.99),
  epsilon = c( 1e-6),
  max_w2=10)

hyper_params<- list(
  activation = "RectifierWithDropout",
  hidden = c(100,100),
  epochs = c(100),
  l1=1e-6, # L1/L2 regularization, improve generalization
  l2=1e-6,
  rho = c(0.99,0.99,0.99,0.99),
  epsilon = c( 1e-6),
  max_w2=10)


search_criteria <- list(strategy = "RandomDiscrete", 
                        max_models = 100,
                        max_runtime_secs = 900,
                        stopping_tolerance = 0.001,
                        stopping_rounds = 15,
                        
                        seed = 42)

tam=c(0.99)
train=list()
valid=list()
train1<-list()
valid1<-list()
modelo<-list()
h=0
hiden1<-data.frame(matrix(NA))
epo1<-data.frame(matrix(NA))
max_acu1<-data.frame(matrix(NA))
for (i in tam) {
  h=h+1
  splits <- h2o.splitFrame(df, c(i), seed=1234)
  train <- h2o.assign(splits[[1]], "train.hex")
  valid <- h2o.assign(splits[[2]], "valid.hex") # 12%
  
  train1[[h]]<-train
  valid1[[h]]<-valid
  #total=list(train_index)
  
  y <- "Class"
  x <- setdiff(names(train), y)
  
  dl_model <- h2o.grid(algorithm = "deeplearning", 
                       x = x,
                       y = y,
                       grid_id = "dl_grid",
                       training_frame = train,
                       validation_frame = df_val,
                       nfolds = 50,                           
                       fold_assignment = "Stratified",
                       hyper_params = hyper_params,
                       search_criteria = search_criteria,
                       seed = 42
  )
  d_grid <- h2o.getGrid("dl_grid", sort_by = "accuracy", decreasing = TRUE)
  best_model<-h2o.getModel(d_grid@model_ids[[1]])
  
  h2o.saveModel(object = best_model, path = "C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2", force = TRUE) # force overwriting
  name <- file.path( "C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2", best_model@model_id) # destination file name at the same folder location
  file.rename(file.path("C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/", best_model@model_id), paste0(c('modelo_DNN_v2_part4'),1))
  
}

mean(as.numeric(d_grid@summary_table$accuracy))

var(as.numeric(d_grid@summary_table$accuracy))

modelo10<-h2o.loadModel(path = "C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/modelo_DNN1_v210000")
h2o.removeAll()
h2o.shutdown(prompt=FALSE)




#############################################################3
# Hiperparámetros
xgb_grid = expand.grid(
  nrounds = c(200), 
  max_depth = c(10),
  eta = c(0.002),
  gamma = c(0.5), 
  colsample_bytree = 0.9, 
  min_child_weight = c(1),
  subsample = c(1)
)



mod_xgb_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_xgb1<-list()
cf_xgb11<-list()
ll=0

for (l in seq(1,10)) {
  
  ll=ll+1
  mod_xgb_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_xgb<-list()
  cf_xgb1<-list()
  kkk=0
  pb <- txtProgressBar(min = 0, max = 20, style = 3)
  for (kk in seq(1,3)) {
    Sys.sleep(0.1)
    kkk=kkk+1
    model_xgb = caret::train(x = entrena[[l]][[kk]][,-8], 
                             y = entrena[[l]][[kk]][,8],
                             method = "xgbTree",
                             trControl = fitControl,
                             tuneGrid = xgb_grid,
                             metric = "Accuracy")
    
    a<-model_xgb$results[which.max(model_xgb$results$Accuracy),]
    accu<-a$Accuracy
    prediccion<-confusionMatrix(predict( model_xgb,test_total10),as.factor(test_total10$Class))
    prec<-prediccion$overall[1]
    prec_xgb[[kkk]]<-prec
    cf_xgb<-prediccion$table
    cf_xgb1[[kkk]]<-cf_xgb
    mod_xgb_fin[[kkk]]=model_xgb
    param_opti[[kkk]]<-as.numeric(model_xgb$bestTune)
    accu_opti1[[kkk]]<-accu
    accu_opti<-rbind(accu_opti,accu)
    setTxtProgressBar(pb, kk)
  }
  close(pb)
  
  mod_xgb_fin1[[ll]]=mod_xgb_fin
  param_opti1[[ll]]=param_opti
  accu_opti11[[ll]]=accu_opti1
  df_accu_opti<-cbind(df_accu_opti,accu_opti1)
  prec_xgb1[[ll]]=prec_xgb
  cf_xgb11[[ll]]=cf_xgb1
  
  
}

precision<-rbindlist(prec_xgb1)
precision2<-data.frame(t(precision))
promedio2<-data.frame(matrix(NA))
stde2<-data.frame(matrix(NA))
maxii<-data.frame(matrix(NA))
#param_max<-data.frame(matrix(NA))
for (s in seq(1,ncol(precision2))) {
  prom<-mean(precision2[,s])
  sd<-sd(precision2[,s])
  maxi<-which.max(precision2[,s])
  maxii<-cbind(maxii,maxi)
  #p_max<-parametro2[maxi,s]
  promedio2<-cbind(promedio2,prom)
  stde2<-cbind(stde2,sd)
  # param_max<-cbind(param_max,p_max)
}
promedio2<-promedio2[-1]
max_total<-which.max(promedio2)
maxii<-maxii[-1]

cm_xgb <- confusionMatrix(data = predict(mod_xgb_fin1[[4]][[as.numeric(1)]], newdata =test_total10),
                          test_total10$Class)

mod_xgb<-mod_xgb_fin1[[4]][[as.numeric(1)]]
save(cm_xgb,mod_xgb,mod_xgb_fin1,file = "modelo_xgb_part3.RData")

############################################################
#############GBM                   ########################
##########################################################
# Grid of tuning parameters
#Creating grid
gbm_grid <- expand.grid(n.trees=c(500),shrinkage=c(0.5),n.minobsinnode = c(10),interaction.depth=c(10))

mod_gbm_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_gbm1<-list()
cf_gbm11<-list()
ll=0
for (l in  seq(6,10)) {
  
  ll=ll+1
  mod_gbm_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_gbm<-list()
  cf_gbm1<-list()
  kkk=0
  pb <- txtProgressBar(min = 0, max = 20, style = 3)
  for (kk in seq(1,3)) {
    Sys.sleep(0.1)
    kkk=kkk+1
    model_gbm <- caret::train(Class ~.,method = "gbm", data = entrena[[l]][[kk]],
                              tuneGrid = gbm_grid,
                              trControl=fitControl)
    #op<-which(model_rf$results["mtry"]==as.numeric(model_rf$bestTune))
    a<-model_gbm$results[which.max(model_gbm$results$Accuracy),]
    accu<-a$Accuracy
    prediccion<-confusionMatrix(predict(model_gbm,test_total10),as.factor(test_total10$Class))
    prec<-prediccion$overall[1]
    prec_gbm[[kkk]]<-prec
    cf_gbm<-prediccion$table
    cf_gbm1[[kkk]]<-cf_gbm
    #accu<-model_gbm$results[as.numeric(rownames(model_gbm$bestTune)),"Accuracy"]
    #accu<-model_rf$results[op,"Accuracy"]
    
    mod_gbm_fin[[kkk]]=model_gbm
    param_opti[[kkk]]<-as.numeric(model_gbm$bestTune)
    accu_opti1[[kkk]]<-accu
    setTxtProgressBar(pb, l)
    #    accu_opti<-rbind(accu_opti,accu)
  }
  mod_gbm_fin1[[ll]]=mod_gbm_fin
  param_opti1[[ll]]=param_opti
  accu_opti11[[ll]]=accu_opti1
  df_accu_opti<-cbind(df_accu_opti,accu_opti1)
  prec_gbm1[[ll]]=prec_gbm
  cf_gbm11[[ll]]=cf_gbm1
}
close(pb)

precision<-rbindlist(prec_gbm1)
precision2<-data.frame(t(precision))
promedio2<-data.frame(matrix(NA))
stde2<-data.frame(matrix(NA))
maxii<-data.frame(matrix(NA))
#param_max<-data.frame(matrix(NA))
for (s in seq(1,ncol(precision2))) {
  prom<-mean(precision2[,s])
  sd<-sd(precision2[,s])
  maxi<-which.max(precision2[,s])
  maxii<-cbind(maxii,maxi)
  #p_max<-parametro2[maxi,s]
  promedio2<-cbind(promedio2,prom)
  stde2<-cbind(stde2,sd)
  # param_max<-cbind(param_max,p_max)
}
promedio2<-promedio2[-1]
max_total<-which.max(promedio2)
maxii<-maxii[-1]

cm_gbm <- confusionMatrix(data = predict(mod_gbm_fin1[[2]][[as.numeric(2)]], newdata =test_total10),
                          test_total10$Class)

mod_gbm2<-mod_gbm_fin1[[2]][[as.numeric(2)]]
prom_gbm2<-promedio2
std_gbm2<-stde2[-1]
mod_gbm_parte2<-mod_gbm_fin1
save(cm_gbm,mod_gbm2,mod_gbm_parte2,prom_gbm2,std_gbm2,file = "mod_gbm_part3-2.RData")

#############################################################
#######           MARS                   ####################
#############################################################
#hiperparametros <- data.frame(k = c(1,2,3,4,5,6,7,8,9,10))
hiperparametros <- expand.grid(degree = c(2),nprune=c(20))

mod_mars_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_mars1<-list()
cf_mars11<-list()
ll=0
pb <- txtProgressBar(min = 0, max = 20, style = 3)
for (l in seq(1,10)) {
  Sys.sleep(0.1)
  ll=ll+1
  mod_mars_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_mars<-list()
  cf_mars1<-list()
  kkk=0
  for (kk in seq(1,3)) {
    kkk=kkk+1
    model_mars <- caret::train(Class ~ ., data = entrena[[l]][[kk]],
                               method = "earth",
                               tuneGrid = hiperparametros,
                               metric = "Accuracy")
    a<-model_mars$results[which.max(model_mars$results$Accuracy),]
    accu<-a$Accuracy
    prediccion<-confusionMatrix(predict( model_mars,test_total10),as.factor(test_total10$Class))
    prec<-prediccion$overall[1]
    prec_mars[[kkk]]<-prec
    cf_mars<-prediccion$table
    cf_mars1[[kkk]]<-cf_mars
    mod_mars_fin[[kkk]]=model_mars
    param_opti[[kkk]]<-as.numeric(model_mars$bestTune)
    accu_opti1[[kkk]]<-accu
    #    accu_opti<-rbind(accu_opti,accu)
  }
  mod_mars_fin1[[ll]]=mod_mars_fin
  param_opti1[[ll]]=param_opti
  accu_opti11[[ll]]=accu_opti1
  df_accu_opti<-cbind(df_accu_opti,accu_opti1)
  prec_mars1[[ll]]=prec_mars
  cf_mars11[[ll]]=cf_mars1
  
  setTxtProgressBar(pb, l)
}
close(pb)

precision<-rbindlist(prec_mars1)
precision2<-data.frame(t(precision))
promedio2<-data.frame(matrix(NA))
stde2<-data.frame(matrix(NA))
maxii<-data.frame(matrix(NA))
#param_max<-data.frame(matrix(NA))
for (s in seq(1,ncol(precision2))) {
  prom<-mean(precision2[,s])
  sd<-sd(precision2[,s])
  maxi<-which.max(precision2[,s])
  maxii<-cbind(maxii,maxi)
  #p_max<-parametro2[maxi,s]
  promedio2<-cbind(promedio2,prom)
  stde2<-cbind(stde2,sd)
  # param_max<-cbind(param_max,p_max)
}
promedio2<-promedio2[-1]
max_total<-which.max(promedio2)
maxii<-maxii[-1]

cm_mars <- confusionMatrix(data = predict(mod_mars_fin1[[4]][[as.numeric(3)]], newdata =test_total10),
                           test_total10$Class)

mod_mars<-mod_mars_fin1[[4]][[as.numeric(2)]]
save(cm_mars,mod_mars,file = "modelo_mars_part3.RData")

prom_mars<-promedio2
std_mars<-stde2[-1]
save(mod_mars_fin1,prom_mars,std_mars,file = "mod_mars10_posA.RData")
###################################################################################3
###############################################################################
#DNN

point <- entrena[[10]][[3]]
point$Class<- as.factor(point$Class)
test_total10$Class<- as.factor(test_total10$Class)

point$Class <- mapvalues(point$Class,
                         from = c("claseA", "claseB"),
                         to = c(1,2))

test_total10$Class <- mapvalues(test_total10$Class,
                                from = c("claseA", "claseB"),
                                to = c(1,2))

##inicializar el cluster h2o
library(h2o)
localH2o <- h2o.init(nthreads = -1, max_mem_size = "50G")
####
df<- as.h2o(point)
df$Class<-as.factor(df$Class)
df_val<- as.h2o(test_total10)
df_val$Class<-as.factor(df_val$Class)


hyper_params<- list(
  activation = "RectifierWithdfDropout",
  hidden = c(50,50),
  epochs = c(200),
  l1=1e-6, # L1/L2 regularization, improve generalization
  l2=1e-6,
  rho = c(0.999),
  epsilon = c( 1e-6),
  max_w2=10)

search_criteria <- list(strategy = "RandomDiscrete", 
                        max_models = 100,
                        max_runtime_secs = 900,
                        stopping_tolerance = 0.001,
                        stopping_rounds = 15,
                        
                        seed = 42)

tam=c(0.99)
train=list()
valid=list()
train1<-list()
valid1<-list()
modelo<-list()
h=0
hiden1<-data.frame(matrix(NA))
epo1<-data.frame(matrix(NA))
max_acu1<-data.frame(matrix(NA))
for (i in tam) {
  h=h+1
  splits <- h2o.splitFrame(df, c(i), seed=1234)
  train <- h2o.assign(splits[[1]], "train.hex")
  valid <- h2o.assign(splits[[2]], "valid.hex") # 12%
  
  train1[[h]]<-train
  valid1[[h]]<-valid
  #total=list(train_index)
  
  y <- "Class"
  x <- setdiff(names(train), y)
  
  dl_model <- h2o.grid(algorithm = "deeplearning", 
                       x = x,
                       y = y,
                       grid_id = "dl_grid",
                       training_frame = train,
                       validation_frame = df_val,
                       nfolds = 25,                           
                       fold_assignment = "Stratified",
                       hyper_params = hyper_params,
                       search_criteria = search_criteria,
                       seed = 42
  )
  d_grid <- h2o.getGrid("dl_grid", sort_by = "accuracy", decreasing = TRUE)
  best_model<-h2o.getModel(d_grid@model_ids[[1]])
  
  h2o.saveModel(object = best_model, path = "C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2", force = TRUE) # force overwriting
  name <- file.path( "C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2", best_model@model_id) # destination file name at the same folder location
  file.rename(file.path("C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/", best_model@model_id), paste0(c('modelo_DNN_v2_part3000'),1))
  
}

mean(as.numeric(d_grid@summary_table$accuracy))

var(as.numeric(d_grid@summary_table$accuracy))

modelo10<-h2o.loadModel(path = "C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/modelo_DNN1_v210000")
h2o.removeAll()
h2o.shutdown(prompt=FALSE)





