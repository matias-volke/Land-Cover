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
load("C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/test2_ndvi_part2.RData")
load("C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/test_real_ndvi2.RData")
load("D:/test2_ndvi_part2.RData")
load("D:/test_real_ndvi2.RData")
#####################################################
#####################################################
n_folds <- 10
#set.seed(321)
folds <- createFolds(1:length(entrena[[1]]), k = n_folds)


fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
######################################

svm_gridR <- expand.grid(sigma = c(2^-3, 2^-2, 2^-1, 2^0, 2^1, 2^2),C = c(2^7, 2^6, 2^5, 2^3, 2^1, 2^-1))

mod_svm_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_svm1<-list()
cf_svm11<-list()
ll=0
pb=txtProgressBar(min=0,max = 20,style = 2)
for (l in seq(10,10)) {
  Sys.sleep(0.1)
  ll=ll+1
  mod_svm_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_svm<-list()
  cf_svm1<-list()
  kkk=0
  for (kk in seq(1,1)) {
    kkk=kkk+1
    model_svm <- caret::train(Class ~ . , method = 'svmRadial', data = entrena[[10]][[1]],
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
#parametro<-rbindlist(param_opti1)
#parametro2<-data.frame(t(parametro))
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
colnames(promedio2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
stde2<-stde2[-1]
colnames(stde2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
max_total<-which.max(promedio2)
maxii<-maxii[-1]

confu_svm<-cf_svm11[[max_total]][[as.numeric(maxii[max_total])]]
ensa_svm <- confusionMatrix(data = predict(mod_svm_fin1[[max_total]][[as.numeric(maxii[max_total])]], newdata =test_total),
                            test_total$Class)

#########################################################################################
# Hiperparámetros
xgb_grid = expand.grid(
  nrounds = c(1), 
  max_depth = seq(5, 15, 5),
  eta = c(0.002, 0.02, 0.2),
  gamma = c(0.1, 0.5, 1.0), 
  colsample_bytree = 1, 
  min_child_weight = c(1, 2, 3),
  subsample = c(0.5, 0.75, 1)
)


mod_xgb_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
ll=0
pb <- txtProgressBar(min = 0, max = 20, style = 3)
for (l in seq(10,10)) {
  Sys.sleep(0.1)
  ll=ll+1
  mod_xgb_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  kkk=0
  for (kk in seq(1,1)) {
    kkk=kkk+1
    model_xgb = caret::train(x = entrena[[l]][[kk]][,-8], 
                             y = entrena[[l]][[kk]][,8],
                             method = "xgbTree",
                             trControl = fitControl,
                             tuneGrid = xgb_grid,
                             metric = "Accuracy")
    
    a<-model_xgb$results[which.max(model_xgb$results$Accuracy),]
    accu<-a$Accuracy
    
    mod_xgb_fin[[kkk]]=model_xgb
    param_opti[[kkk]]<-as.numeric(model_xgb$bestTune)
    accu_opti1[[kkk]]<-accu
    #    accu_opti<-rbind(accu_opti,accu)
  }
  mod_xgb_fin1[[ll]]=mod_xgb_fin
  param_opti1[[ll]]=param_opti
  accu_opti11[[ll]]=accu_opti1
  df_accu_opti<-cbind(df_accu_opti,accu_opti1)
  
  setTxtProgressBar(pb, l)
}
close(pb)
############################################
precision<-rbindlist(accu_opti11)
precision2<-data.frame(t(precision))
parametro<-rbindlist(param_opti1)
parametro2<-data.frame(t(parametro))

promedio2<-data.frame(matrix(NA))
stde2<-data.frame(matrix(NA))
maxii<-data.frame(matrix(NA))
param_max<-data.frame(matrix(NA))
for (s in seq(1,ncol(precision2))) {
  prom<-mean(precision2[,s])
  sd<-sd(precision2[,s])
  maxi<-which.max(precision2[,s])
  maxii<-cbind(maxii,maxi)
  p_max<-parametro2[maxi,s]
  promedio2<-cbind(promedio2,prom)
  stde2<-cbind(stde2,sd)
  param_max<-cbind(param_max,p_max)
}
promedio2<-promedio2[-1]
colnames(promedio2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
stde2<-stde2[-1]
colnames(stde2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
param_max<-param_max[-1]
colnames(param_max)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
max_total<-which.max(promedio2)
best_param<-param_max[max_total]
maxii<-maxii[-1]

cm_xgb <- confusionMatrix(data = predict(mod_xgb_fin1[[max_total]][[as.numeric(maxii[max_total])]], newdata = test_total),
                          test_total$Class)

prom_xgb<-promedio2
stde_xgb<-stde2
param_max_xgb<-param_max
max_total_xgb<-max_total
best_param_xgb<-best_param
maxi_xgb<-maxii
plot(mod_xgb_fin1[[max_total]][[as.numeric(maxii[max_total])]]) # tuning results
## SAVE PARAMETROS
save(mod_xgb_fin1,prom_xgb,stde_xgb,maxi_xgb,max_total_xgb,best_param_xgb, cm_xgb,file = "XGB.RData")
########################################################################


mod_nb_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_nb1<-list()
cf_nb11<-list()
ll=0
pb=txtProgressBar(min=0,max = 20,style = 2)
for (l in seq(10,10)) {
  Sys.sleep(0.1)
  ll=ll+1
  mod_nb_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_nb<-list()
  cf_nb1<-list()
  kkk=0
  for (kk in seq(1,5)) {
    kkk=kkk+1
    model_nb <- caret::train(Class ~ . , method = 'nb', data = entrena[[l]][[kk]],
                             allowParallel = TRUE)
    #    op<-which(model_svm$results["mtry"]==as.numeric(model_svm$bestTune))
    accu<-model_nb$results["Accuracy"]
    
    mod_nb_fin[[kkk]]=model_nb
    accu_opti1[[kkk]]<-accu[1,]
    prediccion<-confusionMatrix(predict( model_nb,test_total),as.factor(test_total$Class))
    prec<-prediccion$overall[1]
    prec_nb[[kkk]]<-prec
    cf_nb<-prediccion$table
    cf_nb1[[kkk]]<-cf_nb
    
    #    accu_opti<-rbind(accu_opti,accu)
  }
  mod_nb_fin1[[ll]]=mod_nb_fin
  accu_opti11[[ll]]=accu_opti1
  df_accu_opti<-cbind(df_accu_opti,accu_opti1)
  prec_nb1[[ll]]=prec_nb
  cf_nb11[[ll]]=cf_nb1
  setTxtProgressBar(pb,l)
}
close(pb)

precision<-rbindlist(accu_opti11)
precision2<-data.frame(t(precision))
promedio2<-data.frame(matrix(NA))
stde2<-data.frame(matrix(NA))
maxii<-data.frame(matrix(NA))
for (s in seq(1,ncol(precision2))) {
  prom<-mean(precision2[,s],na.rm=TRUE)
  sd<-sd(precision2[,s],na.rm=TRUE)
  promedio2<-cbind(promedio2,prom)
  stde2<-cbind(stde2,sd)
  maxi<-which.max(precision2[,s])
  maxii<-cbind(maxii,maxi)
  
}
promedio2<-promedio2[-1]
colnames(promedio2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
stde2<-stde2[-1]
colnames(stde2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
max_total<-which.max(promedio2)
best<-precision2[max_total]
maxii<-maxii[,-1]

cm_nb <- confusionMatrix(data = predict(mod_nb_fin1[[max_total]][[as.numeric(maxii[max_total])]], newdata = test_total),
                         test_total$Class)


prom_nb=promedio2
std_nb=stde2
max_nb=maxii
max_total_nb=max_total
save(mod_nb_fin1,prom_nb,std_nb,max_nb,max_total_nb, cm_nb,file = "NB100-POSTL.RData")
load("RF.RData")
##########################################################
######    v                                        #######
##########################################################
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



hyper_params <- list(
  activation = "RectifierWithDropout", 
  hidden = lapply(1:50, function(x)10+sample(50,sample(4), replace=TRUE)),
  epochs = c(10,50, 100, 150 ,200),
  l1=1e-6, # L1/L2 regularization, improve generalization
  l2=1e-6,
  rho = c(0.9, 0.95, 0.99, 0.999),
  epsilon = c(1e-10, 1e-8, 1e-6, 1e-4),
  max_w2 = 10
)


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
  file.rename(file.path("C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/", best_model@model_id), paste0(c('modelo_DNN_v2_part40'),1))
  
}

mean(as.numeric(d_grid@summary_table$accuracy))

var(as.numeric(d_grid@summary_table$accuracy))

modelo10<-h2o.loadModel(path = "C:/Users/matias/Desktop/PAPER-ML/R-PROGRAM2/modelo_DNN_v2_part401")
h2o.removeAll()
h2o.shutdown(prompt=FALSE)

###################################################################
#############GBM
# Grid of tuning parameters
#Creating grid
gbm_grid <- expand.grid(n.trees=c(100,200,500),shrinkage=c(0.05,0.1,0.5),n.minobsinnode = c(1,5,10),interaction.depth=c(5,10))
gbm_grid <- expand.grid(n.trees=c(100),shrinkage=c(0.05),n.minobsinnode = c(1),interaction.depth=c(10))
gbm_grid <- expand.grid(n.trees=c(50,100,200,500),shrinkage=c(0.05),n.minobsinnode = c(1,5,10),interaction.depth=c(10))
gbm_grid <- expand.grid(n.trees=c(50,100,200,500),shrinkage=c(0.05,0.1,0.5),n.minobsinnode = c(1),interaction.depth=c(10))

mod_gbm_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
ll=0
for (l in  seq(10,10)) {
  
  ll=ll+1
  mod_gbm_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  kkk=0
  pb <- txtProgressBar(min = 0, max = 20, style = 3)
  for (kk in seq(1,1)) {
    Sys.sleep(0.1)
    kkk=kkk+1
    model_gbm <- caret::train(Class ~.,method = "gbm", data = entrena[[10]][[1]],
                              tuneGrid = gbm_grid,
                              trControl=fitControl)
    #op<-which(model_rf$results["mtry"]==as.numeric(model_rf$bestTune))
    accu<-model_gbm$results[as.numeric(rownames(model_gbm$bestTune)),"Accuracy"]
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
}
close(pb)

precision<-rbindlist(accu_opti11)
precision2<-data.frame(t(precision))
parametro<-rbindlist(param_opti1)
parametro2<-data.frame(t(parametro))
promedio2<-data.frame(matrix(NA))
stde2<-data.frame(matrix(NA))
maxii<-data.frame(matrix(NA))
param_max<-data.frame(matrix(NA))
for (s in seq(1,ncol(precision2))) {
  prom<-mean(precision2[,s])
  sd<-sd(precision2[,s])
  maxi<-which.max(precision2[,s])
  maxii<-cbind(maxii,maxi)
  p_max<-parametro2[maxi,s]
  promedio2<-cbind(promedio2,prom)
  stde2<-cbind(stde2,sd)
  param_max<-cbind(param_max,p_max)
}
promedio2<-promedio2[-1]
colnames(promedio2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
stde2<-stde2[-1]
colnames(stde2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
param_max<-param_max[-1]
colnames(param_max)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
max_total<-which.max(promedio2)
best_param<-param_max[max_total]
maxii<-maxii[-1]
cm_rf <- confusionMatrix(data = predict(mod_rf_fin1[[max_total]][[as.numeric(maxii[max_total])]], newdata =test_total),
                         test_total$Class)

plot(mod_rf_fin1[[max_total]][[as.numeric(maxii[max_total])]]) # tuning results
## SAVE PARAMETROS
acu_test8<-precision2
acu_test78<-cbind(acu_test67,acu_test8)
prom_rf=promedio2
std_rf=stde2
max_rf=maxii
max_total_rf=max_total
best_param_rf=best_param
save(acu_test78,file = "test1_gbm.RData")
# To load the data again
load("test1_gbm.RData")


###########################################################3

hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 6) %>% floor()
)

mod_mars_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_mars1<-list()
cf_mars11<-list()
ll=0
pb <- txtProgressBar(min = 0, max = 20, style = 3)
for (l in seq(10,10)) {
  Sys.sleep(0.1)
  ll=ll+1
  mod_mars_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_mars<-list()
  cf_mars1<-list()
  kkk=0
  for (kk in seq(10,10)) {
    kkk=kkk+1
    model_mars <- caret::train(Class ~ ., data = entrena[[l]][[kk]],
                               method = "earth",
                               tuneGrid = hyper_grid,
                               metric = "Accuracy")
    a<-model_mars$results[which.max(model_mars$results$Accuracy),]
    accu<-a$Accuracy
    prediccion<-confusionMatrix(predict( model_mars,test_total),as.factor(test_total$Class))
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

cm_mars <- confusionMatrix(data = predict(mod_mars_fin1[[1]][[as.numeric(maxii)]], newdata =test_total),
                           test_total$Class)

mod_mars<-mod_mars_fin1[[1]][[as.numeric(maxii)]]
save(mod_mars,file = "modelo_mars_posA.RData")

prom_mars<-promedio2
std_mars<-stde2[-1]
save(mod_mars_fin1,prom_mars,std_mars,file = "mod_mars10_posA.RData")

#####################################

rf_grid <- expand.grid(mtry = c(1,2,3))


mod_rf_fin1=list()
param_opti1<-list()
df_accu_opti<-data.frame(matrix(NA))
accu_opti11<-list()
prec_rf1<-list()
cf_rf11<-list()
ll=0
pb=txtProgressBar(min=0,max = 20,style = 2)
for (l in seq(10,10)) {
  Sys.sleep(0.1)
  ll=ll+1
  mod_rf_fin=list()
  param_opti<-list()
  accu_opti<-data.frame(matrix(NA))
  accu_opti1<-list()
  prec_rf<-list()
  cf_rf1<-list()
  kkk=0
  for (kk in seq(10,10)) {
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
colnames(promedio2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
stde2<-stde2[-1]
colnames(stde2)<-c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%")
max_total<-which.max(promedio2)
maxii<-maxii[-1]

confu_rf<-cf_rf11[[max_total]][[as.numeric(maxii[max_total])]]
ensa_rf <- confusionMatrix(data = predict(mod_rf_fin1[[max_total]][[as.numeric(maxii[max_total])]], newdata =test_total),
                           test_total$Class)

mod_rf_param<-mod_rf_fin1[[1]][[1]]
plot(mod_rf_fin1[[1]][[1]]) # tuning results
save(mod_rf_param,file = "parametros_rf.RData")
#####################################################################################

