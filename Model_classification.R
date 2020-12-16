# Run SVM and MLR (with 5-fold cross validation)

# setwd("") # Set directory

# Choice MNIST or Fashion-MNIST
# data_type <- "MNIST"
# data_type <- "Fashion-MNIST"

library(ggplot2)
library(dplyr)
library(caret)
library(e1071)
library(nnet)
library(gridExtra)

model_class <- function(testX, testY, choice_model, z_code,class_model) {    
  X_test <- read.csv(paste0(testX), header = F)
  x_test <- as.matrix(X_test)
  
  Y_test <- read.csv(paste0(testY), header = F)
  y_test <- Y_test[, 1]
  
  #z_list <- c(4, 8, 12, 16, 20)
  z_list=c(z_code)
  
  model_list <- c(choice_model)
  
  kf <- 5
  idx <- createFolds(factor(y_test), k = kf)
  
  total_classifi_data <- data.frame(classifi_method = c(), z_size = c(), model_type = c(), fold_num = c(), class_num = c(), sensi = c(), ppv = c(), bacc = c())
  
  # 1. Run SVM (with parameter tuning)
  if (class_model=="SVM") {
    classifi_method <- "SVM"
    
    for(z_size in z_list){
      
      data <- read.csv(paste0(choice_model, "_test_z_", z_size, ".csv"), header = F)
      
      data_list <- list(data)
      
      comp_value <- ncol(data)
      
      # SVM Tuning (with 10% data)
      
      Start_Time <- Sys.time()
      
      tune_kf <- 10
      tune_idx <- createFolds(factor(y_test), k = tune_kf)
      
      tuning_data <- data.frame(row.names = c(paste0("F", seq(1, comp_value)), "label"))
      
      for(data_index in 1:length(data_list)){
        temp_data <- data_list[[data_index]]
        temp_data <- cbind(temp_data, y_test)
        colnames(temp_data) <- c(paste0("F", seq(1, comp_value)), "label")
        
        temp_data <- temp_data[tune_idx[[1]], ]
        
        tuning_data <- rbind(tuning_data, temp_data)
      }
      
      rows <- sample(nrow(tuning_data))
      tuning_data <- tuning_data[rows, ]
      
      tune <- tune.svm(as.factor(label) ~ ., data = tuning_data, gamma = 10^c(-3:3), cost = 10^c(-3:3))
      
      End_Time <- Sys.time()
      
      Run_Time <- End_Time - Start_Time
      
      print(paste0("SVM Tuning, z size : ", z_size, ", finish!"))
      print(Run_Time)
      
      # Run SVM
      
      for(data_index in 1:length(data_list)){
        
        Start_Time <- Sys.time()
        
        model_type <- model_list[data_index]
        
        train_data <- data_list[[data_index]]
        
        train_data <- cbind(train_data, y_test)
        colnames(train_data) <- c(paste0("F", seq(1, comp_value)), "label")
        
        train_data$label <- as.factor(train_data$label)
        
        test_list <- rep(list(), kf)
        test_ylist <- rep(list(), kf)
        
        for(i in 1:kf){
          kf_train_data <- train_data[-idx[[i]], ]
          kf_test_data <- train_data[idx[[i]], ]
          
          m <- svm(label ~ ., data = kf_train_data, cost = tune$best.parameters[1, "cost"], gamma = tune$best.parameters[1, "gamma"], kernel = "radial")
          
          test_list[[i]] <- predict(m, newdata = kf_test_data[, -(comp_value+1)])
          test_ylist[[i]] <- y_test[idx[[i]]]
        }
        
        test_actual <- c()
        test_predict <- c()
        
        for(i in 1:kf){
          test_actual <- test_ylist[[i]]
          test_predict <- test_list[[i]]
          
          test_actual <- as.factor(test_actual)
          test_predict <- as.factor(test_predict)
          
          levels(test_actual) <- 0:9
          levels(test_predict) <- 0:9
          
          test_confu_mat <- confusionMatrix(test_predict, test_actual)$byClass
          
          for(ind in 0:9){
            sensi_value <- test_confu_mat[ind+1, "Sensitivity"]
            ppv_value <- test_confu_mat[ind+1, "Pos Pred Value"]
            bacc_value <- test_confu_mat[ind+1, "Balanced Accuracy"]
            
            classifi_data <- data.frame(z_size = z_size, classifi_method = classifi_method, model_type = model_type, fold_num = i, class_num = ind, sensi = sensi_value, ppv = ppv_value, bacc = bacc_value)
            
            total_classifi_data <- rbind(total_classifi_data, classifi_data)
          }
        }
        
        End_Time <- Sys.time()
        
        Run_Time <- End_Time - Start_Time
        
        print(paste0("classifi method : ", classifi_method, ", z size : ", z_size, ", model type : ", model_type, ", finish!"))
        print(Run_Time)
        
      }
      
    }
    
    # 3. Visualize results
    
    summary_classifi_data <- total_classifi_data %>% 
      group_by(z_size, classifi_method, model_type, class_num) %>% 
      summarise(sensi_mean = mean(sensi), 
                sensi_se = sd(sensi)/sqrt(n()),
                ppv_mean = mean(ppv), 
                ppv_se = sd(ppv)/sqrt(n()),
                bacc_mean = mean(bacc), 
                bacc_se = sd(bacc)/sqrt(n()))
    
    # write.csv(summary_classifi_data, file = paste0(data_type, "_summary_classifi_data.csv"), row.names = F)
    
    selected_method <- "SVM" # "SVM" or "MLR"
    selected_z <- z_code # 4, 8, 12, 16, 20
    
    selected_data <- summary_classifi_data[summary_classifi_data$classifi_method == selected_method & summary_classifi_data$z_size == selected_z, ]
    write.csv(selected_data, file = paste0(choice_model,"_",class_model,"_",z_code,"_classifi_data.csv"), row.names = F)
    
    
    p1 <- ggplot(selected_data, aes(x = class_num, y = sensi_mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = sensi_mean + sensi_se, ymin = sensi_mean - sensi_se), width = 0.5, size = 1) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", selected_z,", ", selected_method, ", Sensitivity")) + theme(text = element_text(size = 12), plot.title = element_text(hjust = 0.5))
    p2 <- ggplot(selected_data, aes(x = class_num, y = ppv_mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = ppv_mean + ppv_se, ymin = ppv_mean - ppv_se), width = 0.5, size = 1) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", selected_z,", ", selected_method, ", Pos Pred Value")) + theme(text = element_text(size = 12), plot.title = element_text(hjust = 0.5))
    p3 <- ggplot(selected_data, aes(x = class_num, y = bacc_mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = bacc_mean + bacc_se, ymin = bacc_mean - bacc_se), width = 0.5, size = 1) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", selected_z,", ", selected_method, ", Balanced Accuracy")) + theme(text = element_text(size = 12), plot.title = element_text(hjust = 0.5))
    
    grid.arrange(p1,p2,p3,ncol=3)
  }
  
  # 2. Run MLR
  else {
    classifi_method <- "MLR"
    
    for(z_size in z_list){
      
      data <- read.csv(paste0(choice_model, "_test_z_", z_size, ".csv"), header = F)
      
      data_list <- list(data)
      
      comp_value <- ncol(data)
      
      for(data_index in 1:length(data_list)){
        
        Start_Time <- Sys.time()
        
        model_type <- model_list[data_index]
        
        train_data <- data_list[[data_index]]
        
        train_data <- cbind(train_data, y_test)
        colnames(train_data) <- c(paste0("F", seq(1, comp_value)), "label")
        
        train_data$label <- as.factor(train_data$label)
        
        test_list <- rep(list(), kf)
        test_ylist <- rep(list(), kf)
        
        for(i in 1:kf){
          kf_train_data <- train_data[-idx[[i]], ]
          kf_test_data <- train_data[idx[[i]], ]
          
          m <- multinom(label ~ ., data = kf_train_data, MaxNWts = 10000000, maxit = 300)
          
          test_list[[i]] <- predict(m, newdata = kf_test_data[, -(comp_value+1)])
          test_ylist[[i]] <- y_test[idx[[i]]]
        }
        
        test_actual <- c()
        test_predict <- c()
        
        for(i in 1:kf){
          test_actual <- test_ylist[[i]]
          test_predict <- test_list[[i]]
          
          test_actual <- as.factor(test_actual)
          test_predict <- as.factor(test_predict)
          
          levels(test_actual) <- 0:9
          levels(test_predict) <- 0:9
          
          test_confu_mat <- confusionMatrix(test_predict, test_actual)$byClass
          
          for(ind in 0:9){
            sensi_value <- test_confu_mat[ind+1, "Sensitivity"]
            ppv_value <- test_confu_mat[ind+1, "Pos Pred Value"]
            bacc_value <- test_confu_mat[ind+1, "Balanced Accuracy"]
            
            classifi_data <- data.frame(z_size = z_size, classifi_method = classifi_method, model_type = model_type, fold_num = i, class_num = ind, sensi = sensi_value, ppv = ppv_value, bacc = bacc_value)
            
            total_classifi_data <- rbind(total_classifi_data, classifi_data)
          }
        }
        
        End_Time <- Sys.time()
        
        Run_Time <- End_Time - Start_Time
        
        print(paste0("classifi method : ", classifi_method, ", z size : ", z_size, ", model type : ", model_type, ", finish!"))
        print(Run_Time)
        
      }
      
    }
    # 3. Visualize results
    
    summary_classifi_data <- total_classifi_data %>% 
      group_by(z_size, classifi_method, model_type, class_num) %>% 
      summarise(sensi_mean = mean(sensi), 
                sensi_se = sd(sensi)/sqrt(n()),
                ppv_mean = mean(ppv), 
                ppv_se = sd(ppv)/sqrt(n()),
                bacc_mean = mean(bacc), 
                bacc_se = sd(bacc)/sqrt(n()))
    
    # write.csv(summary_classifi_data, file = paste0(data_type, "_summary_classifi_data.csv"), row.names = F)
    
    selected_method <- "MLR" # "SVM" or "MLR"
    selected_z <- z_code # 4, 8, 12, 16, 20
    
    selected_data <- summary_classifi_data[summary_classifi_data$classifi_method == selected_method & summary_classifi_data$z_size == selected_z, ]
    
    write.csv(selected_data, file = paste0(choice_model,"_",class_model,"_",z_code,"_classifi_data.csv"), row.names = F)
    
   
    p1 <- ggplot(selected_data, aes(x = class_num, y = sensi_mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = sensi_mean + sensi_se, ymin = sensi_mean - sensi_se), width = 0.5, size = 1) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", selected_z,", ", selected_method, ", Sensitivity")) + theme(text = element_text(size = 12), plot.title = element_text(hjust = 0.5))
    p2 <- ggplot(selected_data, aes(x = class_num, y = ppv_mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = ppv_mean + ppv_se, ymin = ppv_mean - ppv_se), width = 0.5, size = 1) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", selected_z,", ", selected_method, ", Pos Pred Value")) + theme(text = element_text(size = 12), plot.title = element_text(hjust = 0.5))
    p3 <- ggplot(selected_data, aes(x = class_num, y = bacc_mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = bacc_mean + bacc_se, ymin = bacc_mean - bacc_se), width = 0.5, size = 1) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", selected_z,", ", selected_method, ", Balanced Accuracy")) + theme(text = element_text(size = 12), plot.title = element_text(hjust = 0.5))
    
    grid.arrange(p1,p2,p3, ncol=3)
  }
}
