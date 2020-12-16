# Calculate and visualize loss

library(ggplot2)
library(dplyr)
library(gridExtra)


# 1. Calculate loss per code size
cal_loss <- function(testX, choice_model, z_code) {
  X_test <- read.csv(testX, header = F)
  X_test <- as.matrix(X_test)
  
  total_data <- data.frame(model_type = c(), z_size = c(), mean = c(), se = c())
  

  z_list <- c(z_code)
  
  for(loop_index in 1:length(z_list)) {
    
    z_size <- z_list[loop_index]
    
    rec_data <- read.csv(paste0(choice_model,"_test_output_", z_size, ".csv"), header = F)
    rec_data <- as.matrix(rec_data)
    rec_loss <- c()
    
    
    for(i in 1:nrow(rec_data))
      rec_loss[i] <- mean((X_test[i, ] - rec_data[i, ])^2)
    
    
    total_rec_loss <- data.frame(model_type = rep(c(choice_model), each = length(rec_loss)), z_size = z_size, loss = c(rec_loss))
    
    temp_data <- total_rec_loss %>% group_by(z_size, model_type) %>% summarise(mean = mean(loss), se = sd(loss)/sqrt(n()))
    
    temp_data <- data.frame(temp_data)
    
    total_data <- rbind(total_data, temp_data)
    
    print(paste(z_size, "finish!"))
    
  }
  
  p1 <-ggplot(total_data, aes(x = z_size, y = mean, group = model_type, color = model_type)) + geom_line() + geom_errorbar(aes(ymax = mean + se, ymin = mean - se), width = 1, size = 0.5) + scale_x_continuous(breaks = seq(4, 20, 4)) + ggtitle(paste0(data_type, ", model recon loss")) + theme(text = element_text(size = 13), plot.title = element_text(hjust = 0.5))
  
  write.csv(total_data, file = paste0(choice_model,"_",z_code, "_total_loss_data.csv"), row.names = F)
  
  
  
  # 2. Calculate loss per code size and each class
  
  zcode_data <- data.frame(model_type = c(), z_size = c(), class_index = c(), mean = c(), se = c())
  
  for(class_index in 0:9) {  
    
    X_test <- read.csv(paste0("MNIST_X_test_class", class_index, ".csv"), header = F)
    
    X_test <- as.matrix(X_test)
    
    for(z_size in z_list) {
      rec_data <- read.csv(paste0(choice_model,"_test_output_", z_size, "_class", class_index, ".csv"), header = F)
      rec_data <- as.matrix(rec_data)
      rec_loss <- c()
      
      for(i in 1:nrow(rec_data))
        rec_loss[i] <- mean((X_test[i, ] - rec_data[i, ])^2)
      
      
      total_rec_loss <- data.frame(model_type = rep(c(choice_model), each = length(rec_loss)), z_size = z_size, class_index = class_index, loss = c(rec_loss))
      
      temp_data <- total_rec_loss %>% group_by(z_size, class_index, model_type) %>% summarise(mean = mean(loss), se = sd(loss)/sqrt(n()))
      
      temp_data <- data.frame(temp_data)
      
      zcode_data <- rbind(zcode_data, temp_data)
      
      print(paste(class_index, z_size, "finish!"))
      
    }
    
  }
  
  zcode_data <- zcode_data[zcode_data$z_size == z_code, ]
  
  p2 <- ggplot(zcode_data, aes(x = class_index, y = mean, group = model_type, color = model_type)) + geom_errorbar(aes(ymax = mean + se, ymin = mean - se), width = 0.5, size = 0.5) + scale_x_continuous(breaks = c(0:9)) + ggtitle(paste0(data_type, ", Z = ", 4 ,", model recon loss")) + theme(text = element_text(size = 13), plot.title = element_text(hjust = 0.5))
  
  grid.arrange(p1,p2, ncol=2)
  write.csv(zcode_data, file = paste0(choice_model,"_",z_code, "_class_loss_data.csv"), row.names = F)

}
