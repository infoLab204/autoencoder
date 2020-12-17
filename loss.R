# Calculate and visualize loss

library(ggplot2)
library(dplyr)

loss <- function(ctest, coutput) {
  zcode_data <- data.frame(class_index = c(), mean = c(), se = c())
  
  
  class_index=substr(ctest,17,17)

  X_test <- read.csv(paste0(ctest), header = F)
  
  X_test <- as.matrix(X_test)
  
  
  rec_data <- read.csv(paste0(coutput), header = F)
  rec_data <- as.matrix(rec_data)
  rec_loss <- c()
  
  for(i in 1:nrow(rec_data))
    rec_loss[i] <- mean((X_test[i, ] - rec_data[i, ])^2)
  
  
  total_rec_loss <- data.frame(each = length(rec_loss), loss = c(rec_loss))
  
  temp_data <- total_rec_loss  %>% summarise(mean = mean(loss), se = sd(loss)/sqrt(n()))
  
  temp_data <- data.frame(temp_data)
  
  zcode_data <- rbind(zcode_data, temp_data)
  
  print(zcode_data)
  print(paste("finish!"))
  
  
  ggplot(zcode_data, aes(x = class_index, y = mean)) + geom_errorbar(aes(ymax = mean + se, ymin = mean - se), width = 0.5, size = 0.5) +  ggtitle(paste0("model recon loss")) + theme(text = element_text(size = 13), plot.title = element_text(hjust = 0.5))


}
