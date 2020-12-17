# Load MNIST test data and run PCA


# Load MNIST test data
pca <- function(test, code) {
    X_test <- read.csv(test, header = F)
    x_test <- as.matrix(X_test)

    # Run PCA and save result(reconstruct and code(Z, principal component))

    # Define code size list
     z_list <- c(code) 

    pca <- prcomp(x_test, center = T)

    loss_data <- data.frame(z_size = z_list, loss = rep(NA, length(z_list)))

    mu <- (x_test - scale(x_test, scale = F))

    loop_index <- 1

    for(comp_value in z_list){ 
  
          pca_test_z <- scale(x_test, scale = F) %*% pca$rotation[, 1:comp_value]
  
          pca_test_recon <- pca_test_z %*% t(pca$rotation[, 1:comp_value]) + mu
  
          loss_data$loss[loop_index] <- mean((pca_test_recon - x_test)^2)

          pca_test_z <- data.frame(pca_test_z)
          colnames(pca_test_z) <- NULL
  
          pca_test_recon <- data.frame(pca_test_recon)
          colnames(pca_test_recon) <- NULL
  
          write.csv(pca_test_z, file = paste0("test_code.csv"), row.names = FALSE)
          write.csv(pca_test_recon, file = paste0("test_out.csv"), row.names = FALSE)
  
          loop_index = loop_index + 1
  
          print(paste(comp_value, "finish!"))
  
    }

    print(loss_data)
}

# end of pca.R