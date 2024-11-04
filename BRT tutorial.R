#####################################################################################
######                                                                        #######
#####                Boosted Regression Trees Tutorial                        #######
####                    using presence/absence data                           #######
####                                                                          #######
#####################################################################################

# Author: Kimberly Thompson

# This code walks through the steps of determining the optimal settings for a 
# BRT model, running the model with the optimal settings, and then evaluating
# the accuracy of the model results.


########## load required packages ####################
library( tidyverse ) # data wrangling
library( ggplot2 ) # plots
library( dismo ) # BRT modelling
library( gbm) # BRT predictions
library( caret ) # sensitivity and specificity
library( psych ) # Cohen's kappa




###############################################
###                                         ###
###             Model Explanation           ###
###                                         ###
###############################################

# Two sets of presence/absence data are necessary:
# one for model training (building) and one for model testing (evaluation).
# We will use ~70% of the data for training, and ~30% for testing. 

# Order of operations will be as follows:

# 1. Determine the optimal model settings

# Optimal settings will be determined using predictive deviance, with the 
# best settings having the lowest predictive deviance (i.e., the smallest
# difference between predicted values and actual values).

#    a. Examine different learning rates and number of trees for 
#       four different tree complexity values

#    Learning rate: reduces the contribution of each tree that is added
#                   to the model, placing higher importance on earlier
#                   trees that explain more of the observed patterns in
#                   the data
#    Tree complexity: indicates the interaction order, with 1 corresponding
#                     to no interactions, 2 corresponding to 2-way 
#                     interactions, and so on

# In general models perform best with a decreased (slow) learning rate,
# a high number of total trees, and a tree complexity that reflect the 
# true interaction of the predictors.

#    b. Examine different bag fraction values for the optimal combo of
#       learning rate, number of trees, and tree complexity identified
#       in step 1a.

#    Bag fraction: Indicates the level of stochasticity and ranges from 
#                  0 to 1, with 1 indicating a completely deterministic
#                  model. Elith et al. (2008) suggest that values btw
#                  0.50 and 0.75 have provided the best results for 
#                  presence/absence responses.


# 2. Run the model for training data using the optimal settings.

# 3. Evaluate the model with a suite of metrics

#    AUC: values range from 0 to 1, where 1 indicates perfect discrimination
#         between presences and absences, 0.5 indicates discrimination
#         no better than chance, and values below 0.5 indicate performance
#         worse than that of random chance.
#    Sensitivity: rate of correctly identifying presences
#    Specificity: rate of correctly identifying absences
#    Prevalence: measure of how often do presences actually appear in the 
#                validation dataset
#    Cohen's kappa: ranges from -1 to 1, with values of 1 indicating perfect 
#                   discrimination and values of 0 or less indicating that 
#                   the model predictions were no better than random chance: 
#         2 types:
#            Unweighted Cohen's kappa: Use this when you want to assess 
#                                      agreement or disagreement without
#                                      concern for how "far apart" the
#                                      agreement is 
#            Weighted Cohen's kappa: Use this when you need to reflect the 
#                                    severity of disagreements in your
#                                    evaluation metrics
#    True Skill Statistic (TSS): ranges from -1 to 1, with values of 1 
#                   indicating perfect discrimination and values of 
#                   0 or less indicating that the model predictions were 
#                   no better than random chance



#### Notes about model syntax for other settings
# data = training dataset
# gbm.x = indices of names of predictor columns in data
# gbm.y = index of response variable
# family = use Bernoulli for presence absence (binomial)

# Keep in mind these models are stochastic and so slightly different each time you run 
# them unless you make them deterministic
# by using a bag fraction of 1. 



###############################################
###                                         ###
###             Data Loading                ###
###                                         ###
###############################################

# Define the path where the data is saved
# Fill in with your own directory path
path <- "H:/My Drive/Coding Club"

# Read in the data
presence <- read.csv(paste(path, "/Presence_Absence_external_treatment.csv", sep = ""),
                     header = TRUE)


# Examine the data
head(presence)
str(presence)

### What is the data?

# Presence: 0 or 1 representing the absence or presence of the subnivium
#              (thermally stable habitat under the snow)
# Taimmin: Minimum daily air temperature (C)
# Tairmax: Maximum daily air temperature (C)
# Snowmed: Median daily snow depth (cm)
# Wind: Average daily wind speed (m/s)
# density.Mean_SNODAS: Average daily snow density (g/cm3)
# Cover: Dec, Con, or Open, representing deciduous, conifer, or open land cover


###############################################
###                                         ###
###             Data Preparation            ###
###                                         ###
###############################################

# Define variable Cover as a factor
presence$Cover <- as.factor(presence$Cover)

#Add ID column to aid in subsetting
presence$ID <- seq(from=1, to=1089, by=1)

# Total number of sites sampled in presence/absence data:
M <- max( presence$ID )

# Define number of data points to use for testing:
# ~30% of data points
TP <- floor(M*0.3) 

# Select which rows to use for testing from the presence/absence data:
t.parows <- sample( x = 1:M, size = TP, replace = FALSE )

# Subset the complete dataset to exclude these rows thereby creating
# the training dataset (~70% of the data)
presence.train <- presence[ -t.parows, ]

# Create testing dataset:
presence.test <- presence[ t.parows, ]



###############################################
###  Determine the optimal model settings   ###
###      # of trees, learning rate,         ###
###         and tree complexity             ###
###############################################

# We will systematically test different numbers of trees and learning
# rates for 4 different tree complexity values

# Create a list of learning rates to test
lr <- c(0.05, 0.01, 0.005, 0.001, 0.0005)

# Create a list of different numbers of trees to test
tree.list <- seq(100, 20000, by=100)

# Create a blank dataframe to store best settings for each tree
# tree complexity
settings <- data.frame(Tree.Complex = seq(1, 4, by = 1),
                       Learning.Rate = numeric(4),
                       Number.Trees = integer(4),
                       Pred.Dev = numeric(4))


#######################
###                 ###
###      TC = 1     ###
###                 ###
#######################

# Create an empty dataframe in which to store the predicted deviances
deviance.tc1 <- data.frame(lr05=numeric(200), lr01=numeric(200),
                           lr005=numeric(200), lr001=numeric(200),
                           lr0005=numeric(200), nt=seq(100, 20000, by=100))

# Model each different learning rate and store them in a data frame
for (i in 1:length(lr)) {
  
  mod <- dismo :: gbm.fixed(data = presence.train,
                            gbm.x = c(2:7),
                            gbm.y = 1,
                   family = "bernoulli",
                   tree.complexity = 1, 
                   learning.rate = lr[i],
                   n.trees = 20000)
  
  # Create matrix of predictions, each column = predictions from the model
  # For example, the predictions
  # in column 5 are for tree.list[5]=500 trees
  pred <- gbm :: predict.gbm(mod, presence.test,
                             n.trees = tree.list, 
                             type = "response")
  
  # Calculate deviance of all of these results and store them in a dataframe:
  for (j in 1:length(deviance.tc1[,i])) {
    
    deviance.tc1[,i][j] <- calc.deviance(presence.test$Presence,
                                         pred[,j], calc.mean = T)
    
  } # end of j loop
} # end of i loop


### Store and Graph the results ###

# Find the minimum value of predictive deviance
min_value <- min(deviance.tc1, na.rm=TRUE)

# Identify the row and column index of the min value
row_index <- which(deviance.tc1 == min_value, arr.ind = TRUE)[1, 1]
col_index <- which(deviance.tc1 == min_value, arr.ind = TRUE)[1, 2]

# Get the column name where the minimum value was found
min_col_name <- colnames(deviance.tc1)[col_index]

# Get the corresponding number of trees from the row of the min val
nt_value <- deviance.tc1[row_index, 'nt']

# Fill in the optimal values in the settings dataframe
settings[1, 'Learning.Rate'] <- as.numeric(paste0("0.", 
                                                  sub("lr", "", min_col_name)))
settings[1, 'Number.Trees'] <- nt_value

settings[1, 'Pred.Dev'] <- min_value

# Graph the results of the learning rate comparison for tc=1
tc1 <- ggplot() +
  geom_line(data = deviance.tc1, 
            aes(x = nt, y = lr05, linetype = "0.05"), linewidth = 1) +
  geom_line(data = deviance.tc1, 
            aes(x = nt, y = lr01, linetype = "0.01"), linewidth = 1) +
  geom_line(data = deviance.tc1, 
            aes(x = nt, y=lr005, linetype = "0.005"), linewidth = 1) +
  geom_line(data = deviance.tc1, 
            aes(x = nt, y = lr001, linetype = "0.001"), linewidth = 1) +
  geom_line(data = deviance.tc1, 
            aes(x = nt, y = lr0005, linetype = "0.0005"), linewidth = 1) +
  geom_hline(yintercept = min_value, linetype = "solid", color = "red",
             linewidth = 1.2) +
  geom_vline(xintercept = nt_value, 
             linetype = "solid", color = "green", linewidth = 1.2) +
  theme_bw() +
  theme(axis.text.x = element_text(size = 20, face = "bold")) +
  theme(axis.text.y = element_text(size = 20, face = "bold")) +
  theme(axis.title.x = element_text(size = 22, face = "bold", color = "gray30")) +
  theme(axis.title.y = element_text(size = 22, face = "bold", color = "gray30")) +
  scale_x_continuous(name = "\nNumber of Trees", limits = c(0, 20000),
                     breaks=c(0, 5000, 10000, 15000, 20000)) +
  scale_y_continuous(name = "Predictive Deviance\n", limits = c(0.5, 1.7)) +
  scale_linetype_manual(name = '', values = c('0.05' = "twodash", 
                                            '0.01' = "solid",
                                            '0.005' = "longdash",
                                            '0.001' = "dotted",
                                            '0.0005' = "dotdash"),
                        labels = c("0.0005", "0.001", "0.005", "0.01", "0.05")) +
  theme(aspect.ratio = 1) + #aspect ratio expressed as y/x
  # theme(legend.position = "none") +
  theme(legend.text = element_text(size=14)) +
  guides(colour = guide_legend(override.aes = (list(size=3))))
  



#######################
###                 ###
###      TC = 2     ###
###                 ###
#######################

# Create an empty dataframe in which to store the predicted deviances
deviance.tc2 <- data.frame(lr05=numeric(200), lr01=numeric(200),
                           lr005=numeric(200), lr001=numeric(200),
                           lr0005=numeric(200), nt=seq(100, 20000, by=100))

# Model each different learning rate and store them in a data frame
for (i in 1:length(lr)) {
  
  mod <- dismo :: gbm.fixed(data = presence.train,
                            gbm.x = c(2:7),
                            gbm.y = 1,
                            family = "bernoulli",
                            tree.complexity = 2, 
                            learning.rate = lr[i],
                            n.trees = 20000)
  
  # Create matrix of predictions, each column = predictions from the model
  # For example, the predictions
  # in column 5 are for tree.list[5]=500 trees
  pred <- gbm :: predict.gbm(mod, presence.test,
                             n.trees = tree.list, 
                             type = "response")
  
  # Calculate deviance of all of these results and store them in a dataframe:
  for (j in 1:length(deviance.tc2[,i])) {
    
    deviance.tc2[,i][j] <- calc.deviance(presence.test$Presence,
                                         pred[,j], calc.mean = T)
    
  } # end of j loop
} # end of i loop


### Store and Graph the results ###

# Find the minimum value of predictive deviance
min_value <- min(deviance.tc2, na.rm=TRUE)

# Identify the row and column index of the min value
row_index <- which(deviance.tc2 == min_value, arr.ind = TRUE)[1, 1]
col_index <- which(deviance.tc2 == min_value, arr.ind = TRUE)[1, 2]

# Get the column name where the minimum value was found
min_col_name <- colnames(deviance.tc2)[col_index]

# Get the corresponding number of trees from the row of the min val
nt_value <- deviance.tc2[row_index, 'nt']

# Fill in the optimal values in the settings dataframe
settings[2, 'Learning.Rate'] <- as.numeric(paste0("0.", 
                                                  sub("lr", "", min_col_name)))
settings[2, 'Number.Trees'] <- nt_value

settings[2, 'Pred.Dev'] <- min_value

# Graph the results of the learning rate comparison for tc=2
tc2 <- ggplot() +
  geom_line(data = deviance.tc2, 
            aes(x = nt, y = lr05, linetype = "0.05"), linewidth = 1) +
  geom_line(data = deviance.tc2, 
            aes(x = nt, y = lr01, linetype = "0.01"), linewidth = 1) +
  geom_line(data = deviance.tc2, 
            aes(x = nt, y=lr005, linetype = "0.005"), linewidth = 1) +
  geom_line(data = deviance.tc2, 
            aes(x = nt, y = lr001, linetype = "0.001"), linewidth = 1) +
  geom_line(data = deviance.tc2, 
            aes(x = nt, y = lr0005, linetype = "0.0005"), linewidth = 1) +
  geom_hline(yintercept = min_value, linetype = "solid", color = "red",
             linewidth = 1.2) +
  geom_vline(xintercept = nt_value, 
             linetype = "solid", color = "green", linewidth = 1.2) +
  theme_bw() +
  theme(axis.text.x = element_text(size = 20, face = "bold")) +
  theme(axis.text.y = element_text(size = 20, face = "bold")) +
  theme(axis.title.x = element_text(size = 22, face = "bold", color = "gray30")) +
  theme(axis.title.y = element_text(size = 22, face = "bold", color = "gray30")) +
  scale_x_continuous(name = "\nNumber of Trees", limits = c(0, 20000),
                     breaks=c(0, 5000, 10000, 15000, 20000)) +
  scale_y_continuous(name = "Predictive Deviance\n", limits = c(0.5, 1.7)) +
  scale_linetype_manual(name = '', values = c('0.05' = "twodash", 
                                              '0.01' = "solid",
                                              '0.005' = "longdash",
                                              '0.001' = "dotted",
                                              '0.0005' = "dotdash"),
                        labels = c("0.0005", "0.001", "0.005", "0.01", "0.05")) +
  theme(aspect.ratio = 1) + #aspect ratio expressed as y/x
  # theme(legend.position = "none") +
  theme(legend.text = element_text(size=14)) +
  guides(colour = guide_legend(override.aes = (list(size=3))))



#######################
###                 ###
###      TC = 3     ###
###                 ###
#######################

# Create an empty dataframe in which to store the predicted deviances
deviance.tc3 <- data.frame(lr05=numeric(200), lr01=numeric(200),
                           lr005=numeric(200), lr001=numeric(200),
                           lr0005=numeric(200), nt=seq(100, 20000, by=100))

# Model each different learning rate and store them in a data frame
for (i in 1:length(lr)) {
  
  mod <- dismo :: gbm.fixed(data = presence.train,
                            gbm.x = c(2:7),
                            gbm.y = 1,
                            family = "bernoulli",
                            tree.complexity = 3, 
                            learning.rate = lr[i],
                            n.trees = 20000)
  
  # Create matrix of predictions, each column = predictions from the model
  # For example, the predictions
  # in column 5 are for tree.list[5]=500 trees
  pred <- gbm :: predict.gbm(mod, presence.test,
                             n.trees = tree.list, 
                             type = "response")
  
  # Calculate deviance of all of these results and store them in a dataframe:
  for (j in 1:length(deviance.tc3[,i])) {
    
    deviance.tc3[,i][j] <- calc.deviance(presence.test$Presence,
                                         pred[,j], calc.mean = T)
    
  } # end of j loop
} # end of i loop


### Store and Graph the results ###

# Find the minimum value of predictive deviance
min_value <- min(deviance.tc3, na.rm=TRUE)

# Identify the row and column index of the min value
row_index <- which(deviance.tc3 == min_value, arr.ind = TRUE)[1, 1]
col_index <- which(deviance.tc3 == min_value, arr.ind = TRUE)[1, 2]

# Get the column name where the minimum value was found
min_col_name <- colnames(deviance.tc3)[col_index]

# Get the corresponding number of trees from the row of the min val
nt_value <- deviance.tc3[row_index, 'nt']

# Fill in the optimal values in the settings dataframe
settings[3, 'Learning.Rate'] <- as.numeric(paste0("0.", 
                                                  sub("lr", "", min_col_name)))
settings[3, 'Number.Trees'] <- nt_value

settings[3, 'Pred.Dev'] <- min_value

# Graph the results of the learning rate comparison for tc=3
tc3 <- ggplot() +
  geom_line(data = deviance.tc3, 
            aes(x = nt, y = lr05, linetype = "0.05"), linewidth = 1) +
  geom_line(data = deviance.tc3, 
            aes(x = nt, y = lr01, linetype = "0.01"), linewidth = 1) +
  geom_line(data = deviance.tc3, 
            aes(x = nt, y=lr005, linetype = "0.005"), linewidth = 1) +
  geom_line(data = deviance.tc3, 
            aes(x = nt, y = lr001, linetype = "0.001"), linewidth = 1) +
  geom_line(data = deviance.tc3, 
            aes(x = nt, y = lr0005, linetype = "0.0005"), linewidth = 1) +
  geom_hline(yintercept = min_value, linetype = "solid", color = "red",
             linewidth = 1.2) +
  geom_vline(xintercept = nt_value, 
             linetype = "solid", color = "green", linewidth = 1.2) +
  theme_bw() +
  theme(axis.text.x = element_text(size = 20, face = "bold")) +
  theme(axis.text.y = element_text(size = 20, face = "bold")) +
  theme(axis.title.x = element_text(size = 22, face = "bold", color = "gray30")) +
  theme(axis.title.y = element_text(size = 22, face = "bold", color = "gray30")) +
  scale_x_continuous(name = "\nNumber of Trees", limits = c(0, 20000),
                     breaks=c(0, 5000, 10000, 15000, 20000)) +
  scale_y_continuous(name = "Predictive Deviance\n", limits = c(0.5, 1.7)) +
  scale_linetype_manual(name = '', values = c('0.05' = "twodash", 
                                              '0.01' = "solid",
                                              '0.005' = "longdash",
                                              '0.001' = "dotted",
                                              '0.0005' = "dotdash"),
                        labels = c("0.0005", "0.001", "0.005", "0.01", "0.05")) +
  theme(aspect.ratio = 1) + #aspect ratio expressed as y/x
  # theme(legend.position = "none") +
  theme(legend.text = element_text(size=14)) +
  guides(colour = guide_legend(override.aes = (list(size=3))))




#######################
###                 ###
###      TC = 4     ###
###                 ###
#######################

# Create an empty dataframe in which to store the predicted deviances
deviance.tc4 <- data.frame(lr05=numeric(200), lr01=numeric(200),
                           lr005=numeric(200), lr001=numeric(200),
                           lr0005=numeric(200), nt=seq(100, 20000, by=100))

# Model each different learning rate and store them in a data frame
for (i in 1:length(lr)) {
  
  mod <- dismo :: gbm.fixed(data = presence.train,
                            gbm.x = c(2:7),
                            gbm.y = 1,
                            family = "bernoulli",
                            tree.complexity = 4, 
                            learning.rate = lr[i],
                            n.trees = 20000)
  
  # Create matrix of predictions, each column = predictions from the model
  # For example, the predictions
  # in column 5 are for tree.list[5]=500 trees
  pred <- gbm :: predict.gbm(mod, presence.test,
                             n.trees = tree.list, 
                             type = "response")
  
  # Calculate deviance of all of these results and store them in a dataframe:
  for (j in 1:length(deviance.tc4[,i])) {
    
    deviance.tc4[,i][j] <- calc.deviance(presence.test$Presence,
                                         pred[,j], calc.mean = T)
    
  } # end of j loop
} # end of i loop


### Store and Graph the results ###

# Find the minimum value of predictive deviance
min_value <- min(deviance.tc4, na.rm=TRUE)

# Identify the row and column index of the min value
row_index <- which(deviance.tc4 == min_value, arr.ind = TRUE)[1, 1]
col_index <- which(deviance.tc4 == min_value, arr.ind = TRUE)[1, 2]

# Get the column name where the minimum value was found
min_col_name <- colnames(deviance.tc4)[col_index]

# Get the corresponding number of trees from the row of the min val
nt_value <- deviance.tc4[row_index, 'nt']

# Fill in the optimal values in the settings dataframe
settings[4, 'Learning.Rate'] <- as.numeric(paste0("0.", 
                                                  sub("lr", "", min_col_name)))
settings[4, 'Number.Trees'] <- nt_value

settings[4, 'Pred.Dev'] <- min_value

# Graph the results of the learning rate comparison for tc=4
tc4 <- ggplot() +
  geom_line(data = deviance.tc4, 
            aes(x = nt, y = lr05, linetype = "0.05"), linewidth = 1) +
  geom_line(data = deviance.tc4, 
            aes(x = nt, y = lr01, linetype = "0.01"), linewidth = 1) +
  geom_line(data = deviance.tc4, 
            aes(x = nt, y=lr005, linetype = "0.005"), linewidth = 1) +
  geom_line(data = deviance.tc4, 
            aes(x = nt, y = lr001, linetype = "0.001"), linewidth = 1) +
  geom_line(data = deviance.tc4, 
            aes(x = nt, y = lr0005, linetype = "0.0005"), linewidth = 1) +
  geom_hline(yintercept = min_value, linetype = "solid", color = "red",
             linewidth = 1.2) +
  geom_vline(xintercept = nt_value, 
             linetype = "solid", color = "green", linewidth = 1.2) +
  theme_bw() +
  theme(axis.text.x = element_text(size = 20, face = "bold")) +
  theme(axis.text.y = element_text(size = 20, face = "bold")) +
  theme(axis.title.x = element_text(size = 22, face = "bold", color = "gray30")) +
  theme(axis.title.y = element_text(size = 22, face = "bold", color = "gray30")) +
  scale_x_continuous(name = "\nNumber of Trees", limits = c(0, 20000),
                     breaks=c(0, 5000, 10000, 15000, 20000)) +
  scale_y_continuous(name = "Predictive Deviance\n", limits = c(0.5, 1.7)) +
  scale_linetype_manual(name = '', values = c('0.05' = "twodash", 
                                              '0.01' = "solid",
                                              '0.005' = "longdash",
                                              '0.001' = "dotted",
                                              '0.0005' = "dotdash"),
                        labels = c("0.0005", "0.001", "0.005", "0.01", "0.05")) +
  theme(aspect.ratio = 1) + #aspect ratio expressed as y/x
  # theme(legend.position = "none") +
  theme(legend.text = element_text(size=14)) +
  guides(colour = guide_legend(override.aes = (list(size=3))))



###############################################
###                                         ###
###   Determine the optimal model settings: ###
###              bag fraction               ###
###############################################

# We will systematically test different values for the bag
# fraction.

# Extract the optimal settings for tree complexity and
# learning rate resulting from previous step
tc.fin <- which(settings$Pred.Dev == min_value, arr.ind = TRUE)
lr.fin <- settings$Learning.Rate[settings$Tree.Complex == tc]

# Define the bag fraction values to test
bf <- seq(0.25, 0.75, by = 0.05)

# Create an empty dataframe in which to store the predicted deviances
deviance.bf<-data.frame(bf25=numeric(200), bf30=numeric(200), 
                        bf35=numeric(200), bf40=numeric(200),
                        bf45=numeric(200), bf50=numeric(200),
                        bf55=numeric(200), bf60=numeric(200),
                        bf65=numeric(200), bf70=numeric(200), 
                        bf75=numeric(200), nt=seq(100, 20000, by=100))

# Model each different learning rate and store them in a data frame
for (i in 1:length(bf)) {
  
  mod <-  dismo :: gbm.fixed(data=presence.train,
                   gbm.x = c(2:7), gbm.y = 1,
                   family = "bernoulli", 
                   tree.complexity = tc.fin, 
                   learning.rate = lr.fin,
                   n.trees = 20000, bag.fraction = bf[i])
  
  # Create matrix of predictions, each column = predictions from the model
  pred <- gbm :: predict.gbm(mod, presence.test, 
                             n.trees = tree.list, type="response")
  
  # Calculate deviance of all of these results and store them in a dataframe:
  for (j in 1:length(deviance.bf[,i])) {
    
    deviance.bf[,i][j] <- calc.deviance(presence.test$Presence, 
                                        pred[,j], calc.mean = T)
  } # end of j loop
} # end of i loop


### Store and Graph the results ###

# Find the minimum value of predictive deviance
min_value <- min(deviance.bf, na.rm=TRUE)

# Identify the row and column index of the min value
row_index <- which(deviance.bf == min_value, arr.ind = TRUE)[1, 1]
col_index <- which(deviance.bf == min_value, arr.ind = TRUE)[1, 2]

# Get the column name where the minimum value was found
min_col_name <- colnames(deviance.bf)[col_index]

# Get the corresponding number of trees from the row of the min val
nt_value <- deviance.bf[row_index, 'nt']

# Fill in the optimal values in the settings dataframe
settings$Bag.Frac[4] <- as.numeric(paste0("0.", 
                                          sub("bf", "", min_col_name)))
settings$Bf_nt[4] <- nt_value

settings$Bf_Pred.Dev[4] <- min_value


# Graph the results of the bag fraction comparison
bf <- ggplot() +
  geom_line(data = deviance.bf, 
            aes(x  =  nt, y = bf25, color = "bf25"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf30, color = "bf30"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf35, color = "bf35"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf40, color = "bf40"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf45, color = "bf45"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf50, color = "bf50"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf55, color = "bf55"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf60, color = "bf60"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf65, color = "bf65"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf70, color = "bf70"), linewidth = 1) +
  geom_line(data = deviance.bf, 
            aes(x = nt, y = bf75, color = "bf75"), linewidth = 1) +
  geom_hline(yintercept = min_value, linetype = "solid", color = "red",
             linewidth = 1.2) +
  geom_vline(xintercept = nt_value, linetype = "solid", color = "green",
             linewidth = 1.2) +
  theme_bw() +
  theme(axis.text.x = element_text(size = 20, face = "bold")) +
  theme(axis.text.y = element_text(size = 20, face = "bold")) +
  theme(axis.title.x = element_text(size = 22, face = "bold", color = "gray30")) +
  theme(axis.title.y = element_text(size = 22, face = "bold", color = "gray30")) +
  scale_x_continuous(name="\nNumber of Trees", limits = c(0, 20000),
                     breaks=c(0, 5000, 10000, 15000, 20000)) +
  scale_y_continuous(name="Predictive Deviance\n") +
  scale_colour_manual(name='', values = c('bf25' = "red", 
                                          'bf30' = "orange",
                                          'bf35' = "yellow",
                                          'bf40' = "green",
                                          'bf45' = "blue",
                                          'bf50' = "purple",
                                          'bf55' = "darkgreen",
                                          'bf60' = "turquoise2",
                                          'bf65' = "brown3",
                                          'bf70' = "darkgoldenrod3",
                                          'bf75' = "hotpink"), 
                      labels = c("0.25", "0.30", "0.35", "0.40", "0.45", "0.50",
                                 "0.55", "0.60", "0.65", "0.70", "0.75")) +
  theme(aspect.ratio=1) + #aspect ratio expressed as y/x
  theme(legend.text = element_text(size=14)) +
  guides(colour = guide_legend(override.aes = (list(size=3))))




###############################################
###                                         ###
###          Run the final model            ###
###                                         ###
###############################################

# Rerun test and training separation so that new random sets 
# (different from those used above in the optimal settings 
# identification) can be generated

# Select which rows to use for testing from the presence/absence data:
t.parows <- sample( x = 1:M, size = TP, replace = FALSE )

# Subset the complete dataset to exclude these rows thereby creating
# the training dataset (~70% of the data)
presence.train <- presence[ -t.parows, ]

# Create testing dataset:
presence.test <- presence[ t.parows, ]

# Optimal Settings
# tree.complexity already defined (tc.fin)
# learning.rate already defined (lr.fin)

# Number of trees should be maximum between those between step 1a
# and step 1b.
min_row_index <- which.min(settings$Pred.Dev)
nt.fin <- max(settings[min_row_index, "Number.Trees"],
              settings[min_row_index, "Bf_nt"], na.rm = TRUE)

bf.fin <- settings[min_row_index, "Bag.Frac"]


# Model with optimal settings
# We now use gbm.step which assesses the number of boosting trees
# using k-fold cross validation
mod.fin <-  dismo :: gbm.step(data=presence.train,
                               gbm.x = c(2:7), gbm.y = 1,
                               family = "bernoulli", 
                               tree.complexity = tc.fin, 
                               learning.rate = lr.fin,
                               max.trees = nt.fin, bag.fraction = bf.fin)



###############################################
###                                         ###
###           Model Evaluation              ###
###                                         ###
###############################################

# Make dataframe to store evaluation metrics
evaluation <- data.frame(AUC = character(1))

# Set threshold for determining whether predictions should be recoded
# as 0 (absence) or 1 (presence)
# Could adjust threshold depending on what we are trying to maximize, but
# this is for optimal sensitivity and specificity
threshold = 0.5

# Run predictions using the testing dataset
preds <- gbm :: predict.gbm(mod.fin, presence.test,
                               n.trees=mod.fin$gbm.call$best.trees,
                               type = "response")

#######################
###                 ###
###       AUC       ###
###                 ###
#######################

# Receiver Operating Characteristic Curve (AUC)

# Create matrix of actual values and predicted values
d <- cbind(presence.test$Presence, preds)

# Extract true presences and true absences
pres <- d[d[,1]==1, 2]
abs <- d[d[,1]==0, 2]

# Use evaluate function to calculate AUC
e <- dismo :: evaluate(p=pres, a=abs)

# Add value to evaluation dataframe
evaluation$AUC <- e@auc

# Plot for fun
plot(e, 'ROC')

#######################
###   Sensitivity,  ###
###   Specificity,  ###
###     and TSS     ###
#######################

# Recode predictions to be either 1 or 0
preds_new <- ifelse(preds > threshold, 1, 0)

# Create vector of actual values
actual_values <- presence.test$Presence


# Create a confusion matrix
conf_matrix <- base :: table(preds_new, actual_values)
conf_matrix

# Sensitivity
evaluation$sensitivity <- caret :: sensitivity(conf_matrix)

# Specificity
evaluation$specificity <- caret :: specificity(conf_matrix)

# TSS = sensitivity + specificity -1
evaluation$TSS <- caret :: sensitivity(conf_matrix) + 
  caret :: specificity(conf_matrix) - 1


#######################
###                 ###
###  Cohen's kappa  ###
###                 ###
#######################

# We will use weighted Cohen's kappa
Cohens <- psych :: cohen.kappa(conf_matrix)

evaluation$Cohens.wtd <- Cohens$weighted.kappa


#######################
###                 ###
###    Prevalence   ###
###                 ###
#######################


evaluation$Prevalence <- (as.numeric(conf_matrix[1, 2]) + 
                            as.numeric(conf_matrix[2, 2])) / sum(conf_matrix)





###############################################
###                                         ###
###                 Extras                  ###
###                                         ###
###############################################

###### Identifying interactions ######
find.int <- dismo :: gbm.interactions(mod.fin)
find.int$interactions
find.int$rank.list

gbm.perspec(mod.fin, 5, 3)


#Variable importance
mod.fin$contributions

#partial dependence plots
gbm.plot(mod.fin)

