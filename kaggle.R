library(e1071)
library(rpart)
library(ada)
library(adabag)
library(ggplot2)
library(randomForest)
library(imputation)
library(GA)
library(ROCR)

Plot.Factor <- function(data) {
  # Plot.Factor plots a histogram comparing the percent
  # of our data that defaulted and the percent that did not.
  p1 <- ggplot(data) + 
    geom_bar(aes(x = factor(SeriousDlqin2yrs), y = ..count../sum(..count..))) + 
    ylab("Percent") + xlab("Classification")
  p1
}

Split.Data <-function(data, size) {
  # Split.Data partitions a full dataset into two smaller
  # ones that may be used for training/testing.
  n <- nrow(data)
  prob <- ifelse(data$SeriousDlqin2yrs == 1, 0.15, 0.015)
  prob.sampled <- data[sample(n, n / size, prob = prob, replace = TRUE), ] 
  return(prob.sampled)
}

# Someone has age 0. We should take out of the set, unless we can come up 
# good interpolation. NumberOfTime30.59DaysPastDueNotWorse 
# contains "non-sense" data such as 96 and 98 use to code for some qualitative
# quantity ("other", or "refused to say"). The same corruption is present in 
# NumberOfTime60.89DaysPastDueNotWorse and NumberOfTimes90DaysLate
# See tabulated values using the raw data from cs-training.csv
table(cs.training$age)
table(cs.training$NumberOfTime30.59DaysPastDueNotWorse)
table(cs.training$NumberOfTimes90DaysLate)
table(cs.training$NumberOfTime60.89DaysPastDueNotWorse)

# TODO: check for more outliers like this:
boxplot(cs.training$RevolvingUtilizationOfUnsecuredLines, horizontal = T)
# boxplot of each variable. Note that 98/96 are the outliers for the categories
# mentioned above.
apply(cs.training, 2, boxplot)

# We should think about ways to weight observations where the person did 
# default since defaulting obs. are underrepresented in the data.
Plot.Factor(cs.training)

# We should impute all the data before we begin training/testing with it. 
# Here is a naive attempt that removes the offending observations from our data.
# We use a boosted regression tree imputation method
all.obs <- rbind(cs.training, cs.test)[ , -1]  # Gets rid of the "X" column

# filter nonsense variables
all.obs$age <- ifelse(all.obs$age > 0, all.obs$age, NA)

all.obs$RevolvingUtilizationOfUnsecuredLines <- ifelse(all.obs$RevolvingUtilizationOfUnsecuredLines <= 1,
                                                       all.obs$RevolvingUtilizationOfUnsecuredLines, NA)

all.obs$NumberOfTime30.59DaysPastDueNotWorse <- ifelse(all.obs$NumberOfTime30.59DaysPastDueNotWorse < 90,
                                                       all.obs$NumberOfTime30.59DaysPastDueNotWorse, NA)


all.obs$NumberOfTimes90DaysLate <- ifelse(all.obs$NumberOfTimes90DaysLate < 90, 
                                          all.obs$NumberOfTimes90DaysLate, NA)

all.obs$NumberOfTime60.89DaysPastDueNotWorse <- ifelse(all.obs$NumberOfTime60.89DaysPastDueNotWorse < 90,
                                                       all.obs$NumberOfTime60.89DaysPastDueNotWorse, NA)

all.obs$DebtRatio <- ifelse(is.na(all.obs$MonthlyIncome), NA, all.obs$DebtRatio)


# impute data using test and training data; ignore response
impute.cleaned <- gbmImpute(all.obs[, -1], cv.fold = 5, max.iters = 3)
knn.impute <- kNNImpute(all.obs[, -1], cv.fold = 5, max.iters = 3)
full.train <- cbind(SeriousDlqin2yrs = cs.training$SeriousDlqin2yrs, # add reponse back to dataframe
                    impute.cleaned$x[1:nrow(cs.training), ]) 
full.test <- cbind(SeriousDlqin2yrs = cs.test$SeriousDlqin2yrs,
                   impute.cleaned$x[c(-1:-nrow(cs.training)), ])

# If training takes too long, split training data up:
unskew.data <- Split.Data(full.train, 1)

# let's check how defaults are distributed in these:
Plot.Factor(unskew.data)
Plot.Factor(all.obs[0:nrow(cs.training), ])

# TODO: Someone should tweak the settings of this random forest to see if
# it gives us any better predictions. May also consider how we weight predictions
# this gives us and what boosting gives us.
rf <- randomForest(as.factor(SeriousDlqin2yrs) ~ ., data = unskew.data)

# Naive Bayes
nb <- naiveBayes(as.factor(SeriousDlqin2yrs) ~ ., data = unskew.data)

# Boosting: as of now we are using weak classifiers in our boosting routine and 
# only running for 100 iterations.; may want to bump that up if we see
# the need to
thirty <- rpart.control(cp = -1, maxdepth = 5, minsplit = 1) #16-node tree
sixteen <- rpart.control(cp = -1, maxdepth = 4, minsplit = 1) #16-node tree
eight <- rpart.control(cp=-1,maxdepth=3, minsplit = 1) # 8-node tree
four <- rpart.control(cp = -1, maxdepth = 2, minsplit = 1) # 4-node tree
stump <- rpart.control(cp = -1, maxdepth = 1,minsplit = 1) # 2-node tree

boost <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = unskew.data, iter = 100)

boost.stump <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = unskew.data, 
                   control = stump, iter = 100)

boost.four <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = unskew.data, 
                  control = four, iter = 100)

boost.eight <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = unskew.data, 
                   control = eight, iter = 100)

boost.sixteen <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = unskew.data, 
                   control = sixteen, iter = 100)

boost.thirty  <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = unskew.data, 
                     control = thirty, iter = 100)


boost.train.pred <- predict(boost.sixteen, full.train, type = "prob")
forest.train.pred <- predict(rf, full.train, type = "prob" )
bayes.train.pred <- predict(nb, full.train, type = "raw")
all.pred <- data.frame(bayes.train.pred[, 2], forest.train.pred[ , 2], 
                       boost.train.pred[ , 2])
  
# The fitness function for the genetic algorithm
# calculates the AUC 
# w1,w2,w3 are weights
Fitness <- function(x, guess) {
  w <- x / sum(x)
  pred <- prediction(as.matrix(guess) %*% w , full.train[, 1])
  auc <- performance(pred, "auc")
  auc <- unlist(slot(auc, "y.values"))
  return (1 - auc)
}

# The genetic algorithm finds the optimal weights
# The range of w1,w2 is [0,0.5]
# There are quite a few parameters we can play with
GA <- ga(type = "real-valued",fitness = Fitness, maxiter = 10, guess = all.pred,
         min = c(0, 0, 0), max = c(1,1,1))
summary(GA)
weights <- c(0.6520063, 0.006869555, 0.3802973)
weights <- weights / sum(weights)

boost.test.pred <- predict(boost.sixteen, full.test, type = "prob")
forest.test.pred <- predict(rf, full.test, type = "prob" )
bayes.test.pred <- predict(nb, full.test, type = "raw")
all.pred <- data.frame(bayes.test.pred[, 2], forest.test.pred[ , 2], 
                       boost.test.pred[ , 2])
final.test.pred <- as.matrix(all.pred) %*% weights
results <- data.frame(Id = 1:nrow(cs.test), Probability = final.test.pred)
write.table(results, "imputed.csv", quote=F, row.names=F, sep=",")

