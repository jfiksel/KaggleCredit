library(e1071)
library(rpart)
library(ada)
library(adabag)
library(ggplot2)
library(randomForest)
library(imputation)

Plot.Factor <- function(data) {
  # Plot.Factor plots a histogram comparing the percent
  # of our data that defaulted and the percent that did not.
  p1 <- ggplot(data) + 
    geom_bar(aes(x = factor(SeriousDlqin2yrs), y = ..count../sum(..count..))) + 
    ylab("Percent") + xlab("Classification")
  p1
}

Split.Data <-function(data) {
  # Split.Data partitions a full dataset into two smaller
  # ones that may be used for training/testing.
  n <- nrow(data)
  half.cleaned <- data[sample(n, n/1.5), ]
  train.rows <- sample(nrow(half.cleaned), floor(nrow(half.cleaned)*0.8))
  train.set <- half.cleaned[train.rows, 1:ncol(half.cleaned)]
  test.set <- half.cleaned[-train.rows, 1:ncol(half.cleaned)]
  return(list("train" = train.set, "test" = test.set))
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
boxplot(cs.training$RevolvingUtilizationOfUnsecuredLines, horizontal=T)
# boxplot of each variable. Note that 98/96 are the outliers for the categories
# mentioned above.
apply(cs.training, 2, boxplot)

# We should think about ways to weight observations where the person did 
# default since defaulting obs. are underrepresented in the data.
Plot.Factor(cs.training)

# We should impute all the data before we begin training/testing with it. 
# Here is a naive attempt that removes the offending observations from our data.
# We use a boosted regression tree imputation method
all.obs <- rbind(cs.training, cs.test)[, -1]  # Gets rid of the "X" column

# make sure we demarkate factor variables so we dont get nonsense values
# (like NumDepedents = 0.56) after imputation. 
all.obs$age <- ifelse(all.obs$age > 0, all.obs$age, NA)

all.obs$NumberOfTime30.59DaysPastDueNotWorse <- ifelse(all.obs$NumberOfTime30.59DaysPastDueNotWorse < 90,
                                                       all.obs$NumberOfTime30.59DaysPastDueNotWorse, NA)


all.obs$NumberOfTimes90DaysLate <- ifelse(all.obs$NumberOfTimes90DaysLate < 90, 
                                          all.obs$NumberOfTimes90DaysLate, NA)

all.obs$NumberOfTime60.89DaysPastDueNotWorse <- ifelse(all.obs$NumberOfTime60.89DaysPastDueNotWorse < 90,
                                                       all.obs$NumberOfTime60.89DaysPastDueNotWorse, NA)

# Should these all be categorical?
all.obs$age <- factor(all.obs$age)
all.obs$NumberOfTime30.59DaysPastDueNotWorse <- factor(all.obs$NumberOfTime30.59DaysPastDueNotWorse)
all.obs$NumberOfOpenCreditLinesAndLoans <- factor(all.obs$NumberOfOpenCreditLinesAndLoans)
all.obs$NumberOfTimes90DaysLate <- factor(all.obs$NumberOfTimes90DaysLate)
all.obs$NumberRealEstateLoansOrLines <- factor(all.obs$NumberRealEstateLoansOrLines)
all.obs$NumberOfTime60.89DaysPastDueNotWorse <- factor(all.obs$NumberOfTime60.89DaysPastDueNotWorse)
all.obs$NumberOfDependents <- factor(all.obs$NumberOfDependents)

# impute data using test data as well
impute.cleaned <- gbmImpute(all.obs[, -1], cv.fold = 10, n.trees = 500)
full.train <- cbind(SeriousDlqin2yrs = cs.training$SeriousDlqin2yrs, impute.cleaned$x[1:nrow(cs.training), ])

# TODO: after imputation, we should make one more pass over the data to fill in dependent
# fields; i.e. some columns are dependent on other columns and we should update
# accordingly after cleaning things up.

# If training takes too long, split training data up:
split.data <- Split.Data(full.train)
test <- split.data$test
train <- split.data$train

# let's check how defaults are distributed in these:
Plot.Factor(full.train)
Plot.Factor(train)

# TODO: Someone should tweak the settings of this random forest to see if
# it gives us any better predictions. May also consider how we weight predictions
# this gives us and what boosting gives us.
rf <- randomForest(as.factor(SeriousDlqin2yrs) ~ ., data= full.train)

# Boosting: as of now we are using weak classifiers in our boosting routine and only running for
# 100 iterations. We may want to play around with the types of classifiers that we use here
# and may also want to play around with how long we let these guys run for.
one.four <- rpart.control(cp = -1, maxdepth = 10, minsplit = 1)
sixty.four  <- rpart.control(cp = -1, maxdepth = 6, minsplit = 1)
thirty.two <- rpart.control(cp = -1, maxdepth = 5, minsplit = 1) #16-node tree
sixteen <- rpart.control(cp = -1, maxdepth = 4, minsplit = 1) #16-node tree
eight <- rpart.control(cp=-1,maxdepth=3, minsplit = 1) # 8-node tree
four <- rpart.control(cp = -1, maxdepth = 2, minsplit = 1) # 4-node tree
stump <- rpart.control(cp = -1, maxdepth = 1,minsplit = 1) # 2-node tree

boost <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = full.train, 
             iter = 1000, test.x = test[ , -1], test.y = test[ , 1])

boost.stump <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = full.train, 
                   control = stump, iter = 1000, test.x = test[ , -1], 
                   test.y = test[ , 1])

boost.four <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = full.train, 
                  control = four, iter = 100, test.x = test[ , -1], 
                  test.y = test[ , 1])

boost.eight <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = full.train, 
                   control = eight, iter = 100, test.x = test[ , -1], 
                   test.y = test[ , 1])

boost.sixteen <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = full.train, 
                   control = sixteen, iter = 100, test.x = test[ , -1], 
                   test.y = test[ , 1])

boost.thirty <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = full.train, 
                              control = thirty.two, iter = 100, 
                              test.x = test[ , -1], test.y = test[ , 1])

  boost.sixty <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = full.train, 
                      control = sixty.four, iter = 400, 
                      test.x = test[ , -1], test.y = test[ , 1])
  
  boost.ten <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = full.train, 
                   control = one.four, iter = 1000, 
                   test.x = test[ , -1], test.y = test[ , 1])

## performance plots
varplot(boost.stump)
plot(boost.stump)
varplot(boost.four)
plot(boost.four)
varplot(boost.eight)
plot(boost.eight)
varplot(boost.sixteen)
plot(boost.sixteen)

pred <- predict(boost.ten, cs.test[, -1], type = "probs")
results <- data.frame(Id = 1:nrow(cs.test), Probability = pred[, 2])
write.table(results, "derose.csv", quote=F, row.names=F, sep=",")

library(GA)
library(ROCR)

# pred1,pred2,pred3 are predicted probabilities from different
# models (e.g. boosting, random forest and svm)
# TODO: we need to obtain the actual data
# we can train these models with the training data
# except for a small subset and predict the subset
pred1 <- runif(1000)
pred2 <- rnorm(1000)
pred3 <- runif(1000)
# label is the vector of true values {0,1}
label <- runif(1000)
label[label < 0.5] <- 0
label[label > 0.5] <- 1

# The fitness function for the genetic algorithm
# calculates the AUC 
# w1,w2,w3 are weights
Fitness <- function(w1,w2) {
  w3 <- 1 - w1 - w2
  pred <- prediction(w1*pred1 + w2*pred2 + w3*pred3,label)
  auc <- performance(pred,"auc")
  auc <- unlist(slot(auc, "y.values"))
  return (1 - auc)
}

# The genetic algorithm finds the optimal weights
# The range of w1,w2 is [0,0.5]
# There are quite a few parameters we can play with
GA <- ga(type = "real-valued",fitness = 
  function(x) - Fitness(x[1],x[2]), 
  min = c(0,0), max = c(0.5,0.5))

summary(GA)
 
