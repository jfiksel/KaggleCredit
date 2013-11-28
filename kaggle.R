library(e1071)
library(rpart)
library(ada)
library(adabag)
library(ggplot2)
library(randomForest)
library(imputation)

## Someone has age 0. We should take out of the set, unless we can come up 
## good interpolation. NumberOfTime30.59DaysPastDueNotWorse 
## contains "non-sense" data such as 96 and 98 use to code for some qualitative
## quantity ("other", or "refused to say"). The same corruption is present in 
## NumberOfTime60.89DaysPastDueNotWorse and NumberOfTimes90DaysLate
## See tabulated values using the raw data from cs-training.csv
table(cs.training$age)
table(cs.training$NumberOfTime30.59DaysPastDueNotWorse)
table(cs.training$NumberOfTimes90DaysLate)
table(cs.training$NumberOfTime60.89DaysPastDueNotWorse)

## DATA CLEANUP TODO:
## We should think about ways to weight observations where the person did 
## default since defaulting obs. are underrepresented in the data.
p1 <- ggplot(cs.training) + 
  geom_bar(aes(x = factor(SeriousDlqin2yrs), y = ..count../sum(..count..))) + 
  ylab("Percent") + xlab("Classification")
p1

## We should impute all the data before we begin training/testing with it. 
## Here is a naive attempt that removes the offending observations from our data.
## We use a boosted regression tree imputation method
all.obs <- rbind(cs.training, cs.test)[, -1] ##Gets rid of the "X" column
View(all.obs)

## make sure you demarkate factor variable
# replace nonsense data with NA, to be filled in by imputation
all.obs$age <- ifelse(all.obs$age == 0, NA, all.obs$age)
all.obs$age <- as.factor(all.obs$age)


all.obs$NumberOfTime30.59DaysPastDueNotWorse <- ifelse(all.obs$NumberOfTime30.59DaysPastDueNotWorse > 90,
                                                       NA, all.obs$NumberOfTime30.59DaysPastDueNotWorse)
all.obs$NumberOfTime30.59DaysPastDueNotWorse <- as.factor(all.obs$NumberOfTime30.59DaysPastDueNotWorse)


all.obs$NumberOfTimes90DaysLate <- ifelse(all.obs$NumberOfTimes90DaysLate > 90, 
                                          NA, all.obs$NumberOfTimes90DaysLate)
all.obs$NumberOfTimes90DaysLate <- as.factor(all.obs$NumberOfTimes90DaysLate)

all.obs$NumberOfTime60.89DaysPastDueNotWorse <- ifelse(all.obs$NumberOfTime60.89DaysPastDueNotWorse > 90,
                                                       NA, all.obs$NumberOfTime60.89DaysPastDueNotWorse)
all.obs$NumberOfTime60.89DaysPastDueNotWorse <- as.factor(all.obs$NumberOfTime60.89DaysPastDueNotWorse)



impute.cleaned <- gbmImpute(all.obs[-1, ], cv.fold = 4, n.trees = 200, max.iters = 4)

## after we impute, we should make one more pass over the data to fill in dependent
## fields; i.e. some columns are dependent on other columns and we should update
## accordingly after cleaning things up.
n <- nrow(impute.cleaned)
half.cleaned <- impute.cleaned[sample(n, n/1.5), ]

# split data in half and sample from it.
train.rows <- sample(nrow(half.cleaned), floor(nrow(half.cleaned)*0.8))
train.set <- half.cleaned[train.rows, 1:ncol(half.cleaned)]
test.set <- half.cleaned[-train.rows, 1:ncol(half.cleaned)]

## proof we actally removed offending samples
table(train.set$age)
table(train.set$NumberOfTime30.59DaysPastDueNotWorse)
table(train.set$NumberOfTimes90DaysLate)
table(train.set$NumberOfTime60.89DaysPastDueNotWorse)

# let's check how defaults are distributed in these:
p2 <- ggplot(half.cleaned) + 
  geom_bar(aes(x = factor(SeriousDlqin2yrs), y = ..count../sum(..count..))) + 
  ylab("Percent") + xlab("Classification")
p3 <- ggplot(train.set) + 
  geom_bar(aes(x = factor(SeriousDlqin2yrs), y = ..count../sum(..count..))) + 
  ylab("Percent") + xlab("Classification")
p2
p3

### Boosting: as of now we are using weak classifiers in our boosting routine and only running for
### 100 iterations. We may want to play around with the types of classifiers that we use here
### and may also want to play around with how long we let these guys run for.
ada.cv <- boosting.cv(SeriousDlqin2yrs ~ ., data = train.set, control = stump)

sixteen <- rpart.control(cp=-1,maxdepth=4,minsplit=1) #16-node tree
eight <- rpart.control(cp=-1,maxdepth=3,minsplit=1) # 8-node tree
four <- rpart.control(cp=-1,maxdepth=2,minsplit=1) # 4-node tree
stump <- rpart.control(cp= -1, maxdepth=1,minsplit=1) # 2-node tree


boost <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = train.set, 
             iter = 200, test.x = test.set[ , -1], test.y = test.set[ , 1])

boost.stump <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = train.set, 
                   control = stump, iter = 200, test.x = test.set[ , -1], 
                   test.y = test.set[ , 1])

boost.four <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = train.set, 
                  control = four, iter = 200, test.x = test.set[ , -1], 
                  test.y = test.set[ , 1])

boost.eight <- ada(as.factor(SeriousDlqin2yrs) ~ ., data = train.set, 
                   control = eight, iter = 200, test.x = test.set[ , -1], 
                   test.y = test.set[ , 1])

boost.sixteen <- ada(as.factor(SeriousDlqin2yrs) ~ ., data=train.set, 
                     control = sixteen, iter = 200, test.x = test.set[ , -1],
                     test.y = test.set[ , 1])
## performance plots
varplot(boost.stump)
plot(boost.stump)
varplot(boost.four)
plot(boost.four)
varplot(boost.eight)
plot(boost.eight)
varplot(boost.sixteen)
plot(boost.sixteen)

pred <- predict(boost.four, cs.test[, -1], type = "probs")
results <- data.frame(Id = 1:nrow(cs.test), Probability = pred[, 2])
write.table(results, "derose.csv", quote=F, row.names=F, sep=",")
