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
temp <- cs.training[, -1] ##Gets rid of the "X" column
temp <- cs.training[cs.training$age > 0, ] 
temp <- temp[temp$NumberOfTime30.59DaysPastDueNotWorse < 90, ]
temp <- temp[temp$NumberOfTimes90DaysLate < 90, ]
temp <- temp[temp$NumberOfTime60.89DaysPastDueNotWorse < 90, ]
## I think we *can* use the response in our imputation. We are training
## on this, we might as well use all the information we have
impute.cleaned <- gbmImpute(temp, cv.fold = 4)$x[ , -1]
n <- nrow(naive.cleaned)
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

