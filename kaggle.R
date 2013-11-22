library(e1071)
library(rpart)
library(ada)
library(ggplot2)

## we should do some data clean up. There's a guy with age 0 that we should probably take out of the set entirely
## when we look at cs.training$NumberOfTime30.59DaysPastDueNotWorse a value of 96/98 coded for some qualitative
## quantity ("other", or "refused to say") and so we shoud clean these up because they skew the data.
## should think of a way to weight observations where the person did default 


# split data in half and sample from it. Our training/test data is all pulled from cs.training, each
# subset representing about a quarter of the data in total
cs.training.half <-cs.training[sample(nrow(cs.training), nrow(cs.training)/2), ]
sample.rows <- sample(nrow(cs.training.half), nrow(cs.training.half)/2) 
train.sub <- na.omit(cs.training.half[sample.rows, 2:ncol(cs.training.half)])
test.sub <- na.omit(cs.training.half[-sample.rows, 2:ncol(cs.training.half)])

### SVM: these take a long time too. The first call to tune.svm is a cross-validation step
### so that we are using the optimal parameters to train with
if (false) {
  tuned <- tune.svm(as.factor(SeriousDlqin2yrs)~., data = train.sub,
                    gamma = 10^(-6:2), cost = 10^(1:3), kernel="polynomial")

  svm_fit <- svm(as.factor(SeriousDlqin2yrs)~., data = train.sub, method="C-classification", 
               kernel="sigmoid",cross=10, cost=10, gamma=1e-4, probability=T)

  svm_predict <- as.vector(predict(svm_fit, newdata=cs.training[, 3:ncol(cs.test)]))
  
  sum(ifelse(svm_predict == cs.training[2], 1 , 0)) / nrow(cs.training)
}


### random forest
library(randomForest)
# this will take a while
rf.fit <- randomForest(SeriousDlqin2yrs~., data = train.sub, ntrees = 500)

### boosting: as of now we are using weak classifiers in our boosting routine and only running for
### 100 iterations. We may want to play around with the types of classifiers that we use here
### and may also want to play around with how long we let these guys run for.
sixteen <- rpart.control(cp=-1,maxdepth=4,minsplit=0) #16-node tree
eight <- rpart.control(cp=-1,maxdepth=3,minsplit=0) # 8-node tree
four <- rpart.control(cp=-1,maxdepth=2,minsplit=0) # 4-node tree
stump <- rpart.control(cp=-1,maxdepth=1,minsplit=0) # 2-node tree

boost.stump <- ada(as.factor(SeriousDlqin2yrs)~., data=train.sub, 
                       type = "real", control = stump, 
                       test.x = test.sub[, -1], test.y = test.sub[, 1], iter = 100)

boost.four <- ada(as.factor(SeriousDlqin2yrs)~., data=train.sub, 
                       type = "real", control = four, 
                       test.x = test.sub[, -1], test.y = test.sub[, 1], iter = 100)
boost.eight <- ada(as.factor(SeriousDlqin2yrs)~., data=train.sub, 
                 type = "real", control = eight, 
                 test.x = test.sub[, -1], test.y = test.sub[, 1], iter = 100)
boost.sixteen <- ada(as.factor(SeriousDlqin2yrs)~., data=train.sub, 
                       type = "real", control = sixteen, 
                       test.x = test.sub[, -1], test.y = test.sub[, 1], iter = 200)
varplot(boost.fit)