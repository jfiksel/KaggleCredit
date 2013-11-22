library(e1071)
library(rpart)
library(ada)
library(ggplot2)
data(kag_subset)
set.seed(as.numeric(Sys.time()))
trainingNoNA <- na.omit(cs.training)
training.rows <-sample(nrow(cs.training), nrow(cs.training)/2)
train.sub <- na.omit(cs.training[training.rows, 2:ncol(cs.training)])
test.sub <- na.omit(cs.training[-training.rows, 2:ncol(cs.training)])

### SVM
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
rf.fit <- randomForest(SeriousDlqin2yrs~., data = train.sub, ntrees = 500)

### boosting
eight <- rpart.control(cp=-1,maxdepth=3,minsplit=0)
four <- rpart.control(cp=-1,maxdepth=2,minsplit=0)
stump <- rpart.control(cp=-1,maxdepth=1,minsplit=0)

boost.fit <- ada(as.factor(SeriousDlqin2yrs)~., data=train.sub, 
                 type = "real", control = four, 
                 test.x = test.sub[, -1], test.y = test.sub[, 1], iter = 100)
