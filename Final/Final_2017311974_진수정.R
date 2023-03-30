# ----- Question 1. ----- #

### (1)
Q1 = read.csv('Q1.csv')
dim(Q1)
head(Q1)
lm.fit = lm(Y ~ ., data = Q1)
summary(lm.fit)
plot(lm.fit)
dwtest(lm.fit)

# X2: X2의 p-value가 0.05보다 크기 때문이다. 
# 이 때, Linear regression model은 constant variance, independence, normality 조건을 모두 충족한다.

### (2) 
gam.fit = gam(Y ~ s(X1,5) + s(X2,5) + s(X3,5), data = Q1)
plot(gam.fit)
ggplot(Q1) + geom_point(aes(x = X1,y = Y))
ggplot(Q1) + geom_point(aes(x = X2,y = Y))
ggplot(Q1) + geom_point(aes(x = X3,y = Y))
gam2 = gam(Y ~ X1 + X2 + s(X3,5), data = Q1)
anova(gam.fit,gam2)

# X2는 (1)에 의해 유의하지 않은 것으로 확인되었으므로 제거하였다.
# GAM 결과 X3은 Quadratic term이 적절해보이므로 이를 추가하였다.

final.fit = lm(Y ~ X1 + X3 + I(X3^2), data = Q1)
summary(final.fit)

# Y = 31.6220 + 4.8742 * X1 + 13.8073 * X3 - 0.9626 * X3^2

### (3)
plot(final.fit)
# We need Linear Variance function

### (4)

f = function(beta,X) {
  X1 = X[,1]; X3 = X[,2]
  beta[1] + beta[2]*X1 + beta[3]*X3 + beta[4]*X3^2
}

RSS = function(beta,X,Y,S) {
  t(Y - f(beta,X)) %*% solve(S) %*% (Y - f(beta,X))
}

grv = function(beta,X,Y,S) {
  X1 = X[,1]; X2 = X[,2]
  sigma2 = diag(S)
  R = Y - f(beta,X)
  c(-2*sum(R/sigma2),-2*sum(R*X1/sigma2),-2*sum(R*X3/sigma2),-2*sum(R*X3^2/sigma2))
}

X = data.frame('X1' = Q1$X1,'X3' = Q1$X3)
Y = Q1$Y

beta.new = final.fit$coefficients
W = diag(rep(1,nrow(Q1)))
mdif = 100000

library(matrixcalc)
while (mdif > 0.000001) {
  Yhat = f(beta.new,X)
  r = Y - Yhat
  Z = cbind(1,Yhat)
  gam.hat = solve(t(Z)%*%W%*%Z) %*% t(Z)%*%W%*%abs(r)
  sigma = Z %*% gam.hat
  S = diag(as.vector(sigma^2))
  if (is.non.singular.matrix(S)) W = solve(S)
  else W = solve(S + 0.000000001 * diag(rep(1,nrow(S))))
  
  ml2 = optim(beta.new, RSS, gr = grv, method = 'BFGS', X = X, Y = Y, S = S)
  beta.old = beta.new
  beta.new = ml2$par
  mdif = max(abs(beta.new - beta.old))
}
Yhat = f(beta.new,X)
sigma = Z %*% gam.hat
r = (Y - Yhat) / sigma
plot(Yhat,r)

beta.new
gam.hat
# Y = 31.62 + 4.87 * X1 + 13.81 * X3 - 0.96 * X3^2 + sigma * g * N(0,1)
# where g = 2.75 + 0.12 * Yhat


# ----- Question 2. ----- #

Q2 = read.csv('Q2.csv',header = T)
dim(Q2)
head(Q2)

# 10개의 변수를 2차원으로 줄이기 위해서 dimension reduction 기법을 사용하였다.
# 이 때, 2차원 공간 상에 나타내기 위해서 2개의 PC만 선택하였다.
pr = prcomp(Q2[,-1],center = T,scale = T)
summary(pr)
screeplot(pr,type = 'l')
pca.dat = data.frame(Q2,PC1 = pr$x[,1],
                     PC2 = pr$x[,2])
plot(cbind(pca.dat$PC1,pca.dat$PC2),col = (as.numeric(pca.dat$Y + 1)))

# kpc = kpca(~., data = Q2[,-1], kernel = 'rbfdot', kpar = list(sigma = 3), features = 2)
# pc2 = pcv(kpc)
# pca.dat = data.frame(Q2,KPC1 = pc2[,1],
#                      KPC2 = pc2[,2])
# plot(cbind(pca.dat$KPC1,pca.dat$KPC2),col = (as.numeric(pca.dat$Y + 1)))


# ----- Question 3. ----- #

Q3 = read.csv('Q3.csv',header = T)
dim(Q3)
head(Q3)

### (1)
library(CORElearn)
# Y가 categorical인 classification 문제이므로, Relief 알고리즘 사용
RE = attrEval(Y ~ ., data = Q3, estimator = 'Relief',
              ReliefIterations = 30)
SRE = sort(RE,decreasing = T)
SRE[1:5] # 1888 47 1263 385 1355

### (2)
library(SIS)
# High dimensional한 case이므로 SIS 기법을 사용하였다.
model11 = SIS(x = as.matrix(Q3[,-1]),y = Q3$Y,family = 'binomial',tune = 'bic',penalty = 'SCAD')
model11$ix
model12 = SIS(x = as.matrix(Q3[,-1]),y = Q3$Y,family = 'binomial',tune = 'bic',penalty = 'SCAD',perm = T,q = 0.9)
model12$ix # 95 833 979 1055 1058

### (3)
# Variable importance는 Y와 highly correlated되어 있지만, 변수들 간에도 highly correlated 되어 있는 케이스를 잘 잡아내지 못하고, 높은 variable importance 값을 부여한다.
# 하지만 SIS는 filtering과 regularization을 모두 수행하기 때문에, redundent한 변수들을 regularization 과정에서 걸러낼 수 있다.


# ----- Question 4. ----- #
train = read.csv("Q4train.csv",header = T)
test = read.csv("Q4test.csv",header = T)
dim(train)
head(train)
table(train$Y)
describe(train)
dim(smtrain)

fmeasure = function(cm) {
  TPR = cm[2,2] / sum(cm[2,])
  PPV = cm[2,2] / sum(cm[,2])
  return((2*TPR*PPV) / (TPR+PPV))
}

# imbalanced data이므로, SMOTE를 사용하였다.
smtrain = SMOTE(Y ~ ., data = train)

# EDA : visualization
ggplot(smtrain) + geom_density(aes(x = X1,col = Y))
ggplot(smtrain) + geom_density(aes(x = X2,col = Y))
ggplot(smtrain) + geom_density(aes(x = X3,col = Y))
ggplot(smtrain) + geom_density(aes(x = X4,col = Y))
ggplot(smtrain) + geom_density(aes(x = X5,col = Y))
ggplot(smtrain) + geom_density(aes(x = X6,col = Y))
ggplot(smtrain) + geom_density(aes(x = X7,col = Y))
ggplot(smtrain) + geom_density(aes(x = X8,col = Y))
ggplot(smtrain) + geom_density(aes(x = X9,col = Y))
ggplot(smtrain) + geom_density(aes(x = X10,col = Y))
ggplot(smtrain) + geom_density(aes(x = X11,col = Y))
ggplot(smtrain) + geom_density(aes(x = X12,col = Y))

# Example : train -----
pr = prcomp(smtrain[,-1],center = T,scale = T)
summary(pr)
screeplot(pr,type = 'l') # 1-3

smtrain2 = smtrain
smtrain2 = smtrain2 %>%
  mutate(PC1 = pr$x[,1],
         PC2 = pr$x[,2],
         PC3 = pr$x[,3],
         PC4 = pr$x[,4])

pr2 = principal_curve(as.matrix(smtrain[,-1]))

smtrain2 = smtrain2 %>% 
  mutate(pcurve = pr2$lambda)

fit = kpca(~., data = smtrain[,-1], kernel = 'rbfdot', kpar = list(sigma = 3), features = 2)
pr3 = pcv(fit)

smtrain2 = smtrain2 %>%
  mutate(KPC1 = pr3[,1],
         KPC2 = pr3[,2])

km2 = kmeans(smtrain[,-1],2,nstart = 20)
plot(cbind(smtrain2$PC1,smtrain2$PC2),col = (km2$cluster + 1),pch = 20,cex = 2)
plot(cbind(imp.dat[[1]]$PC1,imp.dat[[1]]$PC2),col = (as.numeric(imp.dat[[1]]$Y) + 1),pch = 20,cex = 2)

km3 = kmeans(smtrain[,-1],3,nstart = 20)
plot(cbind(smtrain2$PC1,smtrain2$PC2),col = (km3$cluster + 1),pch = 20,cex = 2)

smtrain2 = smtrain2 %>% 
  mutate(clus2 = km2$cluster)

# Feature Engineering
dat = rbind(smtrain,test)
pr = prcomp(dat[,-1],center = T,scale = T)
pr2 = principal_curve(as.matrix(dat[,-1]))
fit = kpca(~., data = dat[,-1], kernel = 'rbfdot', kpar = list(sigma = 3), features = 2)
pr3 = pcv(fit)
km2 = kmeans(dat[,-1],2,nstart = 20)
dat = dat %>%
  mutate(PC1 = pr$x[,1],
         PC2 = pr$x[,2],
         PC3 = pr$x[,3],
         PC4 = pr$x[,4],
         pcurve = pr2$lambda,
         KPC1 = pr3[,1],
         KPC2 = pr3[,2]
         )

train2 = dat[1:nrow(smtrain),]
test2 = dat[(nrow(smtrain)+1):nrow(dat),]

# Logistic
lr.fit = glm(Y ~ ., data = train2, family = binomial)
phat.test = predict(lr.fit,test2,type = 'response')
yhat.test = ifelse(phat.test > 0.5,1,0)
cm = table(true = test2$Y,predict = yhat.test)
fmeasure(cm) # 0.2799

# LDA
library(MASS)
lda.fit = lda(Y ~ ., data = train2)
yhat.te = predict(lda.fit,test2)$class
cm = table(true = test2$Y,predict = yhat.te)
fmeasure(cm)

# LASSO
library(glmnet)
grid = 10^seq(-3,3,0.5)
tr.x = model.matrix(Y ~ ., data = train2)[,-1]
tr.y = train2$Y
te.x = model.matrix(Y ~ ., data = test2)[,-1]
te.y = test2$Y
cv.lasso = cv.glmnet(tr.x,tr.y,family = 'binomial',alpha = 1,lambda = grid)
opt.lasso = glmnet(tr.x,tr.y,family = 'binomial',alpha = 1,lambda = cv.lasso$lambda.min)
phat.test2 = predict(opt.lasso,te.x,type = 'response')
yhat.test2 = ifelse(phat.test2 > 0.5,1,0)
cm3 = table(true = test2$Y,predict = yhat.test2)
fmeasure(cm3) # 0.2779

# Random Forest
param_df = data.frame(
  mtry = rep(c(2,3,4,5),5),
  ntree = rep(c(200,400,600,800,1000),each = 4),
  accuracy = rep(NA,20)
)

library(caret)
set.seed(1)
cv = createFolds(train2$Y, k = 3)
library(randomForest)

for (i in 1:nrow(param_df)) {
  temp_acc = NULL
  
  for (j in 1:3) {
    valid_idx = cv[[j]]
    cv_test = train2[valid_idx,]
    cv_train = train2[-valid_idx,]
    
    set.seed(1)
    temp_fit = randomForest(Y ~., cv_train, mtry = param_df[i,'mtry'],ntree = param_df[i,'ntree'])
    temp_pred = predict(temp_fit,newdata = cv_test)
    cm = table(true = cv_test$Y,predict = temp_pred)
    temp_acc[j] = fmeasure(cm)
  }
  param_df[i,'accuracy'] = mean(temp_acc)
}
param_df %>% arrange(desc(accuracy))

set.seed(1)
rf.fit = randomForest(Y ~., train2, mtry = 3,ntree = 200)
set.seed(1)
rf.fit2 = randomForest(Y ~., train2, mtry = 5,ntree = 800)
set.seed(1)
rf.fit2 = randomForest(Y ~., train2, mtry = 4,ntree = 600)
set.seed(1)
rf.fit2 = randomForest(Y ~., train2, mtry = 3,ntree = 200)
yhat.test4 = predict(rf.fit,newdata = test2)
yhat.test5 = predict(rf.fit2,newdata = test2)
yhat.test6 = predict(rf.fit2,newdata = test2)
yhat.test7 = predict(rf.fit2,newdata = test2)
cm4 = table(true = test2$Y,predict = yhat.test4)
cm5 = table(true = test2$Y,predict = yhat.test5)
cm6 = table(true = test2$Y,predict = yhat.test6)
cm7 = table(true = test2$Y,predict = yhat.test7)
fmeasure(cm4) # 0.3861
fmeasure(cm5) # 0.3804
fmeasure(cm6) # 0.3887

# SVM
cv.svm = tune(svm, Y ~ ., data = train2, kernel = 'radial',
              ranges = list(cost = c(0.001,0.01,0.1,1,5,10,100),
                            gamma = c(0.01,0.1,0.5,1,2,3,4)))
opt.svm = cv.svm$best.model
yhat.test6 = predict(opt.svm,test2)
cm6 = table(true = test2$Y,predict = yhat.test6)
fmeasure(cm6)
