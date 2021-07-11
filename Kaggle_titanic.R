# The script reads the data of the titanic data soruce from kaggle and provide the clearance


#library calls.
library(ggplot2)
library(dplyr)
library(datasets)
library(moments)
library(readxl)
library(writexl)
library(pivottabler)
library(tidyverse)
library(stringr)
library(ROCR)
library(InformationValue)
library("caret")

# settings working directory

setwd("D:\\OneDrive\\My Drive\\IPBA course\\R data sets\\Kaggle_Titanic")

#read the dataset 
titanic_master = read.csv("train.csv",
                          strip.white = TRUE,
                          na.strings = "NotApplied")

titanic_test_set = read.csv("test.csv",
                          strip.white = TRUE,
                          na.strings = "NotApplied")

titanic_test_set$Pclass   = as.factor(titanic_test_set$Pclass)
titanic_test_set$Sex      = as.factor(titanic_test_set$Sex)



#view the data set 
#View(titanic_master)
colSums(is.na(train_titanic))
train_titanic= titanic_master[1:800,]
test_titanic= titanic_master[801:889,]
str(train_titanic)
summary(train_titanic)


train_titanic$Age[is.na(train_titanic$Age)] = mean(train_titanic$Age,na.rm=T)
titanic_test_set$Age[is.na(titanic_test_set$Age)] = mean(titanic_test_set$Age,na.rm=T)
test_titanic$Age[is.na(test_titanic$Age)] = mean(test_titanic$Age,na.rm=T)

train_titanic$Fare[is.na(train_titanic$Fare)] = mean(train_titanic$Fare,na.rm=T)
titanic_test_set$Fare[is.na(titanic_test_set$Fare)] = mean(titanic_test_set$Fare,na.rm=T)
test_titanic$Fare[is.na(test_titanic$Fare)] = mean(test_titanic$Fare,na.rm=T)

colSums(is.na(train_titanic))
colSums(is.na(titanic_test_set))

train_titanic$Pclass = as.factor(train_titanic$Pclass)
train_titanic$Sex= as.factor(train_titanic$Sex)



#Lets see if we have misisng values? 
colSums(is.na(train_titanic))

# I want to extract the mr. mrs values from the name
train_titanic$title = str_extract(train_titanic$Name,"\\s\\w+\\.")
test_titanic$title = str_extract(train_titanic$Name,"\\s\\w+\\.")
titanic_test_set$title = str_extract(titanic_test_set$Name,"\\s\\w+\\.")


#Add total 
colnames(train_titanic)

#titanic_group = group_by(train_titanic,title)
#summarise(titanic_group,sum(Survived))

#model now
null_model = glm(Survived~1, family = "binomial", data = train_titanic)

Logit_model_interaction = glm(Survived ~ (SibSp * Age * Parch * Fare  )+
                              as.factor(Embarked)+as.factor(Sex)+ as.factor(Pclass)+as.factor(Name),
                              family = binomial(link = 'logit'),data = train_titanic)


full_model = Logit_model_interaction #glm(Survived~., family = "binomial", data = train_titanic)

summary(full_model)
colnames(train_titanic)

aic = step(null_model,
          scope=list(lower = null_model, upper = full_model),
          direction = "forward", 
          k = 2)


#manually select the fields on the gut feeling. Not cehcking on the interaction terms. 
Logit_model_first = glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked,
                        family = binomial(link = 'logit'),data = train_titanic)

#Picking the values on the AIC STEP terms
Logit_model_second = glm(aic$terms,
                        family = binomial(link = 'logit'),data = train_titanic)

#Copying the values and trying to add more details. 
Logit_model_third = glm(Survived ~ as.factor(Sex) + as.factor(Pclass) + Age + SibSp + 
                          Age:SibSp,
                        family = binomial(link = 'logit'),data = train_titanic)

#lets test summary

summary(Logit_model_second) #best numbers yet


#Lets run the ANOVA model. 

anova(Logit_model_second, test= 'Chisq')


#Any pattern? 

residual_second_model = residuals(Logit_model_second,type = 'pearson')
plot(residual_second_model)

#No, lets get to predict then

predicted_values_second = predict(Logit_model_second, newdata = test_titanic, type= 'response')

summary(predicted_values_second)
table(predicted_values_second)
str(predicted_values_second)

table(test_titanic$Survived)

#test_titanic$Predicted = ifelse(predicted_values_second>0.5,1,0)

#Lets find the optimal values
optimal = optimalCutoff(test_titanic$Survived,predicted_values_second)[1]
optimal

#ROCR curve then
#ROCR
Predict_values = prediction(predicted_values_second,test_titanic$Survived)
Predict_values_formula = performance(Predict_values ,measure = "tpr", x.measure = "fpr")
plot(Predict_values_formula);



#Lets check the data
table(test_titanic$Survived)/nrow(test_titanic)


#Creating the flags without running the accuracy model
predicted_bucket_second = ifelse(predicted_values_second>0.5273538,1,0)

test_titanic$predicted_values_second = predicted_values_second
test_titanic$predicted_bucket_second = predicted_bucket_second

Predicted_dataset = data.frame(test_titanic,predicted_values_second,predicted_bucket_second)

confusionMatrix(test_titanic$Survived,test_titanic$predicted_bucket_second)

misClasificError = mean(predicted_bucket_second!= test_titanic$Survived)

print(paste('Accuracy',1-misClasificError))





predicted_values_final = predict(Logit_model_second, newdata = titanic_test_set, type= 'response')
predicted_bucket_final = ifelse(predicted_values_final>0.5273538,1,0)

predicted_dastaset = data.frame(titanic_test_set,predicted_bucket_final)
predicted_dastaset

write_xlsx(Predicted_dataset,"D:\\OneDrive\\My Drive\\IPBA course\\R data sets\\Kaggle_Titanic\\Predicted_dataset.xlsx")









