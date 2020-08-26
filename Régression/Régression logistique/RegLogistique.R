##Librairies générales
library(tidyr)

##Librairies régression logistique binomiale et multinomiale
library(glmnet)
library(MLmetrics)

##Librairies régression logistique ordinale
library(MASS)


##La fonction logistique

###Importation des données

data = read.csv('audit_risk.csv', sep = ',')
data <- drop_na(data)#Suppression des valeurs manquantes
data <- data[, -c(2)]#Suppression de variable contenant des valeurs qualitatives

###Découpe du dataset

set.seed(0)#Définition de la répétabilité de l'aléatoire
dt = sort(sample(nrow(data), nrow(data)*.8))#Définition de la quantité de données à répartir
data_train <- data[dt,]#Données d'entrainement
data_test <- data[-dt,]#Données de test

X_train <- data_train[, -c(26)]#Variables explicatives entrainement
y_train <- data_train$Risk#Variable cible entrainement
X_test <- data_test[, -c(26)]#Variables explicatives test
y_test <- data_test$Risk#Variable cible test

####Modèle de prédiction : régression logistique

model <- glm(Risk~., family = 'binomial', data = data_train)#Entrainement de l'estimateur, en précisant
#family = binomial pour la régression logistique binaire
y_prob <- predict(model, newdata = X_test, type = 'response')#Calcul des probabilités d'appartenance à la classe positive, sur les données de test
y_pred <- ifelse(y_prob > 0.5, 1, 0)#Prédiction sur les données X de test

##Evaluation de modèle
LogLoss(y_pred, y_test)#Perte logistique
table(y_test, y_pred)#Construction de la matrice de confusion

####Régularisation

X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)

model <- glmnet(X_train, y_train, family = 'binomial', alpha = 0, lambda = 0.05)#Entrainement du classificateur avec 
#une régularisation L2 avec alpha = 0, un coefficient de régularisation lambda et family qui définit la régression logistique
y_prob <- predict(model, newx = X_test, type = 'response')#Prédiction des probabilités sur les données de test
y_pred <- ifelse(y_prob > 0.5, 1,0)#Prédiction de l'état de la fraude


##Evaluation de modèle
LogLoss(y_pred, y_test)#Perte logistique
table(y_test, y_pred)#Construction de la matrice de confusion

####Modèle de prédiction : régression softmax

##Importation des données

data = read.csv('wine_data.csv', sep = ',')

##Découpe du dataset

set.seed(0)
dt = sort(sample(nrow(data), nrow(data)*.8))
data_train<-data[dt,]
data_test<-data[-dt,]

X_train <- data_train[, -c(1)]
y_train <- data_train$Wine
X_test <- data_test[, -c(1)]
y_test <- data_test$Wine


##Modèle de prédiction

X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)

model <- glmnet(X_train, y_train, family = 'multinomial', type.logistic = "Newton" ,alpha = 0, lambda = 0.6000000000000001)
y_pred <- predict(model, newx = X_test, type = 'class')#Prédiction des classes 
y_prob <- predict(model, newx = X_test, type = 'response')#Prédiction des probabilités d'appartenance

##Evaluation modèle

table(y_pred, y_test)#Matrice de confusion
y_pred <- as.numeric(y_pred)
LogLoss(y_pred,y_test)#Perte logistique


####Modèle de prédiction : regression logistique ordinale

model <- polr(as.factor(Wine)~ Alcohol + Malic.acid + Ash + Acl + Mg, data = data_test, method = 'logistic')#Entrainement du modèle
y_pred <- predict(model, X_test)#Prédiction sur les données X
y_prob <- predict(model, X_test, type='p')#Prédiction des probabilités d'appartenance
table(y_test,y_pred)#Matrice de confusion

##Evaluation modèle

table(y_pred, y_test)#Matrice de confusion
y_pred <- as.numeric(y_pred)
LogLoss(y_pred,y_test)#Perte logistique