###Librairies

library(rsvd)
library(factoMineR)
library(explor)
library(psych)
library(imager)

###Imporation des données

data <- read.csv('audit_risk.csv', sep=',')
data <- drop_na(data)#Suppression des valeurs manquantes
data <- data[, -c(2)]#Suppression de variable contenant des valeurs qualitatives

###Découpe des données

X <- subset(data, select = -c(Risk))
X <- as.matrix(X)
y <- data$Risk

###Standardisation des données

X_Centered <- scale(X, center = T)#Centrage des données
X_Centered <- as.data.frame(X_Centered)#Consversion de la matrice en dataframe
X_Centered <- subset(X_Centered, select = -c(24))#Suppression de la colonne de valeurs manquantes générées

###Recherche des composantes principales

SVDecomp <- svd(X_Centered)#Décomposition en valeurs singulières
c1 <- t(SVDecomp$v[,0])#Première composante principale
c2 <- t(SVDecomp$v[,1])#Deuxième composante principale

###Projection dimensionnelle

X_d <- t(SVDecomp$v[1:9,])#On récupère les d = 9 premières composantes principales
X_Centered <- as.matrix(X_Centered)#Conversion en matrice
X_proj <- X_Centered %*% X_d#Projection des données d'entrainement sur l'espace à d dimension

###Construction du modèle de réduction

model <- PCA(data, ncp = 25, scale = TRUE)#Création du modèle avec centrage des données et définition du nombre de composantes à prendre en compte

barplot(model$eig[,3])#Variance expliquée en fonction du nombre de variables

#Vous pouvez aussi utiliser la commande suivante, offrant plus de possibilités d'observations.

explor(model)#Ouverture d'une fenêtre permettant de visualiser les variables importantes


####Traitement d'images

###Décompression des données

X_decompressed <- X_proj %*% t(X_d)#Produit matriciel d'inversion de la méthode ACP

###Importation de l'image

img <- load.image("raton_laveur.png")
plot(img)

###Détermination du nombre de composantes

model <- PCA(img, ncp = 768, scale = T)#Création du modèle avec centrage des données
barplot(model$eig[,3]>=95)#Variance expliquée en fonction du nombre de variables

model <- PCA(img, ncp = 150, scale = T)#Création du modèle avec centrage des données

###Affichage graphique

decomposition_svd <- svd(img)#Décomposition en valeur singulière
composantes_pr <- 150#Définition du nombre de composantes principales à conserver
img.svd <- decomposition_svd$u[,1:composantes_pr] %*% diag(decomposition_svd$d[1:composantes_pr]) %*% t(decomposition_svd$v[,1:composantes_pr])#Produit matricielle

####Critère de Kaiser

summary(model, ncp = 25)#Affichage des valeurs propres des variables du dataset

##Autre méthode 

cormat <- cor(data)
cormat <- cormat[, -c(24)]#Suppression de variable contenant des valeurs qualitatives
scree(cormat)