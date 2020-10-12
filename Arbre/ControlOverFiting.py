def controlOverFiting(data_train, data_test, etiquette, indice = 'gini', min_depth_test = 1, max_depth_test = 30, couleur = "blue"):

    max_depth = []
    acc_indice = []
 
    #Entrainement des différents modèles
    for i in range(min_depth_test, max_depth_test):
        tree = DecisionTreeClassifier(criterion=indice, max_depth=i)
        tree.fit(data_train.drop(etiquette, axis = 1), data_train[etiquette])
        pred = tree.predict(data_test.drop(etiquette, axis = 1))
        acc_indice.append(accuracy_score(data_test[etiquette], pred))
        max_depth.append(i)
 
    #Affichage des différents modèles
    resultat = pd.DataFrame({'acc_indice':pd.Series(acc_indice), 'max_depth':pd.Series(max_depth)})
    plt.plot('max_depth','acc_indice', data=resultat, label=indice, color = couleur)
    plt.xlabel('max_depth')
    plt.ylabel('Précision')
    plt.legend()