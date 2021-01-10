# Ce code permet de tracer le score moyen de SVM et S3VM en fonction de la proportion de données labellisées. 
# Il prend environ trois petites minutes à tourner
nb_trials = 20
percentLabelleds = np.linspace(2,90,20)/100

svmScorePer = []
s3vmScorePer = []

for per in percentLabelleds:
    svmScore = []
    s3vmScore = []
    print("Percent labelled: "+str(per)+", ", end='')
    for i in range(nb_trials):
        X_l, y_l, X_u, y_u = getTrainingAndWorkingSet(Xtrain, ytrain, per)

        mySVC = svm.SVC(kernel='linear')
        mySVC.fit(X_l, y_l)

        prediction = mySVC.predict(Xtest)
        svmScore.append(mySVC.score(Xtest,ytest))

        model = QN_S3VM(list(X_l), list(y_l), list(X_u), my_random_generator, lam=1, lamU=1, kernel_type='Linear')
        model.train()
        predictions = model.getPredictions(Xtest)
        s3vmScore.append(1-np.sum(np.not_equal(predictions,ytest))/len(ytest))
        print('*', end='')

    print(" done!")
    svmScorePer.append(np.mean(svmScore))
    s3vmScorePer.append(np.mean(s3vmScore))


plt.figure()
plt.plot(percentLabelleds, svmScorePer, color='b', label='SVM')
plt.plot(percentLabelleds, s3vmScorePer, color='r', label='S3VM')
plt.xlabel('Percent data labelled')
plt.ylabel('Score on the same dataset')
plt.legend()