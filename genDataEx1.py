n_samples = 100
numberOfLabelledData = 1


X1, y1 = datasets.make_gaussian_quantiles(cov=2.0, n_samples=n_samples, n_features=2, n_classes=1)
X1[:,0] = 9. + X1[:,0]
X1[:,1] = 8. + X1[:,1]/3
X2, y2 = datasets.make_gaussian_quantiles(cov=1.5, n_samples=n_samples, n_features=2, n_classes=1)
X2[:,0] = 8. + X2[:,0]
X2[:,1] = 5. + X2[:,1]/2
Xa = np.concatenate((X1, X2))
ya = np.concatenate((y1, - y2 + 1))
ya = 2*ya-1
Xa, ya = shuffle(Xa, ya)

Xblue = Xa[ya==-1]
Xred = Xa[ya==1]

#We choose a certain number of data that will be labelled
indBlue = np.random.choice(list(range(n_samples)), numberOfLabelledData)
indRed = np.random.choice(list(range(n_samples)), numberOfLabelledData)

Xbluelab = Xblue[indBlue]
Xredlab = Xred[indRed]

ybluelab = [-1 for i in range(numberOfLabelledData)]
yredlab = [1 for i in range(numberOfLabelledData)]

Xblueunlab = np.delete(Xblue, indBlue, axis=0)
Xredunlab = np.delete(Xred, indRed, axis=0)

Xu = np.concatenate((Xblueunlab, Xredunlab))
yu = np.concatenate(([-1 for i in range(Xblueunlab.shape[0])], [1 for i in range(Xredunlab.shape[0])]))

Xu,yu = shuffle(Xu, yu)

Xl = np.concatenate((Xbluelab, Xredlab))
yl = np.concatenate((ybluelab, yredlab))

Xl, yl = shuffle(Xl, yl)



# Display
fig=plt.figure(figsize=(6,6), dpi= 80, facecolor='w', edgecolor='k')
plt.scatter(Xblue[:,0],Xblue[:,1],c='black',s=20, alpha=0.5, label='Unlabelled data')#, edgecolors='k')
plt.scatter(Xred[:,0],Xred[:,1],c='black',s=20, alpha = 0.5);
plt.scatter(Xblue[indBlue,0],Xblue[indBlue,1],c='b',s=20, label='Class 1')#, edgecolors='k')
plt.scatter(Xred[indRed,0],Xred[indRed,1],c='r',s=20, label='Class 2');
plt.legend()
plt.title('The two classes can be distinguished but we have a small amount of labelled data')