import matplotlib.pyplot as plt
import svm
import numpy as np
from sklearn import datasets
"""
Plot using the iris dataset
"""

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y=np.array(list(map(lambda x: -1 if x==0 else 1, iris.target)))

model_linear=svm.SVC(C=0.5,kernel=svm.Kernel.linear())
model_poly=svm.SVC(C=0.5, kernel=svm.Kernel.polynomial(3))
model_rbf=svm.SVC(C=0.5, kernel=svm.Kernel.rbf(2))
for model in [model_linear, model_poly, model_rbf]:
    model.fit(X,y)

def plot_svm(model, X, y, figname, title, axes=[0,10,0,10]):
    x0_lin=np.linspace(axes[0], axes[1], 100)
    x1_lin=np.linspace(axes[2], axes[3], 100)
    x0,x1=np.meshgrid(x0_lin, x1_lin) 
    X_mesh=np.c_[x0.ravel(), x1.ravel()] #convert mesh points into 2d for pred
    y_pred=model.predict(X_mesh).reshape(x0.shape) #predict then convert back to meshgrid for contour plot
    y_decision=model.decision_function(X_mesh).reshape(x0.shape) #
    
    plt.figure(figsize=(10,10))
    plt.plot(X[:, 0][y==-1], X[:,1][y==-1], 'bo', label='Class: -1')
    plt.plot(X[:,0][y==1], X[:,1][y==1], 'go', label='Class: 1')
    
    #plot support vectors in red
    plt.scatter(model.support_vectors[:,0], model.support_vectors[:,1], s=100, c='r', label='Support Vectors') 
    
    plt.contourf(x0, x1, y_pred, colors=['b','g'], alpha=0.2)
    plt.contour(x0, x1, y_decision, colors='k', levels=[-1,0,1], alpha=1, linestyles=['--','-','--']) 
    plt.legend(loc='lower right')
    plt.title(title)
    plt.savefig(figname)
    
    
plot_svm(model_linear, X, y, figname='linear_iris.png', title='Linear')
plot_svm(model_poly, X, y, figname='poly_iris.png', title='Polynomial (n=3)')
plot_svm(model_rbf, X, y, figname='rbf_iris.png', title='RBF (sigma=2)')

