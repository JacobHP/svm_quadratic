import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress'] = False
import time
import pandas as pd
"""
V.0.1 - Initial JHP. Need to clean up, vectorize prediction + vectorize other bits etc.
        May see if I can make this multiclass if time. 
"""

class SVC:
    def __init__(self, C, kernel, threshold=1e-5, soft_margin=True):
        self.C=C
        self.kernel=kernel
        self.threshold=threshold
        self.soft_margin=soft_margin #will add in non-soft margin implementation
        self.support_vectors=None
        self.support_vector_labels=None
        self.weights=None
        self.bias=None
        self.fitted=False
        
    def fit(self, X, y):
        """
        fit SVM to training data by creating support vectors and bias
        """
        tic=time.time()
        lagrange_multipliers=self.lagrange_multipliers(X,y)
        support_vector_indices=lagrange_multipliers > self.threshold
        weights=lagrange_multipliers[support_vector_indices]
        support_vectors=X[support_vector_indices]
        support_vector_labels=y[support_vector_indices]
        self.support_vectors=support_vectors
        self.support_vector_labels=support_vector_labels
        self.weights=weights
        #calculate bias as mean of prediction errors for a zero bias model
        self.bias=0
        b=np.mean([y_k-self.predict(x_k) for (y_k, x_k) in zip(y,X)])
        self.bias=b
        self.fitted=True
        toc=time.time()
        print('SVM fitted in', toc-tic, 'seconds.')
                  
    def predict(self, x):
        """
        Use a fitted svm to predict unseen test data. Currently predicts one value
        """
        if self.fitted==False:
            raise Exception('Fit before you make a prediction.')
        else:
            result=self.bias
            for a_i, x_i, y_i in zip(self.weights, self.support_vectors, self.support_vector_labels):
                result+=a_i*y_i * self.kernel(x_i, x)
            return np.sign(result)
    
    def gram(self, X):
        """
        Construct the Gram matrix for dual problem - try vectorising this
        """
        n_sample=X.shape[0]
        gram=np.zeros((n_sample, n_sample))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                gram[i,j]= self.kernel(x_i, x_j)
        return gram
    
    
    def lagrange_multipliers(self, X,y):
        n_samples, n_features=X.shape
        q=-np.ones((n_samples, 1))
        P=y*y.T*self.gram(X)
        A=y.reshape(1,n_samples).astype(float)
        b=0.0
        G=np.concatenate((np.eye(n_samples), -np.eye(n_samples)))
        h=np.concatenate((self.C*np.ones((n_samples, 1)),np.zeros((n_samples,1))))
        #call qp solver in cvxopt (must convert to cvxopt matrices)
        sol=cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G),\
                              cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
        return np.ravel(sol['x'])
        
      

class Kernel:
    """
    Kernel class - we only use positive definite kernels (as full rank) given that cvxopt requires full rank matrix
    https://math.stackexchange.com/questions/130554/gaussian-kernels-why-are-they-full-rank
    https://en.wikipedia.org/wiki/Positive-definite_kernel
    """
    @staticmethod
    def linear():
        def f(x,y):
            return np.dot(x,y)
        return f
    
    @staticmethod
    def polynomial(dimension):
        def f(x,y):
            return (np.dot(x,y))**dimension
        return f
    
    @staticmethod
    def rbf(sigma):
        def f(x,y):
            return np.exp(-(np.linalg.norm(x-y)**2)/(2*sigma**2))
        return f


#test
df=pd.read_csv('data_banknote_authentication.txt', delimiter=',', index_col=False)
df.columns=['f_1','f_2','f_3','f_4','targ_temp']
df['target']=df['targ_temp'].apply(lambda x: x-1 if x == 0 else x)
X=df[['f_1', 'f_2','f_3','f_4']].to_numpy()
y=df['target'].to_numpy()


model_linear=SVC(C=0.5,kernel=Kernel.linear(), threshold=1e-5, soft_margin=True)
model_polynomial=SVC(C=0.5, kernel=Kernel.polynomial(2), threshold=1e-5, soft_margin=True)
model_rbf=SVC(C=0.5, kernel=Kernel.rbf(0.5), threshold=1e-5, soft_margin=True)

for model in [model_linear, model_polynomial, model_rbf]:
    model.fit(X,y)
    
