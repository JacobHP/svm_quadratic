import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress'] = False
import time
"""
V.1 - Initial JHP. Need to clean up, vectorize etc.
        TODO: Explore making this multiclass
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
        print(support_vector_indices)
        weights=lagrange_multipliers[support_vector_indices]
        support_vectors=X[support_vector_indices]
        support_vector_labels=y[support_vector_indices]
        self.support_vectors=support_vectors
        self.support_vector_labels=support_vector_labels
        self.weights=weights
        #calculate bias as mean of prediction errors for a zero bias model
        self.bias=0
        self.fitted=True
        b=np.mean([y_k-self.decision_function([x_k]).item() for (y_k, x_k) in zip(self.support_vector_labels,self.support_vectors)])
        self.bias=b
        toc=time.time()
        print('SVM fitted in', toc-tic, 'seconds.')
    
    def decision_function(self, X):
        """
        computes the raw decision function on array X
        """
        if self.fitted==False:
            raise Exception('Fit before you make a prediction.')
        else:
            results=[]
            for x in X:
                result=self.bias
                for a_i, x_i, y_i in zip(self.weights, self.support_vectors, self.support_vector_labels):
                    result+=a_i*y_i * self.kernel(x_i, x)
                results.append(result)
            return np.array(results)

        
    def predict(self, X):
        """
        Compute class predictions (if single obs call model.predict([x]).item())
        """
        if self.fitted==False:
            raise Exception('Fit before you make a prediction.')
        else:
            results=[]
            for x in X: 
                result_raw=self.bias
                for a_i, x_i, y_i in zip(self.weights, self.support_vectors, self.support_vector_labels):
                    result_raw+=a_i*y_i * self.kernel(x_i, x)
                result=np.sign(result_raw)
                results.append(result)
            return np.array(results)
    
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
        P=np.outer(y,y)*self.gram(X) 
        A=y.reshape(1,n_samples).astype(float)
        b=0.0
        G=np.concatenate((np.eye(n_samples), -np.eye(n_samples)))
        h=np.concatenate((self.C*np.ones((n_samples, 1)),np.zeros((n_samples,1)))) 
        sol=cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G),\
                              cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
        return np.ravel(sol['x']) 
    
    

class Kernel:
    """
    Kernel class - we only use positive definite kernels (as full rank) given that cvxopt requires full rank matrix
    Can try other kernels providing they're positive definite
    """
    @staticmethod
    def linear():
        def f(x,y):
            return np.inner(x,y)
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
    
