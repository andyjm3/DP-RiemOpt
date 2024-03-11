import numpy as np
import warnings

class Loss():
    def __init__(self, name="squareloss"):
        self.__name = name
        self.eval = self.__assignloss(name)

    def eval(self):
        print("Warning: eval is not defined")

    def info(self):
        print("Loss for structured learning: ", self.__name)

    # Implemented loss functions
    def squareloss(self, Y0, Y1):
        return sqdist(Y0, Y1)

    def gaussianloss(self, Y0, Y1, gamma=1):
        dist = self.squareloss(Y0, Y1)
        return (1 - np.exp(-dist * gamma))

    def hellinger(self, Y0, Y1):
        ## Only for distances between vectors
        if (Y0.ndim + Y1.ndim) <= 3:
            axis = np.max((Y0.ndim, Y1.ndim)) - 1
            return np.sum(np.square((np.sqrt(Y0) - np.sqrt(Y1))), axis=axis)
        else:
            if Y1.ndim > 1 and Y0.ndim > 1:
                newdim0 = Y0.ndim
                Y0_extradim = np.expand_dims(np.sqrt(Y0), axis=newdim0)
                Y0_rep      = np.repeat(Y0_extradim, repeats=Y1.shape[0], axis=newdim0)
            else:
                Y0_rep = Y0

            if Y0.ndim > 1:
                newdim1 = Y1.ndim
                Y1_extradim = np.expand_dims(np.sqrt(Y1), axis=newdim1).transpose((2,1,0))
                Y1_rep      = np.repeat(Y1_extradim, repeats=Y0.shape[0], axis=0)
            else:
                Y1_rep = Y1

            #print("Y0_rep shape: ", Y0_rep.shape, "Y1_rep shape: ", Y1_rep.shape)
            loss_matrix = np.sum( np.square(Y0_rep - Y1_rep), 1)

            return loss_matrix

    def squaredLorentzGeodesic(self, Y0, Y1):
        """Computes the squared geodesitg loss on the hyperbolic manifold with
        Lorentz model
        :param Y0: A set of P0 points in H_n seen as a set of row vectors in R^(n+1)
        :param Y1: A set of P1 points in H_n seen as a set of row vectors in R^(n+1)
        :return: A matrix G of size P0 x P1 where each element
        (G)_ij = arcosh(-<(Y0)_i, (Y1)_j>_L)^2
        """

        def lorentzDotThresholded(Y0, Y1):
            G = np.dot(Y0[:, 1:], Y1[:, 1:].T) - Y0[:, 0][:, np.newaxis] * Y1[:, 0][:, np.newaxis]
            G[(G > 0) & (G < 1)] = 1
            G[(G > -1) & (G <= 0)] = -1
            return G
        x = -lorentzDotThresholded(Y0, Y1)
        return np.log(x + np.sqrt(x**2 -1))**2


    def __assignloss(self, name):
        switcher = {
            "squareloss"                : self.squareloss,
            "hellinger"                 : self.hellinger,
            "guassianloss"              : self.gaussianloss,
            "squaredLorentzGeodesic"    : self.squaredLorentzGeodesic,
        }
        if name not in switcher.keys():
            print("Loss.__assignloss warning: Couldn't find specified loss")
        return switcher.get(name)


class Alpha():
    def __init__(self, kernel=None, kernelname=None):
        if kernel is None:
            print("Alpha.__init__() warning: no kernel was specified, using default Gaussian kernel")
            self.kernel = self.gausskernel
            self.__kernelname = "Gaussian kernel"
        elif isinstance(kernel, str):
            self.kernel = self.__assignkernel(kernel)
            self.__kernelname = kernel
        else:
            self.kernel = kernel
            self.__kernelname = kernelname

    def info(self):
        print("Alpha coefficients for structured learning, kernel: ", self.__kernelname)

    def kernel(self, X0, X1):
        print("Alpha.kernel Warning: the kernel is not defined")

    def eval(self, X, x, lam=0, sigma=1):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        K_x = self.kernel(X, x, sigma=sigma)
        K   = self.kernel(X, X, sigma=sigma)
        n   = X.shape[0]
        alpha = np.linalg.solve((K + n*lam*np.identity(n)), K_x)
        if np.isnan(alpha).any():
            warning.warn("Alpha(): Some alphas are NaN", UserWarning)
        return np.squeeze(alpha)

    # Implemented kernels
    def gausskernel(self, X0, X1, sigma=1):
        return np.exp(-sqdist(X0, X1) / (2*sigma**2))

    def __assignkernel(self, name):
        switcher = {
            "gausskernel"   : self.gausskernel
        }
        if name not in switcher.keys():
            print("Alpha.__assignkernel warning: Couldn't find specified kernel")
        return switcher.get(name)

    
def sqdist(X1, X2):
    assert isinstance(X1, np.ndarray), "First argument is not a numpy array"
    assert isinstance(X2, np.ndarray), "Second argument is not a numpy array"
    assert (X1.ndim == 2 and X2.ndim ==2), "Please supply all points as 2-dim arrays"

    sqx = np.sum(np.multiply(X1, X1), 1)
    rows_X1     = sqx.shape[0]
    sqy = np.sum(np.multiply(X2, X2), 1)
    rows_X2     = sqy.shape[0]

    X1_squares      = np.squeeze(np.outer(np.ones(rows_X1), sqy.T))
    X2_squares      = np.squeeze(np.outer(sqx, np.ones(rows_X2)))
    double_prod     = np.squeeze(2 * np.dot(X1,X2.T))

    return X1_squares + X2_squares - double_prod


def gausskernel(X0, X1, sigma=1):
    return np.exp(-sqdist(X0, X1) / (2*sigma**2))