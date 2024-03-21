import numpy as np
import warnings
import sys
sys.path.append('../')
warnings.filterwarnings("ignore")
from sklearn.datasets import make_moons



def rotationMat(rotTheta):
    return np.array([[np.cos(rotTheta), -np.sin(rotTheta)], 
                     [np.sin(rotTheta), np.cos(rotTheta)]])


def Source_Target_generation(n_samples,noise):
    source, y_s = make_moons(n_samples=n_samples, noise=noise,random_state=42)
    R=rotationMat(np.deg2rad((70)))
    target=np.dot(source,R)
    Y=[source,target]
    b=[np.ones(source.shape[0])/source.shape[0],np.ones(target.shape[0])/target.shape[0]]
    M=5
    return Y,b,M,y_s