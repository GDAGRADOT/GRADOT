from sklearn import svm
import warnings
import sys
sys.path.append('../')
warnings.filterwarnings("ignore")
from GRADOT import Gradot_Moons
from Dataset import Source_Target_generation



def Model():
    model = svm.SVC(kernel='rbf')
    return model

def GST_GRADOT_Moons(Domains,y_s,M):
    y=y_s
    model = Model()
    for i in range(5):
        #print(i)
        model.fit(Domains[i],y_s)
        #print('Model',i+1,'performance', model.score(Domains[i+1],y))
        y_s=model.predict(Domains[i+1])
    print('Accuracy of GRADOT on Moons dataset (%) = ',model.score(Domains[-1],y))





Y,b,M,y_s=Source_Target_generation(n_samples=200,noise=0.04)
Domains=Gradot_Moons(Y,b,M)
#TargetAccuracy=GST_GRADOT_Moons(Domains,y_s,M=5)