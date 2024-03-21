import matplotlib.pyplot as plt
import warnings
import sys
from WassersteinGeodesic import algo2
sys.path.append('../')
warnings.filterwarnings("ignore")




def Gradot_Moons(Y,b,M):
    Domains=[]
    t=1/M-1
    for i in range (1,M):
        t=i/M
        Intermediate, a = algo2(Y,b,n=100, weight=(1-t,t), max_iter=[30, 30, 30])
        Domains.append(Intermediate.T)
    Domains.insert(0,Y[0])
    Domains.append(Y[1])
    
    couleurs = ['r', 'black', 'black', 'black', 'black', 'b']    
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    Names=['Source','Intermediate1','Intermediate2','Intermediate3','Intermediate4','Target']
    for i, d in enumerate(Domains):
        row = i // 3
        col = i % 3
        axs[row, col].scatter(Domains[i][:, 0], Domains[i][:, 1],color=couleurs[i])
        axs[row, col].set_title(Names[i])
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()
    return Domains