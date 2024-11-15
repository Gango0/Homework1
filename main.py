import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data= loadmat('eeg_CP4_MI_LH_s09.mat')
segnali=data['eeg_CP4_MI_LH_s09'].flatten()
n1=400
n2=2800
t1=0.002
segnali2= segnali[n1:n2] #limito i valori al range [400,2800]


t=np.linspace(n1,n2,n2-n1)*t1


plt.figure(figsize=(10, 6))

plt.plot(t,segnali2,label='Segnale $x_n$',color='#84B89F')
plt.title("Esercizio 1 \n Segnale EEG per Motor Imagery (Left Hand) al sensore CP4")
plt.xlabel("Tempo (ms)")
plt.ylabel("Ampiezza")

plt.legend(loc='upper left')

box = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
energia=0

for i in segnali2:
    energia+=i**2

plt.gcf().text( 0.1,0.8,f'Energy: {energia:.2f}', fontsize=10, bbox=box)


plt.tight_layout()
plt.grid()
plt.show()