import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft, fftshift, fftfreq

# Esercizio 1
data = loadmat('eeg_CP4_MI_LH_s09.mat')
segnali = data['eeg_CP4_MI_LH_s09'].flatten()
n1 = 400
n2 = 2800
t1 = 0.002
segnali_split = segnali[n1:n2]

#t=np.linspace(n1,n2,n2-n1)*t1
t = np.arange(0, 4.8, t1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))  # Usa subplot con variabili ax1 e ax2

# Primo grafico
ax1.plot(t, segnali_split, label='Segnale $x_n$', color='#84B89F')
ax1.set_title("Esercizio 1 \n Segnale EEG per Motor Imagery (Left Hand) al sensore CP4")
ax1.set_xlabel("Tempo (s)")
ax1.set_ylabel("Ampiezza")
ax1.legend(loc='upper left')
ax1.grid()

box = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
energia = sum(abs(segnali_split)**2)
ax1.text(0.01, 0.85, f'Energia (J): {energia:.2f}', transform=ax1.transAxes, fontsize=10, bbox=box, ha='left', va='top')

##################################################################################

# Esercizio 2
data1 = loadmat('eeg_C4_MI_LH_s09.mat')
segnali1 = data1['eeg_C4_MI_LH_s09'].flatten()

n1 = 400
n2 = 2800
t1 = 0.002
segnali1_split = segnali1[n1:n2]
media = np.mean(segnali1_split)

t = np.arange(0, 4.8, t1)
ax2.plot(t, segnali1_split - media, label='Segnale $y_n$', color='#84B89F')
ax2.set_title("Esercizio 2 \n Segnale EEG per Motor Imagery (Left Hand) al sensore C4")
ax2.set_xlabel("Tempo (s)")
ax2.set_ylabel("Ampiezza")
ax2.legend(loc='upper left')
ax2.grid()

energia2 = sum(abs(segnali1_split - media)**2)
ax2.text(0.01, 0.85, f'Energia (J): {energia2:.2f}', transform=ax2.transAxes, fontsize=10, bbox=box, ha='left', va='top')
corrcoef = np.corrcoef(segnali_split,segnali1_split)[0,1]
print(corrcoef)
ax2.text(0.01, 0.75, f'Coefficiente di correlazione: {corrcoef:.2f}', transform=ax2.transAxes, fontsize=10, bbox=box, ha='left', va='top')

# Regola spaziatura tra i subplot
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
###########################################################################################################
#Esercizio 3

ft=1/t1

plt.show()
