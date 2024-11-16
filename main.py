import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.io import loadmat
from scipy.fft import fft, fftshift, fftfreq, ifft


def separa_intervalli(segnale,num_campioni):
    import math
    num_finestre=math.ceil(len(segnale)/num_campioni)
    energia_media=[]
    for i in range(0,len(segnale),num_campioni):
        fine_finestra = min(i + num_campioni, len(segnale))  # Non superare la lunghezza del segnale
        finestra=segnale[i:fine_finestra]
        energia_media.append(sum(j**2 for j in finestra)/num_campioni)
    return energia_media

#Creo la maschera delle frequenze
def ideal_filter(x_f, freqs, f_low=None, f_high=None):
    H=np.ones_like(x_f)
    if f_low is not None:
        H[np.abs(freqs)<f_low]=0
    if f_high is not None:
        H[np.abs(freqs)>f_high]=0
    return H

# Esercizio 1
data = loadmat('eeg_CP4_MI_LH_s09.mat')
segnali = data['eeg_CP4_MI_LH_s09'].flatten()
n1 = 400
n2 = 2800
t1 = 0.002
segnali_split = segnali[n1:n2]

#t=np.linspace(n1,n2,n2-n1)*t1
t = np.arange(0, 4.8, t1)
figure(figsize=(10,10))
plt.subplot(1, 1, 1)

# Primo grafico
plt.plot(t, segnali_split, label='Segnale $x_n$', color='limegreen')
plt.title("Esercizio 1 \n Segnale EEG per Motor Imagery (Left Hand) al sensore CP4")
plt.xlabel("Tempo (s)")
plt.ylabel("Ampiezza")
plt.legend(loc='upper left')
plt.grid()

box = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
energia = sum(abs(segnali_split)**2)

plt.text(-0.17, -10.67, f'Energia (J): {energia:.2f}', fontsize=10, bbox=box, ha='left', va='center')
plt.tight_layout()
#plt.savefig("grafico_1.png", dpi=1200, bbox_inches='tight', format='png')

##################################################################################

# Esercizio 2
figure(figsize=(10,10),label='Esercizio 2')
plt.subplot(1, 1, 1)
data1 = loadmat('eeg_C4_MI_LH_s09.mat')
segnali1 = data1['eeg_C4_MI_LH_s09'].flatten()

n1 = 400
n2 = 2800
t1 = 0.002
segnali1_split = segnali1[n1:n2]
media = np.mean(segnali1_split)

t = np.arange(0, 4.8, t1)
plt.plot(t, segnali1_split - media, label='Segnale $y_n$', color='#F17853')
plt.title("Esercizio 2 \n Segnale EEG per Motor Imagery (Left Hand) al sensore C4")
plt.xlabel("Tempo (s)")
plt.ylabel("Ampiezza")
plt.legend(loc='upper left')
plt.grid()

corrcoef = np.corrcoef(segnali_split,segnali1_split-media)[0,1]
plt.text(-0.17, 1.10, f'Coefficiente di correlazione: {corrcoef:.2f}', fontsize=10, bbox=box, ha='left', va='top')
#plt.savefig("grafico_2.png", dpi=1200, bbox_inches='tight', format='png')

###########################################################################################################
#Esercizio 3
#dati trasformata
N=n2-n1 #Numero di campioni
#calcolo della trasformata
x_f=fft(segnali_split)
#calcolo delle frequenze
frequenze = fftfreq(N,t1)
#applicazione del valore assoluto della trasformata
x_f_traslata=fftshift(x_f)
frequenze_traslata=fftshift(frequenze)
figure(figsize=(10,10),label='Esercizio 3')
plt.plot(frequenze_traslata,abs(x_f_traslata),label='Trasformata di $x_n$',color='green')
plt.title("Modulo della Trasformata di Fourier del segnale $x_n$")
plt.xlabel("Frequenza (Hz)")
plt.ylabel("Ampiezza")
plt.xlim()
plt.grid()
#plt.savefig("grafico_3.png", dpi=1200, bbox_inches='tight', format='png')

#Creazione del filtro passa-banda [30,40]hz
min_f=30
max_f=40

maschera=ideal_filter(x_f,frequenze_traslata,min_f,max_f)
#filtro il segnale trasformato
x_f_filtrato= x_f*maschera

#recupero il segnale trasformato e filtrato con l'antitrasformata
z_n=ifft(x_f_filtrato)
figure(figsize=(10,10),label='Esercizio 3 pt 2')
#creo il grafico
plt.plot(frequenze_traslata, np.abs(fftshift(x_f_filtrato)),color='goldenrod')
plt.title("Risposta in frequenza del filtro passa banda [30, 40] Hz")
plt.xlabel("Frequenza (Hz)")
plt.ylabel("Ampiezza")
plt.grid()
#plt.savefig("grafico_4.png", dpi=1200, bbox_inches='tight', format='png')

# Visualizzazione del segnale filtrato nel dominio del tempo
figure(figsize=(10,10),label='Esercizio 3 pt 3')

plt.plot(t, np.real(z_n), label='$z_n$', color='lightskyblue')
plt.title("Segnale filtrato nel dominio del tempo $z_n$")
plt.xlabel("Tempo (s)")
plt.ylabel("Ampiezza")
plt.legend()
plt.grid()
#plt.savefig("grafico_5.png", dpi=1200, bbox_inches='tight', format='png')

############################################################
#Domanda Extra

Nc=500
energia3=separa_intervalli(segnali,Nc)
tbonus= np.linspace(0,len(energia3),len(energia3))

figure(figsize=(10,10),label='Esercizio bonus')
plt.subplot(2,1,1)
plt.plot(tbonus,energia3, label='energia media', color='crimson')
plt.title("Energia media per 500 campioni di CP4 in attivo")
plt.xlabel("Finestre")
plt.ylabel("Energia(J)")
plt.legend()
plt.grid()

data2=loadmat('eeg_CP4_rest_s09.mat')
segnali2=data2['eeg_CP4_rest_s09'].flatten()
plt.subplot(2,1,2)
energia4=separa_intervalli(segnali2,Nc)
tbonus= np.linspace(0,len(energia4),len(energia4))

plt.plot(tbonus,energia4, label='energia media', color='navy')
plt.title("Energia media per 500 campioni di CP4 a riposo")
plt.xlabel('Finestre')
plt.ylabel("Energia (J)")
plt.legend()
plt.subplots_adjust(hspace=0.4)
plt.grid()
#plt.savefig("grafico_6.png", dpi=1200, bbox_inches='tight', format='png')

#UNIONE GRAFICI
tbonus= np.linspace(0,len(energia4),len(energia4))

figure(figsize=(10,10),label='Unione bonus')
plt.plot(tbonus,energia3[:len(energia4)], label='energia media in attivo', color='crimson')



plt.plot(tbonus,energia4, label='energia media a riposo', color='navy')
plt.title("Energia media per 500 campioni di CP4")
plt.xlabel('Finestre')
plt.ylabel("Energia (J)")
plt.legend()


plt.grid()
#plt.savefig("grafico_7.png", dpi=2400, bbox_inches='tight', format='png')


plt.show()