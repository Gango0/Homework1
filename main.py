import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.io import loadmat
from scipy.fft import fft, fftshift, fftfreq, ifft


def energia_media_finestra(segnale,num_campioni):
    energia_media=[]
    for i in range(0,len(segnale),num_campioni):
        fine_finestra = min(i + num_campioni, len(segnale))
        finestra=segnale[i:fine_finestra]
        energia_media.append(sum(j**2 for j in finestra)/num_campioni)
    return energia_media

#Creo la maschera delle frequenze
def crea_filtro_ideale(f_x, freqs, f_low=None, f_high=None):
    h=np.ones_like(f_x)
    if f_low is not None:
        h[np.abs(freqs)<f_low]=0
    if f_high is not None:
        h[np.abs(freqs)>f_high]=0
    return h

# Esercizio 1
data = loadmat('eeg_CP4_MI_LH_s09.mat')
segnali = data['eeg_CP4_MI_LH_s09'].flatten()
n1 = 400
n2 = 2800
t1 = 0.002
segnali_split = segnali[n1:n2]

t = np.linspace(n1, n2,n2-n1)*t1

figure(figsize=(10,10),facecolor="lightcyan")
plt.subplot(1, 1, 1)

# Primo grafico
plt.plot(t, segnali_split, label='Segnale $x_n$', color='limegreen',lw=0.5)
plt.title("Segnale EEG per Motor Imagery (Left Hand) al sensore CP4", fontsize=20, fontweight='bold')
plt.xlabel("Tempo (s)",fontsize=15)
plt.ylabel("Ampiezza",fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.grid()

box = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
energia = sum(abs(segnali_split)**2)

plt.text(0.66, -10.67, f'Energia: {energia:.2f} J', fontsize=15,fontweight='bold', bbox=box, ha='left', va='center')
plt.tight_layout()
# plt.savefig("grafico_1.png", dpi=1200, bbox_inches='tight', format='png')

##################################################################################

# Esercizio 2
figure(figsize=(10,10),label='Esercizio 2',facecolor="lightcyan")
plt.subplot(1, 1, 1)
data1 = loadmat('eeg_C4_MI_LH_s09.mat')
segnali1 = data1['eeg_C4_MI_LH_s09'].flatten()

segnali1_split = segnali1[n1:n2]
media = np.mean(segnali1_split)

plt.plot(t, segnali1_split - media, label='Segnale $y_n$', color='#F17853',lw=0.5)
plt.title("Segnale EEG per Motor Imagery (Left Hand) al sensore C4", fontsize=20, fontweight='bold')
plt.xlabel("Tempo (s)",fontsize=15)
plt.ylabel("Ampiezza",fontsize=15)
plt.legend(loc='upper left',fontsize=15)
plt.grid()

corrcoef = np.corrcoef(segnali_split,segnali1_split-media)[0,1]
plt.text(0.67, 1.10, f'Coefficiente di correlazione: {corrcoef:.2f}', fontsize=15, bbox=box, ha='left', va='top')
# plt.savefig("grafico_2.png", dpi=1200, bbox_inches='tight', format='png')

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
figure(figsize=(10,10),label='Esercizio 3',facecolor="lightcyan")
plt.plot(frequenze_traslata,abs(x_f_traslata),label='Trasformata di $x_n$',color='deeppink',lw=1.5)
plt.title("Modulo della Trasformata di Fourier del segnale $x_n$", fontsize=20, fontweight='bold')
plt.xlabel("Frequenza (Hz)",fontsize=15)
plt.ylabel("Ampiezza",fontsize=15)
# plt.xlim(-4,4)
# plt.ylim(0,30000)
plt.grid()
# plt.savefig("grafico_3.png", dpi=1200, bbox_inches='tight', format='png')
# plt.savefig("grafico_3_zoom.png", dpi=1200, bbox_inches='tight', format='png')

#Creazione del filtro passa-banda [30,40]hz
min_f=30
max_f=40

maschera=crea_filtro_ideale(x_f,frequenze_traslata,min_f,max_f)
#filtro il segnale trasformato
x_f_filtrato= x_f*maschera

#recupero il segnale trasformato e filtrato con l'antitrasformata
z_n=ifft(x_f_filtrato)
figure(figsize=(10,10),label='Esercizio 3 pt 2',facecolor="lightcyan")
#creo il grafico
plt.plot(frequenze_traslata, np.abs(fftshift(x_f_filtrato)),color='goldenrod',lw=0.5)
plt.title("Risposta in frequenza del filtro passa banda [30, 40] Hz", fontsize=20, fontweight='bold')
plt.xlabel("Frequenza (Hz)",fontsize=15)
plt.ylabel("Ampiezza",fontsize=15)
plt.grid()
# plt.savefig("grafico_4.png", dpi=1200, bbox_inches='tight', format='png')

# Visualizzazione del segnale filtrato nel dominio del tempo
figure(figsize=(10,10),label='Esercizio 3 pt 3',facecolor="lightcyan")

plt.plot(t, np.real(z_n), label='$z_n$', color='lightskyblue',lw=0.5)
plt.title("Segnale filtrato nel dominio del tempo $z_n$", fontsize=20, fontweight='bold')
plt.xlabel("Tempo (s)",fontsize=15)
plt.ylabel("Ampiezza",fontsize=15)
plt.legend(fontsize=15)
plt.grid()
# plt.savefig("grafico_5.png", dpi=1200, bbox_inches='tight', format='png')

############################################################
#Domanda Extra

Nc=500
energia3=energia_media_finestra(segnali,Nc)
tbonus= np.linspace(0,len(energia3),len(energia3))

figure(figsize=(10,10),label='Esercizio bonus',facecolor="lightcyan")
plt.subplot(2,1,1)
plt.plot(tbonus,energia3, label='energia media', color='crimson',lw=0.5)
plt.title("Energia media per 500 campioni di CP4 in attivo", fontsize=20, fontweight='bold')
plt.xlabel("Finestra",fontsize=15)
plt.ylabel("Energia(J)",fontsize=15)
plt.legend(fontsize=15)
plt.grid()

data2=loadmat('eeg_CP4_rest_s09.mat')
segnali2=data2['eeg_CP4_rest_s09'].flatten()
plt.subplot(2,1,2)
energia4=energia_media_finestra(segnali2,Nc)
tbonus= np.linspace(0,len(energia4),len(energia4))

plt.plot(tbonus,energia4, label='energia media', color='navy',lw=0.5)
plt.title("Energia media per 500 campioni di CP4 a riposo", fontsize=20, fontweight='bold')
plt.xlabel('Finestra',fontsize=15)
plt.ylabel("Energia (J)",fontsize=15)
plt.legend(fontsize=15)
plt.subplots_adjust(hspace=0.4)
plt.grid()
# plt.savefig("grafico_6.png", dpi=1200, bbox_inches='tight', format='png')

#UNIONE GRAFICI
tbonus= np.linspace(0,len(energia4)-1,len(energia4)-1)

figure(figsize=(10,10),label='Unione bonus',facecolor="lightcyan")
plt.plot(tbonus,energia3[:len(energia4)-1], label='energia media in attivo', color='crimson',lw=0.5)



plt.plot(tbonus,energia4[:len(energia4)-1], label='energia media a riposo', color='navy',lw=0.5)
plt.title("Energia media per 500 campioni di CP4", fontsize=20, fontweight='bold')
plt.xlabel('Finestra',fontsize=15)
plt.ylabel("Energia (J)",fontsize=15)
plt.legend(fontsize=15)


plt.grid()
#plt.savefig("grafico_7.png", dpi=2400, bbox_inches='tight', format='png')


plt.show()
