import numpy as np
import matplotlib.pyplot as mpl
import scipy.fft as sc

#commenti: 
#  fftfreq dice quanto erano distanti i campioni nel tempo e aggiunge questa informazione ai campioni in frequenza. 
#la dft è definita da -pi a pi ma si può prendere un altro intervallo e che fa si che la trasf risulti shiftata; 
# per rimetterla a posto possiamo usare fftshift.

#  1 + 2  #
#Definire un segnale complesso con un componente reale definita come un segnale discreto coseno con ampiezza A=2 
# e frequenza f=20 Hz con 5000 campioni (da 0 a 2s), ed una componente immaginaria definita come un segnale seno con 
#le stesse ampiezza e frequenza;


#parte reale
A=2
f=20
t=np.linspace(0,2,5000)
coswave= A*np.cos(2*np.pi*f*t)

#visualizzo il segnale coseno (parte reale)
mpl.figure(figsize=(14, 7))
mpl.subplot(4,1,1)
mpl.tight_layout(pad=3.5)
mpl.plot(t,coswave,color="red")
mpl.title("parte reale",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel("campioni",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel("coswave",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()

#definisco parte immaginaria
senwave= A*np.sin(2*np.pi*f*t)

#visualizzo il segnale coseno (parte reale)
mpl.subplot(4,1,2)
mpl.tight_layout(pad=3.5)
mpl.plot(t,senwave,color="orange")
mpl.title("parte immaginaria",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel("campioni",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel("senwave",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()



#definisco il segnale complesso come somma della parte reale e parte immaginaria
segnale= coswave+1j*senwave

#visualizzo il segnale complesso
mpl.subplot(4,1,3)
mpl.tight_layout(pad=3.5)
mpl.plot(t,segnale,color="green")
mpl.title("segnale complesso",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel("campioni",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel("segnale",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()


#  3  #

trasf=sc.fft(segnale)
N=5000  #numero di campioni 
T=2   #nella traccia andava da 0 a 2s
simplespace=T/N   #distanza tra i campioni nel tempo
freq_vect=sc.fftfreq(5000,simplespace)  #vettore delle frequenze
shifted_freq=sc.fftshift(freq_vect) #shifto le frequenze
shifted_trasf= sc.fftshift(trasf)  #shifto la trasformata

#visualizzo 

mpl.subplot(4,1,4)
mpl.plot(shifted_freq,shifted_trasf,color="purple") #nel plot metto le frequenze shiftate e la trasformata shiftata;
mpl.tight_layout(pad=3.5)
mpl.title("abs della trasformata di Fourier",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel("f",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel("segnale",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()

#  4  #















mpl.tight_layout()
mpl.show()


