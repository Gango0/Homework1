import numpy as np
import matplotlib.pyplot as mpl

#   1   #

# definire due segnali tra -5 e 5 (con 1000 campioni) che rappresentano una rect e un tri (rispettivamente centrati in 0 e con supporto [-1/2,1/2] e [-1,1]).
# Calcolare la convoluzione:
#   -tra i due segnali;
#   -tra la rect con se stessa;
# Fare un subplot con i tre segnali ottenuti

#Definisco le funzioni rect e tri come da consegna
def rect(x):
    return np.where(abs(x)<0.5,1,0)

def tri(x):
    return np.maximum(1 - np.abs(x), 0)

#definisco la variabile tempo e creo i segnali s_rect e s_tri
t=np.linspace(-5,5,1000)
s_rect=rect(t)
s_tri=tri(t)

#creo il plot della rect
mpl.figure(figsize=(10, 6))
mpl.subplot(2,2,1)
mpl.tight_layout(pad=3.5)
mpl.plot(t,s_rect,color="orange")
mpl.title("RECT",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel("t",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel("s_rect",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()

#creo il plot della tri
mpl.subplot(2,2,2)
mpl.plot(t,s_tri,color="violet")
mpl.title("TRI",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel("t",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel("s_tri",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()

#eseguo la convoluzione tra s_rect e s_tri
f_conv_1= np.convolve(s_rect,s_tri,mode="same")

#creo il plot di f_conv_1
mpl.subplot(2,2,3)
mpl.plot(f_conv_1,color="blue")
mpl.title("CONVOLUZIONE TRA RECT E TRI",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel("t",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel("f_conv_1",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()

#eseguo la convoluzione tra s_rect e s_rect
f_conv_2= np.convolve(s_rect,s_rect,mode="same")

#creo il plot di f_conv_2
mpl.subplot(2,2,4)
mpl.plot(t, f_conv_2,color="green")

mpl.title("CONVOLUZIONE TRA RECT E RECT",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel("t",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel("f_conv_2",fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()


#  2  #

# Utilizzando il segnali y_n ottenuto come uscita della convoluzione calcolare valore medio ed energia media per campione e fare il grafico 
# con segnale, valore medio(asse orizzontale) e box con valore dell'energia

#calcolo il valore medio di f_conv_1
vm1=np.mean(f_conv_1)

#calcolo il valore dell'energia di f_conv_1
en1=np.mean(np.square(f_conv_1))

#creo l'asse orizzontale che indica il valore medio di f_conv_1
mpl.axhline(y=vm1, color='r', linestyle='--', label='valore medio')
mpl.legend()

#creo il box contenente il valore dell'energia di f_conv_1
mpl.text(3.2, 97, 'Energia= 549 J', fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'},bbox=dict(facecolor='lightblue', alpha=0.5))


#  3  #

# Definire un filtro con risposta impulsiva pari ad una sinc e nuovo segnale pari ad un coseno con A=10 f_0=5. 
# Calcolare e disegnare l'uscita del filtro.

#definizione della risposta impulsiva 
def sinc(x):
    return np.sinc(x/np.pi)

#creo il segnale cos
A=10
f_0=5
x= A*np.cos(2*np.pi*f_0*t)

t_filter = np.linspace(-1, 1, 100)
risp = sinc(t_filter)

# Convoluzione
y = np.convolve(x, risp, mode='same') 

#creo il plot del segnale coseno
mpl.figure(figsize=(6, 7))
mpl.subplot(3, 1, 1)
mpl.plot(t, x,color='darkblue')
mpl.title('Segnale coseno',fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel('t',fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel('Ampiezza',fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()

#creo il plot della risposta impulsiva
mpl.subplot(3, 1, 2)
mpl.plot(t_filter, risp,color='orange')
mpl.title('Risposta impulsiva del filtro',fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel('t',fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel('Ampiezza',fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()

#creo il plot dell'uscita del filtro
mpl.subplot(3, 1, 3)
mpl.plot(t, y, color='green')
mpl.title('Uscita del filtro',fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.xlabel('t',fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.ylabel('Ampiezza',fontdict={'fontsize': 8, 'fontweight': 'bold', 'family': 'Arial'})
mpl.grid()

#visualizzo i plots
mpl.tight_layout()
mpl.show()


