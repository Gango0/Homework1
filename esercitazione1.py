import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

A=5
phi=1
f=1/2
f1=1
t=np.linspace(0,4,80)

coswave= A* np.cos(np.pi*f*t+phi)
plt.subplot(2,1,1)
plt.xlim(0,5)
plt.ylim(-7,13)
plt.stem(t,coswave)
plt.title("Cosinusoide")
plt.xlabel("Tempo")
plt.ylabel("Ampiezza")

coswave_compr=A*np.cos(2*np.pi*f1*t+phi)

plt.subplot(2,1,2)
plt.xlim(0,5)
plt.ylim(-7,13)
plt.stem(t,coswave_compr)
plt.title("Cosinusoide compressa")
plt.xlabel("Tempo")
plt.ylabel("Ampiezza")



#calcolo dell'energia 
def integrand(t):
    return (A* np.cos(2*np.pi*t+phi))**2
a=0
b=4
res, err = quad(integrand, -2, 2)

print(f"valore dell'energia: {res:f} (+-{err:g})")
T=b-a
valore_medio= res//T

print("valore medio: ",valore_medio)


x1=0
y1=valore_medio 
x2=5
y2=valore_medio
plt.plot([x1,x2],[y1,y2])



plt.gca().add_patch(plt.Rectangle((4, 8), 3, 3, color='lightgrey', alpha=1, label='ENERGY'))
plt.text(4.5, 9.5, 'Energy: 50', ha='center', va='center', fontsize=10)


plt.tight_layout()
plt.show(block=True)