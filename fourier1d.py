#### Transformada de Fourier 1D

import numpy as np 
import matplotlib.pyplot as plt 
import numpy.fft as fft

## Lectura de Datos 

Datos=np.loadtxt('funcion.dat')

x = Datos[:,0] 
y = Datos[:,1] 

## Constantes 

N = np.shape(Datos)[0]

## Transformada de Fourier

def T_Fourier(z): # Entra por parametro un array Nx1 (256x1)
	# Indices del Kernel
	n = np.arange(N)
	k = np.reshape(n, (N, 1))
	# Matriz del Kernel de la transformacion NxN (256x256)
	M_nk = np.exp(-(2j*np.pi*n*k)/N)
	# Datos en el espacio de frecuencias Nx1 (256x1)
	T_z = np.dot(M_nk, z)
	return T_z
	
## Datos en el espacio de Frecuencias 

dt = x[1]-x[0]

T_x = fft.fftfreq(len(y), dt)
T_y = T_Fourier(y)

## Grafica (curiosidad)

plt.plot(T_x, abs(T_y))
#plt.show()

## Frecuencia

T_y_Max = np.argmax(T_y)
f = -1*T_x[T_y_Max]


## Imprime en la consola

print("La frecuencia es: {}, donde {} es el resultado obtenido con el analisis de Fourier." .format(f, f))

