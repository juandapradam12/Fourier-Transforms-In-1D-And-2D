#### Transformada de Fourier 2D

import numpy as np 
import matplotlib.pyplot as plt 
from scipy import fftpack
from PIL import Image
from scipy.ndimage import imread

## Lee las imagenes y las guarda en arreglos

# .convertL para que quede en escala de grises
 
Barcelona = Image.open('Barcelona.jpg').convert("L")
Paris = Image.open('Paris.jpg').convert("L")
Fractal = Image.open('frac.jpeg').convert("L")
Triangulos = Image.open('triangulos.png').convert("L")

## Figura con cuatro subplots guardada como .pdf 

imagenes = plt.figure()
Imagenes = [Barcelona, Paris, Fractal, Triangulos]

imagen_1 = imagenes.add_subplot(221)
imagen_2 = imagenes.add_subplot(222)
imagen_3 = imagenes.add_subplot(223)
imagen_4 = imagenes.add_subplot(224)

subplots = [imagen_1, imagen_2, imagen_3, imagen_4]

for subplt, img in zip(subplots, Imagenes):
	subplt.imshow(img, cmap = 'gray')
	subplt.axis('off')

plt.savefig('imagenes.pdf')
plt.clf()

## Transformada de Fourier de las cuatro imagenes

T_Barcelona = np.fft.fft2(Barcelona)
T_Paris = np.fft.fft2(Paris)
T_Fractal = np.fft.fft2(Fractal)
T_Triangulos = np.fft.fft2(Triangulos)

## Figura con cuatro subplots de las transformadas guardada como .pdf 

transformadas = plt.figure()
T_Imagenes = [T_Barcelona, T_Paris, T_Fractal, T_Triangulos]

T_imagen_1 = transformadas.add_subplot(221)
T_imagen_2 = transformadas.add_subplot(222)
T_imagen_3 = transformadas.add_subplot(223)
T_imagen_4 = transformadas.add_subplot(224)

T_subplots = [T_imagen_1, T_imagen_2, T_imagen_3, T_imagen_4]

delta = 1.0

for T_subplt, T_img in zip(T_subplots, T_Imagenes):
	T_subplt.imshow(np.log(np.abs(T_img) + delta), cmap = 'inferno')
	T_subplt.axis('off')

plt.savefig('transformadas.pdf')
plt.clf()

## Corte transversal horizontal en el centro de las transformadas 

cortes_horizontales = plt.figure()

ch_1 = cortes_horizontales.add_subplot(221)
ch_2 = cortes_horizontales.add_subplot(222)
ch_3 = cortes_horizontales.add_subplot(223)
ch_4 = cortes_horizontales.add_subplot(224)

ch = [ch_1, ch_2, ch_3, ch_4]

for ch_i, cT_img in zip(ch, T_Imagenes):
	ch_i.plot(np.log(np.abs(T_img[int(np.shape(T_img)[1]/2), :]) + delta))

plt.savefig('cortes_transversales.pdf')
plt.clf()

## Barcelona sin horizontales  

bcn = np.asarray(T_Barcelona)

# Defino el Kernel de la transformacion (elimina las frecuencias bajas y modificado con los parametros tales que eliminan las lineas horizontales de BCN)

def Ker(x, C):
	Ker = 1-np.exp(-C*x**2)
	return (np.array([Ker])).T

# Parametros para el Kernel

d = 8
C = np.pi
x = np.linspace(-bcn.shape[0], bcn.shape[0], int(bcn.shape[0]/d))

# Transformada del Kernel 

Kernel = Ker(x, C)
T_Kernel = np.fft.fft2(Kernel, s = bcn.shape, axes=(0, 1))

# Convoluciono la transformada del Kernel y la de la imagen de BCN en el espacio de Fourier (en este espacio es simplemente una multiplicacion usual)

Conv_en_F = T_Kernel[:, :] * T_Barcelona

# Hago la transformada para volver al espacio de posicion

IT_bcn = np.real(np.fft.ifft2(Conv_en_F, s = bcn.shape))

# Guarda la imagen .pdf

sin_horizontales = plt.figure()

plt.imshow(IT_bcn, cmap='gray')
plt.savefig('sin_horizontales.pdf')
plt.show()
plt.clf()

