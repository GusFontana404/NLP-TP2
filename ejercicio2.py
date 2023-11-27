#Importo las librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""## **Ejercicio 2**

### **Detección de patentes**

### **Item a**
"""

def obtener_patentes(imagen):
  """Segmenta imágenes de autos. Aplica transformaciones a la
  imagen de entrada, y se obtiene la subimagen de la patente"""
  #Cargar la imagen y pasar a escala de girses
  img = cv2.imread(imagen, cv2.IMREAD_COLOR)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #Aplicar filtro Sobel
  imagen_sobel = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=1)
  imagen_sobel = cv2.convertScaleAbs(imagen_sobel)

  #Binarizar la imagen
  imagen_binaria = np.where(imagen_sobel > 108, 1, 0)
  imagen_binaria = imagen_binaria.astype(np.uint8)

  #Aplicar dilatación
  kernel = np.ones((16,16),np.uint8)
  img_dilatada = cv2.dilate(imagen_binaria,kernel,iterations=1)

  #Morfología para mejorar la segmentación obtenida
  morf = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 24))
  #Apertura para remover elementos pequeños
  imagen_apertura = cv2.morphologyEx(img_dilatada, cv2.MORPH_OPEN, morf)

  #Detección de componentes conectadas
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_apertura,8, cv2.CV_32S)
  comp = img.copy()
  for st in stats[1:]:
    bounding = cv2.rectangle(comp,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(255,0,0),thickness=10)

  #Guardar la mayor relación de aspecto
  relacion_aspecto_mayor = 0
  #Guardar las coordenadas de la img con mayor relación de aspecto (patente)
  xx,yy,ll,hh = 0,0,0,0

  #Se itera sobre los stats de las componentes conectadas
  for st in stats:
    #Se filtran los bounding box por área (obtenida experimentalmente)
    if st[4] >= 1400 and st[4] <= 7000:

      #Se extrae las coordenadas de las patentes
      x, y, ancho, alto = st[0],st[1],st[2],st[3]

      #Calcular relación de aspecto (mayor relación de aspecto -> fig más planas)
      relacion_de_aspecto = float(ancho) / alto
      #Se remplaza la relación de aspecto si se encuentra una mejor (mayor)
      if relacion_de_aspecto > relacion_aspecto_mayor:
        relacion_aspecto_mayor = relacion_de_aspecto
        xx,yy,ll,hh = x, y, ancho, alto

  #Extraer ROI usando las coordenadas de la patente
  roi = img[yy:yy+hh, xx:xx+ll]
  return img, imagen_sobel, img_dilatada, imagen_apertura, bounding, roi

def mostrar_resultados(imagenes,f,c):
    """Función para graficar subplots, recibe una imagen y el
    alto y ancho de la figura a graficar."""
    # Configura el subplot
    fig, axs = plt.subplots(f, c, figsize=(12, 8),facecolor='#333333')
    fig.tight_layout(pad=0.5)

    # Itera sobre las imágenes y las muestra en el subplot
    for i in range(f):
        for j in range(c):
            img = imagenes[i * c + j]
            axs[i, j].imshow(img)
            axs[i, j].axis('off')

    plt.show()

"""### **Item b**"""

def segmentar_patentes(patente):
  """Recibe la imagen de una patente y retorno la segmentación
  de los valores dentro de ella aplicando transformaciones."""
  #Pasar a gris la imagen
  gray_roi = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)

  #Binarizar la imagen
  imagen_binaria = np.where(gray_roi > 121, 1, 0)
  imagen_binaria = imagen_binaria.astype(np.uint8)

  #Componentes conectadas
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria,8, cv2.CV_32S)
  im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)

  #Dibujar rectángulos
  for st in stats:
    if st[4] >= 16 and st[4] <= 100 and st[1] > 0 and st[0] > 0:
      bounding = cv2.rectangle(patente,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(255,0,0),thickness=1)
  return imagen_binaria, bounding

"""# **Mostrar resultados:**"""

#Cargo la ruta de las imágenes
ruta = '/content/drive/MyDrive/PI/TP-2/'

#Guardo las rutas completas
imagen_patente = [f'{ruta}img0{n}.png' if n < 10 else f'{ruta}img{n}.png' for n in range(1,13)];

#Segmento las imágenes y las guardo
patentes = [obtener_patentes(img)[5] for img in imagen_patente];
copia_patentes = [obtener_patentes(img)[5] for img in imagen_patente];

#Segmento las patentes y las guardo
lista_patentes = [patentes[1],patentes[3],patentes[4],patentes[5],patentes[6],patentes[8],patentes[9]]
patentes_segmentadas = [segmentar_patentes(img)[1] for img in lista_patentes];

#Mostrar la imagen original y los resultados
plt.figure(figsize=(15,8),facecolor='#333333')
plt.subplot(231), plt.imshow(obtener_patentes(imagen_patente[0])[0], cmap='gray'), plt.title('Imagen original',color='white'),plt.axis('off')
plt.subplot(232), plt.imshow(obtener_patentes(imagen_patente[0])[1], cmap='gray'), plt.title('Imagen con Sobel',color='white'),plt.axis('off')
plt.subplot(233), plt.imshow(obtener_patentes(imagen_patente[0])[2], cmap='gray'), plt.title('Imagen dilatada',color='white'),plt.axis('off')
plt.subplot(234), plt.imshow(obtener_patentes(imagen_patente[0])[3], cmap='gray'), plt.title('Imagen con morfología (apertura)',color='white'),plt.axis('off')
plt.subplot(235), plt.imshow(obtener_patentes(imagen_patente[0])[4], cmap='gray'), plt.title('Componentes conectadas',color='white'),plt.axis('off')
plt.subplot(236), plt.imshow(obtener_patentes(imagen_patente[0])[5], cmap='gray'), plt.title('ROI',color='white'),plt.axis('off')
plt.subplots_adjust(hspace=0.1)
plt.show()

#Mostrar la imagen original y los resultados
plt.figure(figsize=(15,8),facecolor='#333333')
plt.subplot(231), plt.imshow(copia_patentes[4], cmap='gray'), plt.title('Imagen original',color='white'),plt.axis('off')
plt.subplot(232), plt.imshow(segmentar_patentes(patentes[4])[0], cmap='gray'), plt.title('Imagen binaria',color='white'),plt.axis('off')
plt.subplot(233), plt.imshow(patentes[4], cmap='gray'), plt.title('Componentes conectadas',color='white'),plt.axis('off')
plt.subplots_adjust(hspace=0.1)
plt.show()

"""* ### **Las patenetes 1, 8 y 11 son detectadas con un umbral superior a 130.**

* ### **Las patenes 3 y 12 necesitan mas filtros para poder segmentarse**
"""
