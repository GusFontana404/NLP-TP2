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
    #Plotear imágenes
    plt.show()

"""### **Item b**"""

def segmentar_patentes(img):
  """Recibe la imagen de una patente y retorno la segmentación
  de los valores dentro de ella aplicando transformaciones."""
  #Pasar a gris la imagen
  gray_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #Binarizar la imagen
  imagen_binaria = np.where(gray_roi > 121, 1, 0)
  imagen_binaria = imagen_binaria.astype(np.uint8)

  #Componentes conectadas
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria,8, cv2.CV_32S)
  im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)

  # Lista para almacenar las ROIs
  rois = []
  patente_copia = img.copy()

  #Dibujar rectángulos
  for st in stats:
    if st[0] > 0 and st[4] >= 16 and st[4] <= 100 and st[1] > 0:
      # Extraer la ROI y agregar a la lista
      roi = imagen_binaria[st[1]:st[1]+st[3], st[0]:st[0]+st[2]]
      rois.append(roi)
      #Dibujar los rectángulos en la componente conectada
      bounding = cv2.rectangle(patente_copia,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(255,0,0),thickness=1)

  return imagen_binaria, bounding, rois, img

"""# **Mostrar resultados:**"""

#Cargo la ruta de las imágenes
ruta = '/content/drive/MyDrive/PI/TP-2/'

#Guardo las rutas completas
imagen_patente = [f'{ruta}img0{n}.png' if n < 10 else f'{ruta}img{n}.png' for n in range(1,13)];

#Segmento las imágenes y las guardo
patentes = [obtener_patentes(img)[5] for img in imagen_patente];

#Segmento las patentes y las guardo
copia_patentes = patentes.copy();
lista_patentes = [copia_patentes[1],copia_patentes[3],copia_patentes[4],copia_patentes[5],copia_patentes[6],copia_patentes[8],copia_patentes[9]]
patentes_segmentadas = [segmentar_patentes(img)[1] for img in lista_patentes];

#Todas las patente procesadas
todas_las_patentes = [patentes[0],patentes_segmentadas[0],patentes[2],patentes_segmentadas[1],patentes_segmentadas[2],patentes_segmentadas[3],
                      patentes_segmentadas[4],patentes[7],patentes_segmentadas[5],patentes_segmentadas[6],patentes[10],patentes[11]]

"""### **Ejemplo de procesamiento de una imagen**"""

img, sob, dil, aper, bound, roi = obtener_patentes(imagen_patente[0])

#Mostrar la imagen original y los resultados
plt.figure(figsize=(16,8),facecolor='#333333')
plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Imagen original',color='white'),plt.axis('off')
plt.subplot(232), plt.imshow(sob, cmap='gray'), plt.title('Imagen con Sobel',color='white'),plt.axis('off')
plt.subplot(233), plt.imshow(dil, cmap='gray'), plt.title('Imagen dilatada',color='white'),plt.axis('off')
plt.subplot(234), plt.imshow(aper, cmap='gray'), plt.title('Imagen con morfología (apertura)',color='white'),plt.axis('off')
plt.subplot(235), plt.imshow(bound, cmap='gray'), plt.title('Componentes conectadas',color='white'),plt.axis('off')
plt.subplot(236), plt.imshow(roi, cmap='gray'), plt.title('ROI',color='white'),plt.axis('off')
plt.subplots_adjust(hspace=0.1)
plt.show()

"""### **Ejemplo de una patente procesada**"""

patente_plot = patentes.copy()
img_bin, bounding, rois, patente = segmentar_patentes(patente_plot[4])

#Mostrar la imagen original y los resultados
plt.figure(figsize=(15,5),facecolor='#333333')
plt.subplot(131), plt.imshow(patente, cmap='gray'), plt.title('Imagen original',color='white'),plt.axis('off')
plt.subplot(132), plt.imshow(img_bin, cmap='gray'), plt.title('Imagen binaria',color='white'),plt.axis('off')
plt.subplot(133), plt.imshow(bounding, cmap='gray'), plt.title('Componentes conectadas',color='white'),plt.axis('off')
plt.subplots_adjust(hspace=0.1)
plt.show()

"""### **Todas las imágenes segmentadas**"""

mostrar_resultados(patentes,4,3)

"""### **Bounding box sobre las patentes**"""

mostrar_resultados(todas_las_patentes,4,3)

"""### **Ejemplo de segmentar una patente**"""

#Mostrar la imagen original y los resultados
plt.figure(figsize=(15,8),facecolor='#333333')
plt.subplot(161), plt.imshow(rois[5], cmap='gray'), plt.title('Caracter 1',color='white'),plt.axis('off')
plt.subplot(162), plt.imshow(rois[4], cmap='gray'), plt.title('Caracter 2',color='white'),plt.axis('off')
plt.subplot(163), plt.imshow(rois[3], cmap='gray'), plt.title('Caracter 3',color='white'),plt.axis('off')
plt.subplot(164), plt.imshow(rois[2], cmap='gray'), plt.title('Caracter 4',color='white'),plt.axis('off')
plt.subplot(165), plt.imshow(rois[1], cmap='gray'), plt.title('Caracter 5',color='white'),plt.axis('off')
plt.subplot(166), plt.imshow(rois[0], cmap='gray'), plt.title('Caracter 6',color='white'),plt.axis('off')
plt.subplots_adjust(hspace=0.1)
plt.show()

"""* ### **Las patenetes 1, 8 y 11 son detectadas con un umbral superior a 130.**

* ### **Las patenes 3 y 12 necesitan mas filtros para poder segmentarse**
"""
