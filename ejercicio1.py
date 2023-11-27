#Importo las librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""## **Ejercicio 1**

### **Detección y clasificación de Monedas y Dados**
"""

#Cargar la imagen
img = cv2.imread('/content/drive/MyDrive/PI/TP-2/monedas.jpg')

#Imagen para prueba final
img_prueba = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Pasar a escala de grises
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Aplicar borrosidad
gauss = cv2.GaussianBlur(img_gray,(7,7),3)

#Detectar bordes
canny = cv2.Canny(gauss,77.6,5)

#Aplicar dilatación
kernel_1 = np.ones((5,5),np.uint8)
img_mod = cv2.dilate(canny,kernel_1,iterations=5)

#Aplicar erosión
kernel_2 = np.ones((5,5),np.uint8)
img_erosion = cv2.erode(img_mod,kernel_2,iterations=2)

#Binarizar la imagen procesada
th, binary_img = cv2.threshold(img_erosion, 0, 1, cv2.THRESH_BINARY)

#Morfología para mejorar la segmentación obtenida
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

#Clausura para rellenar huecos.
clausura_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, se)

# Mostrar la imagen original y los resultados
plt.figure(figsize=(15,8),facecolor='#333333')
plt.subplot(231), plt.imshow(img_prueba), plt.title('Imagen Original',color='white'),plt.axis('off')
plt.subplot(232), plt.imshow(gauss, cmap='gray'), plt.title('Imagen con borrosidad (Gauss)',color='white'),plt.axis('off')
plt.subplot(233), plt.imshow(canny, cmap='gray'), plt.title('Detección de lineas (Canny)',color='white'),plt.axis('off')
plt.subplot(234), plt.imshow(img_mod, cmap='gray'), plt.title('Imagen dilatada',color='white'),plt.axis('off')
plt.subplot(235), plt.imshow(img_erosion, cmap='gray'), plt.title('Imagen erosionada',color='white'),plt.axis('off')
plt.subplot(236), plt.imshow(clausura_img, cmap='gray'), plt.title('Imagen con morfología (clausura)',color='white'),plt.axis('off')
plt.subplots_adjust(hspace=0.1)
plt.show()

def contar_huecos(img_original, imagen_binaria, posicion):
  """La función recibe la imagen original, una binarizada y la posición del
  contorno a procesar, luego se extrae la ROI y se la procesa para poder
  detectar los contorno interntos. La función retorna la cantidad de contornos
  detectados"""

  #Detectar contornos externos
  contornos, jerarquia = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  #Encuentra las coordenadas de los dados
  x, y, ancho, alto = cv2.boundingRect(contornos[posicion])

  #Extrae la ROI usando las coordenadas del dado
  roi = img_original[y:y+alto, x:x+ancho]

  #Aplicar borrosidad
  gau = cv2.GaussianBlur(roi,(15,15),0)

  #Detectar bordes
  can = cv2.Canny(gau,200,135)

  #Encontrar contornos internos
  contornos, jerarquia = cv2.findContours(can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  #Contar los contornos encontrados
  contornos_internos = sum(1 for h in jerarquia[0] if h[2] != -1)

  #Dibujar contornos
  roi_contorno = roi.copy()
  for i, contour in enumerate(contornos):
    if jerarquia[0][i][3] != -1:
        dibujar_contorno = cv2.drawContours(roi_contorno, [contour], -1, (255, 0, 0), thickness=15)

  return contornos_internos, roi, gau, can, dibujar_contorno

#Mostrar la imagen original y los resultados
plt.figure(figsize=(15,8),facecolor='#333333')
plt.subplot(241), plt.imshow(contar_huecos(img_prueba,clausura_img,10)[1], cmap='gray'), plt.title('ROI dado',color='white'),plt.axis('off')
plt.subplot(242), plt.imshow(contar_huecos(img_prueba,clausura_img,10)[2], cmap='gray'), plt.title('Imagen con borrosidad (Gauss)',color='white'),plt.axis('off')
plt.subplot(243), plt.imshow(contar_huecos(img_prueba,clausura_img,10)[3], cmap='gray'), plt.title('Detección de lineas (Canny)',color='white'),plt.axis('off')
plt.subplot(244), plt.imshow(contar_huecos(img_prueba,clausura_img,10)[4], cmap='gray'), plt.title('Imagen con contornos',color='white'),plt.axis('off')
plt.subplots_adjust(hspace=0.1)
plt.show()

def detectar_objeto(img_bin, img_orig):
  """Función que permite detectar, contar y clasificar monedas y dados. Devuelve
  el reultado etiquetado sobre la imagen. Recibe como argumento la imagen
  binaria y la imagen original"""

  #Encontrar contornos externos
  contornos, jerarquia = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  #Buscar componentes conectadas
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin)

  #Inicializar contadores
  total_monedas = 0
  moneda_10 = 0
  moneda_50 = 0
  moneda_peso = 0
  dado = 0

  #Crear una copia de la imagen original
  imagen_contornos = img_prueba.copy()

  #Umbral de área para aplicar etiquetas (detectadas experimentalmente)
  area_umbral_10 = 85000
  area_umbral_50 = 97000

  #Iterar sobre los contornos
  for c, contorno in enumerate(contornos):
      #Calcular el área del contorno
      area = cv2.contourArea(contorno)

      #Aplicar etiquetas e incrementar variables
      #Primero se itera sobre las monedas
      if c != 10 and c != 18:
          if area <= area_umbral_10:
              #cv2.drawContours(imagen_contornos, [contorno], -1, (0, 255, 0), thickness=15)
              etiqueta = "10c"
              cv2.putText(imagen_contornos, etiqueta, (contorno[0][0][0]-100, contorno[0][0][1]+190), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 0), 15)
              moneda_10 += 1
              total_monedas += 1
          elif area >= area_umbral_50:
              #cv2.drawContours(imagen_contornos, [contorno], -1, (0, 255, 0), thickness=15)
              etiqueta = "50c"
              cv2.putText(imagen_contornos, etiqueta, (contorno[0][0][0]-80, contorno[0][0][1]+220), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 0), 15)
              moneda_50 += 1
              total_monedas += 1
          else:
              #cv2.drawContours(imagen_contornos, [contorno], -1, (0, 255, 0), thickness=15)
              etiqueta = "1p"
              cv2.putText(imagen_contornos, etiqueta, (contorno[0][0][0]-70, contorno[0][0][1]+190), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 0), 15)
              moneda_peso += 1
              total_monedas += 1
      #Si no son monedas, son dados
      else:
          #cv2.drawContours(imagen_contornos, [contorno], -1, (0, 255, 0), thickness=15)
          etiqueta = "Dado"
          dado += 1
          nro_dados = contar_huecos(img_orig,img_bin,c)[0]
          if c == len(contornos) - 1:
            cv2.putText(imagen_contornos, f'{nro_dados}', (contorno[0][0][0]+115, contorno[0][0][1]+140), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 23)
          else:
            cv2.putText(imagen_contornos, f'{nro_dados}', (contorno[0][0][0]+110, contorno[0][0][1]+140), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 23)

  #Agregar una leyenda a la imagen de salida
  cv2.putText(imagen_contornos, f'Hay {total_monedas} monedas en total', (50,2150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10)
  cv2.putText(imagen_contornos, f'Hay {moneda_10} de 10 centavos', (50,2250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10)
  cv2.putText(imagen_contornos, f'Hay {moneda_50} de 50 centavos', (50,2350), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10)
  cv2.putText(imagen_contornos, f'Hay {moneda_peso} de 1 peso', (50,2450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10)
  cv2.putText(imagen_contornos, f'Hay {dado} dados', (50,2550), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10)

  #Marcar los contornos de las figuras
  for st in stats[1:]:
    if st[4] > 3700:
      cv2.rectangle(imagen_contornos,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,0,255),thickness=15)

  #Ploteo
  plt.figure(figsize=(12, 8)),plt.imshow(imagen_contornos),plt.axis('off') ,plt.show()
