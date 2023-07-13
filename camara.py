import math
import pickle
import statistics
import time

import cv2
import numpy as np
from sklearn import preprocessing

import tkinter as tk
from tkinter import messagebox
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

root = tk.Tk()
root.title("COLORES")

label = tk.Label(root, text="Colores")
label.pack()


#METODO PARA CALCULAR LAS MEDIAS EN CADA ESPECTRO CON UNA VENTANA DE 5
def media(vector):
    longitud = len(vector)
    mediaC = []
    for i in range(1,longitud,5):
        mean = statistics.mean(vector[i:i+4])
        mediaC.append(mean)

    return mediaC

#METODO PARA CALCULAR LAS MODAS EN CADA ESPECTRO CON UNA VENTANA DE 5
def moda(vector):
    longitud = len(vector)
    modaC = []
    for i in range(1,longitud,5):
        mode = statistics.mode(vector[i:i+4])
        modaC.append(mode)

    return modaC

#METODO PARA CALCULAR LAS MEDIANAS EN CADA ESPECTRO CON UNA VENTANA DE 5
def mediana(vector):
    longitud = len(vector)
    medianaC = []
    for i in range(1,longitud,5):
        median = statistics.median(vector[i:i+4])
        medianaC.append(median)

    return medianaC

#METODO PARA CALCULAR LAS DESVIACIONES ESTANDAR EN CADA ESPECTRO CON UNA VENTANA DE 5
def desvEstandar(vector):
    longitud = len(vector)
    desvEst = []
    for i in range(1,longitud,5):
        datos = vector[i:i+4]
        media = sum(vector[i:i+4]) / 5
        suma_diferencias_cuadradas = sum([(dato - media) ** 2 for dato in datos])
        varianza = suma_diferencias_cuadradas / len(datos)
        desviacion_estandar = math.sqrt(varianza)
        desvEst.append(desviacion_estandar)
    return desvEst


#METODO PARA CALCULAR RMS (Raiz Media Cuadratica) EN CADA ESPECTRO CON UNA VENTANA DE 5
def mediaCuadratica(vector):
    longitud = len(vector)
    medCuadratica = []

    for i in range(1,longitud,5):
        datos = vector[i:i+4]
        suma_diferencias_cuadradas = sum([x ** 2 for x in datos])
        media_cuadratica = suma_diferencias_cuadradas / len(datos)
        rms = math.sqrt(media_cuadratica)
        medCuadratica.append(rms)

    return medCuadratica

#METODO PARA CALCULAR LAS MEDIAS GEOMETRICAS EN CADA ESPECTRO CON UNA VENTANA DE 5
def mediaGeo(vector):
    longitud = len(vector)
    mGeo = []

    for i in range(1,longitud,5):
        producto = np.prod(vector[i:i+4])
        res = math.pow(producto, 1/5)
        mGeo.append(res)
    return mGeo

#METODO PARA CALCULAR LAS VARIANZAS EN CADA ESPECTRO CON UNA VENTANA DE 5
def varianza(vector):
    longitud = len(vector)
    var = []
    for i in range(1,longitud,5):
        varianza = np.var(vector[i:i+4])
        var.append(varianza)

    return var

#CARACTERISTICAS DE UNA IMAGEN
def procesamientoImagen(imagen):
    #SEPARA LA IMAGEN LOS 3 ESPECTROS
    b, g, r = cv2.split(imagen)

    #SE CONVIERTE CADA ESPECTRO EN UN VECTOR
    vectorB = b.ravel()
    vectorG = g.ravel()
    vectorR = r.ravel()

    # CALCULO DE LAS MEDIA EN CADA ESPECTRO
    mediaB = media(vectorB)
    mediaG = media(vectorG)
    mediaR = media(vectorR)

    # CALCULO DE LAS MODAS EN CADA ESPECTRO
    modaB = moda(vectorB)
    modaG = moda(vectorG)
    modaR = moda(vectorR)

    # CALCULO DE LAS MEDIANAS EN CADA ESPECTRO
    medianaB = mediana(vectorB)
    medianaG = mediana(vectorG)
    medianaR = mediana(vectorR)

    # CALCULO DE LAS DESVIACIONES ESTANDAR EN CADA ESPECTRO
    desvEstandarB = desvEstandar(vectorB)
    desvEstandarG = desvEstandar(vectorG)
    desvEstandarR = desvEstandar(vectorR)

    # CALCULO DE LAS VARIANZAS EN CADA ESPECTRO
    varianzaB = varianza(vectorB)
    varianzaG = varianza(vectorG)
    varianzaR = varianza(vectorR)

    # CALCULO DE LAS MEDIAS GEOMETRICAS EN CADA ESPECTRO
    mediaGeoB = mediaGeo(vectorB)
    mediaGeoG = mediaGeo(vectorG)
    mediaGeoR = mediaGeo(vectorR)

    # CALCULO DE LAS MEDIAS GEOMETRICAS EN CADA ESPECTRO
    mediaCuaB = mediaCuadratica(vectorB)
    mediaCuaG = mediaCuadratica(vectorG)
    mediaCuaR = mediaCuadratica(vectorR)

    # UNION DE ESPECTROS POR FUNCION ESTADISTICA
    mediaEspectrosUnidos = unirEspectrosFuncion(mediaB, mediaR, mediaG)
    modaEspectrosUnidos = unirEspectrosFuncion(modaB, modaR, modaG)
    medianaEspectrosUnidos = unirEspectrosFuncion(medianaB, medianaR, medianaG)
    desvEstandarEspectrosUnidos = unirEspectrosFuncion(desvEstandarB, desvEstandarR, desvEstandarG)
    varianzaEspectrosUnidos = unirEspectrosFuncion(varianzaB, varianzaR, varianzaG)
    mediaGeoEspectosUnidos = unirEspectrosFuncion(mediaGeoB, mediaGeoR, mediaGeoG)
    mediaCuaEspectrosUnidos = unirEspectrosFuncion(mediaCuaB, mediaCuaR, mediaCuaG)

    caracteristicas = mediaEspectrosUnidos + modaEspectrosUnidos + medianaEspectrosUnidos + desvEstandarEspectrosUnidos + varianzaEspectrosUnidos + mediaCuaEspectrosUnidos + mediaGeoEspectosUnidos

    return caracteristicas


#METODO PARA UNIR EN UN SOLO VECTOR LOS VECTORES DE LOS ESPECTROS
def unirEspectrosFuncion(v1, v2, v3):
    caracteristicasImagen = v1 + v2 + v3
    return caracteristicasImagen

#METODO PARA AJUSTAR EL BRILLO
def adjust_brightness(img, brightness_factor):
    """Ajusta el brillo de una imagen"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * brightness_factor, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


with open('knn_model.pkl', 'rb') as file:
    clf = pickle.load(file)
# Cargar la función entrenada
# clf = ... # aquí debes cargar la función entrenada
CaracteristicasSuperior = []
CaracteristicasInferior = []

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while(True):
    # Capturar una imagen de la cámara
    frames = []
    for i in range(2):
        ret, frame = cap.read()
        img = cv2.resize(frame, (20, 20))
        frames.append(img)


    # Preprocesar la imagen
    for i in range(2):
        imagen = frames[i]

        image = adjust_brightness(imagen, 1.5)

        # Dividir la imagen en dos
        height, width = imagen.shape[:2]

        #area = cv2.resize()
        #top_half = imagen[:height // 2, :]

        #bottom_half = imagen[height // 2:, :]


        #Centroide de cada parte
        #heightT, widthT = top_half.shape[:2]
        #heightI, widthI = bottom_half.shape[:2]

        # Calcular el centro de la imagen
        center = (width // 2, height // 2)
        #centerT = (widthT // 2, heightT // 2)
        #centerI = (widthI // 2, heightI // 2)

        # Seleccionar una área de 14x14 píxeles en el centro de la imagen
        area = img[center[1] - 7:center[1] + 7, center[0] - 7:center[0] + 7]
        area = cv2.resize(area, (20, 20))
        #areaSuperior = img[centerT[1] - 7:centerT[1] + 7, centerT[0] - 7:centerT[0] + 7]
        #areaSuperior = cv2.resize(areaSuperior, (20, 20))
        #areaInferior = img[centerI[1] - 7:centerI[1] + 7, centerI[0] - 7:centerI[0] + 7]
        #areaInferior = cv2.resize(areaInferior, (20, 20))


        #VECTOR DE CARACTERISTICAS DE LA IMAGEN
        caracteristicasT = procesamientoImagen(area)
        #caracteristicasS = procesamientoImagen(areaSuperior)
        #caracteristicasI = procesamientoImagen(areaInferior)

        CaracteristicasSuperior.append(caracteristicasT)
        #CaracteristicasSuperior.append(caracteristicasS)
        #CaracteristicasInferior.append(caracteristicasI)



    # Utilizar la función entrenada para predecir el color
    prediction = clf.predict(CaracteristicasSuperior)
    #predictionS = clf.predict(CaracteristicasSuperior)
    #predictionI = clf.predict(CaracteristicasInferior)

    #IMPRIMIR LAS ULTIMAS 3 ETIQUETAS

    # Mostrar la imagen y el resultado
    cv2.imshow("Camera", frame)
    print("                              ")
    print("-----------------------------")
    print("COLORES DETECTADOS")
    print("COLORES DETECTADOS:", prediction[-2:])
    #print("PARTE SUPERIOR:", predictionS[-2:])
    print("                              ")
    #print("PARTE INFERIOR:", predictionI[-2:])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    label.config(text="Colores detectados: {}".format(prediction[-2:]))
    root.update()

    #time.sleep(3)

# Liberar la cámara y cerrar la ventana
root.mainloop()
cap.release()
cv2.destroyAllWindows()