import math
import os
import glob
import numpy as np
import cv2
import statistics

labels = []

def load_images(path):
    images = []

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            for file in glob.glob(os.path.join(subdir_path, '*.png')):
                image = cv2.imread(file)
                resized_img = cv2.resize(image, (50, 50))
                images.append(resized_img)
                aux = subdir_path.split('\\')
                labels.append(aux[1])
    return images

#METODO PARA CALCULAR LAS MEDIAS EN CADA ESPECTRO CON UNA VENTANA DE 10
def media(vector):
    longitud = len(vector)
    mediaC = []
    for i in range(1,longitud,10):
        mean = statistics.mean(vector[i:i+9])
        mediaC.append(mean)

    return mediaC

#METODO PARA CALCULAR LAS MODAS EN CADA ESPECTRO CON UNA VENTANA DE 10
def moda(vector):
    longitud = len(vector)
    modaC = []
    for i in range(1,longitud,10):
        mode = statistics.mode(vector[i:i+9])
        modaC.append(mode)

    return modaC

#METODO PARA CALCULAR LAS MEDIANAS EN CADA ESPECTRO CON UNA VENTANA DE 10
def mediana(vector):
    longitud = len(vector)
    medianaC = []
    for i in range(1,longitud,10):
        median = statistics.median(vector[i:i+9])
        medianaC.append(median)

    return medianaC

#METODO PARA CALCULAR LAS DESVIACIONES ESTANDAR EN CADA ESPECTRO CON UNA VENTANA DE 10
def desvEstandar(vector):
    longitud = len(vector)
    desvEst = []
    for i in range(1,longitud,10):
        datos = vector[i:i+9]
        media = sum(vector[i:i+9]) / 10
        suma_diferencias_cuadradas = sum([(dato - media) ** 2 for dato in datos])
        varianza = suma_diferencias_cuadradas / len(datos)
        desviacion_estandar = math.sqrt(varianza)
        desvEst.append(desviacion_estandar)
    return desvEst

#METODO PARA CALCULAR LAS VARIANZAS EN CADA ESPECTRO CON UNA VENTANA DE 10
def varianza(vector):
    longitud = len(vector)
    var = []
    for i in range(1,longitud,10):
        varianza = np.var(vector[i:i+9])
        var.append(varianza)

    return var

#METODO PARA CALCULAR RMS (Raiz Media Cuadratica) EN CADA ESPECTRO CON UNA VENTANA DE 10
def mediaCuadratica(vector):
    longitud = len(vector)
    medCuadratica = []

    for i in range(1,longitud,10):
        datos = vector[i:i+9]
        suma_diferencias_cuadradas = sum([x ** 2 for x in datos])
        media_cuadratica = suma_diferencias_cuadradas / len(datos)
        rms = math.sqrt(media_cuadratica)
        medCuadratica.append(rms)

    return medCuadratica

#METODO PARA CALCULAR LAS MEDIAS GEOMETRICAS EN CADA ESPECTRO CON UNA VENTANA DE 10
def mediaGeo(vector):
    longitud = len(vector)
    mGeo = []

    for i in range(1,longitud,10):
        producto = np.prod(vector[i:i+9])
        res = math.pow(producto, 1/10)
        mGeo.append(res)
    return mGeo


#METODO PARA UNIR EN UN SOLO VECTOR LOS VECTORES DE LOS ESPECTROS
def unirEspectrosFuncion(v1, v2, v3):
    caracteristicasImagen = v1 + v2 + v3
    return caracteristicasImagen


# Cargar imágenes desde la carpeta 'data' y sus subcarpetas
imagesTrain = load_images('Colores')

imagesTest = load_images('TestColores')

l = len(imagesTrain)



# Procesar y extraer características de las imágenes de entrenamiento
XTrain = []
YTrain = []
i = 0
for image in imagesTrain:
    b, g, r = cv2.split(image)

    #SEPARAR LA IMAGEN EN ESPECTROS
    vectorB = b.ravel()
    vectorG = g.ravel()
    vectorR = r.ravel()

    #CALCULO DE LAS MEDIA EN CADA ESPECTRO
    mediaB = media(vectorB)
    mediaG = media(vectorG)
    mediaR = media(vectorR)

    #CALCULO DE LAS MODAS EN CADA ESPECTRO
    modaB = moda(vectorB)
    modaG = moda(vectorG)
    modaR = moda(vectorR)

    #CALCULO DE LAS MEDIANAS EN CADA ESPECTRO
    medianaB = mediana(vectorB)
    medianaG = mediana(vectorG)
    medianaR = mediana(vectorR)

    #CALCULO DE LAS DESVIACIONES ESTANDAR EN CADA ESPECTRO
    desvEstandarB = desvEstandar(vectorB)
    desvEstandarG = desvEstandar(vectorG)
    desvEstandarR = desvEstandar(vectorR)

    #CALCULO DE LAS VARIANZAS EN CADA ESPECTRO
    varianzaB = varianza(vectorB)
    varianzaG = varianza(vectorG)
    varianzaR = varianza(vectorR)

    #CALCULO DE LAS MEDIAS CUADRATICAS EN CADA ESPECTRO
    mediaCuadB = mediaCuadratica(vectorB)
    mediaCuadG = mediaCuadratica(vectorG)
    mediaCuadR = mediaCuadratica(vectorR)

    #CALCULO DE LAS MEDIAS GEOMETRICAS EN CADA ESPECTRO
    mediaGeoB = mediaGeo(vectorB)
    mediaGeoG = mediaGeo(vectorG)
    mediaGeoR = mediaGeo(vectorR)

    #UNION DE ESPECTROS POR FUNCION ESTADISTICA
    mediaEspectrosUnidos = unirEspectrosFuncion(mediaB, mediaR, mediaG)
    modaEspectrosUnidos = unirEspectrosFuncion(modaB, modaR, modaG)
    medianaEspectrosUnidos = unirEspectrosFuncion(medianaB, medianaR, medianaG)
    desvEstandarEspectrosUnidos = unirEspectrosFuncion(desvEstandarB, desvEstandarR, desvEstandarG)
    varianzaEspectrosUnidos = unirEspectrosFuncion(varianzaB, varianzaR, varianzaG)
    mediaCuadEspectrosUnidos = unirEspectrosFuncion(mediaCuadB, mediaCuadR, mediaCuadG)
    mediaGeoEspectosUnidos = unirEspectrosFuncion(mediaGeoB, mediaGeoR, mediaGeoG)

    caracteristicas = mediaEspectrosUnidos + modaEspectrosUnidos + medianaEspectrosUnidos + desvEstandarEspectrosUnidos + varianzaEspectrosUnidos + mediaCuadEspectrosUnidos + mediaGeoEspectosUnidos
    XTrain.append(caracteristicas)

    YTrain.append(labels[i])

    i = i + 1

f = len(XTrain)
lo = len(XTrain[0])

#print(y)
#print("Filas: ",f)
#print("Columnas: ",lo)

# Procesar y extraer características de las imágenes de testeo
XTest = []
YTest = []
i = 0
for image in imagesTest:
    b, g, r = cv2.split(image)

    #SEPARAR LA IMAGEN EN ESPECTROS
    vectorB = b.ravel()
    vectorG = g.ravel()
    vectorR = r.ravel()

    #CALCULO DE LAS MEDIA EN CADA ESPECTRO
    mediaB = media(vectorB)
    mediaG = media(vectorG)
    mediaR = media(vectorR)

    #CALCULO DE LAS MODAS EN CADA ESPECTRO
    modaB = moda(vectorB)
    modaG = moda(vectorG)
    modaR = moda(vectorR)

    #CALCULO DE LAS MEDIANAS EN CADA ESPECTRO
    medianaB = mediana(vectorB)
    medianaG = mediana(vectorG)
    medianaR = mediana(vectorR)

    #CALCULO DE LAS DESVIACIONES ESTANDAR EN CADA ESPECTRO
    desvEstandarB = desvEstandar(vectorB)
    desvEstandarG = desvEstandar(vectorG)
    desvEstandarR = desvEstandar(vectorR)

    #CALCULO DE LAS VARIANZAS EN CADA ESPECTRO
    varianzaB = varianza(vectorB)
    varianzaG = varianza(vectorG)
    varianzaR = varianza(vectorR)

    #CALCULO DE LAS MEDIAS ARMONICAS EN CADA ESPECTRO
    mediaCuadB = mediaCuadratica(vectorB)
    mediaCuadG = mediaCuadratica(vectorG)
    mediaCuadR = mediaCuadratica(vectorR)

    #CALCULO DE LAS MEDIAS GEOMETRICAS EN CADA ESPECTRO
    mediaGeoB = mediaGeo(vectorB)
    mediaGeoG = mediaGeo(vectorG)
    mediaGeoR = mediaGeo(vectorR)


    #UNION DE ESPECTROS POR FUNCION ESTADISTICA
    mediaEspectrosUnidos = unirEspectrosFuncion(mediaB, mediaR, mediaG)
    modaEspectrosUnidos = unirEspectrosFuncion(modaB, modaR, modaG)
    medianaEspectrosUnidos = unirEspectrosFuncion(medianaB, medianaR, medianaG)
    desvEstandarEspectrosUnidos = unirEspectrosFuncion(desvEstandarB, desvEstandarR, desvEstandarG)
    varianzaEspectrosUnidos = unirEspectrosFuncion(varianzaB, varianzaR, varianzaG)
    mediaCuadEspectrosUnidos = unirEspectrosFuncion(mediaCuadB, mediaCuadR, mediaCuadG)
    mediaGeoEspectosUnidos = unirEspectrosFuncion(mediaGeoB, mediaGeoR, mediaGeoG)

    caracteristicas = mediaEspectrosUnidos + modaEspectrosUnidos + medianaEspectrosUnidos + desvEstandarEspectrosUnidos + varianzaEspectrosUnidos + mediaCuadEspectrosUnidos + mediaGeoEspectosUnidos
    XTest.append(caracteristicas)

    YTest.append(labels[i])

    i = i + 1

f = len(XTrain)
lo = len(XTrain[0])

#print(y)
#print("Filas: ",f)
#print("Columnas: ",lo)






