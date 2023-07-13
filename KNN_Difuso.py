import math
import os
import glob
import pickle
import warnings

import numpy as np
import cv2
import statistics
import FCVT as fcvt

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

labelsTraining = []
labelsTest = []

def load_images(path,op):
    images = []

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            for file in glob.glob(os.path.join(subdir_path, '*.png')):
                image = cv2.imread(file)
                resized_img = cv2.resize(image, (20, 20))
                images.append(resized_img)
                aux = subdir_path.split('\\')
                if op == 'training':
                    labelsTraining.append(aux[1])
                else:
                    labelsTest.append(aux[1])
    return images

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


# Cargar imágenes desde la carpeta 'data' y sus subcarpetas
imagesTrain = load_images('Colores','training')

imagesTest = load_images('TestColores','test')

l = len(imagesTrain)



# Procesar y extraer características de las imágenes de entrenamiento
XTrain = []
YTrain = []
i = 0
for image in imagesTrain:

    caracteristicas = procesamientoImagen(image)

    XTrain.append(caracteristicas)

    YTrain.append(labelsTraining[i])

    i = i + 1

f = len(XTrain)
lo = len(XTrain[0])


# Procesar y extraer características de las imágenes de testeo
XTest = []
YTest = []
i = 0
for image in imagesTest:

    caracteristicas = procesamientoImagen(image)
    XTest.append(caracteristicas)

    YTest.append(labelsTest[i])

    i = i + 1

f = len(XTrain)
lo = len(XTrain[0])


# Entrenamos el clasificador KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(XTrain, YTrain)

# Realizamos las predicciones en el conjunto de prueba
y_pred = knn.predict(XTest)

print(YTest)
print(y_pred)

exactitud = accuracy_score(YTest, y_pred)

with open('knn_model_fuzzy.pkl', 'wb') as file:
    pickle.dump(knn, file)

print(exactitud)



