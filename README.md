# Base de datos lab 10.2
## Integrantes
* Marcelo Surco
* Miguel Alvarado

## Contenido
* [Pregunta 1](#pregunta-1)
* [Pregunta 2](#pregunta-2)
* [Pregunta 3](#pregunta-3)

# Pregunta 1
## Funcionamiento de la carga de imágenes
Esta función de la librería `face_recognition` lo que hace en el caso por defecto, es convertir el `input` en un array de numpy, en otro caso se convierte primero la imagen para luego retornar el array.

```python
import PIL.Image
import numpy as np
from PIL import ImageFile

def load_image_file(file, mode = 'RGB'):

    im = PIL.Image.open(file)

    if mode:
        im = im.convert(mode)

    return np.array(im)
```

Por lo que el tamaño del vector característico sería al obtener el `length(np.array(im))` el cual sería de tamaño de 128.
# Pregunta 2
## Adaptación del Rtree de python
Generacion de los vectores caracteristicos de las imagenes dentro del dataset:

```python
import face_recognition
import os
file = open("vectores.csv", "w")
folder_path = "fotos_bd\lfw"
folders = os.listdir(folder_path)

for folder in folders:
    images = os.listdir(folder_path + '/' + folder)
    for image in images:
        path = folder_path + '/' + folder + '/' + image
        picture = face_recognition.load_image_file(path)
        vector = face_recognition.face_encodings(picture)
        try:
            line = ""
            for i in range(128-1):
                line+= str(vector[0][i]) + ","
            line+= str(vector[0][127]) + "\n"
            file.write(line)
        except:
            print("Fail")
file.close()
```
Estos vectores son guardados dentro de `vectores.csv`.

Se usa R para generar un histograma de  distribución  de  distancias  para  una 
muestra  aleatoria  de  pares  de  5000 puntos del dataset  de 
imágenes de rostros.

```R
library(readr)
library(dplyr)
data<-read_csv("vectores.csv")
```
```R
ED <- function(X, Y){
  return(sqrt(sum((X-Y)^2)))
}
```
```R
genDistancias <- function(data, maxiter){
  dists <- rep(0,maxiter)
  N<-nrow(data)
  for(i in 1:N){
    ind<- sample(1:N, size=2)
    P<-data[ind[1],]
    Q<-data[ind[2],]
    dists[i]<- ED(P,Q)
  }
  return(dists)
}

```

```R
D<-genDistancias(data[,1:128],5000)
Distancias <- hist(D, xlab = "Distancias", breaks = 100)
```
![alt text](https://github.com/marcelo130102/Bd-2-Lab-10.2/blob/master/histogramaPara5000ParesPuntos.png)


# Pregunta 3
## Implementación de la búsqueda por rango y la KNN

## Comparación de resultados
