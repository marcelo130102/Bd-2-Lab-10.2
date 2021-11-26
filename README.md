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
Esta función de la librería `face_recognition` lo que hace en el caso por defecto, es convertir el `input` en un array de numpy, en otro caso se convierte primero la imagen para luego retornar el array de 128 indices.

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

```python
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
Adaptación del Rtree de python

```python
from rtree import index

p = index.Property()
p.dimension = 128 #D
p.buffering_capacity = 10 #M
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index('128d_index',properties=p)
```
lectura de los vectores caracteristicos

```python
import csv

vectores = []
with open('vectores.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        line = [float(x) for x in line]
        vectores.append(line)
```

Insercion de los vectores
```python
for i in range(len(vectores)):
    idx.insert(i, tuple(vectores[i]))
```

los k=10 vecinos de Aaron_Eckhart_0001.jpg
```python
picture = face_recognition.load_image_file("fotos_bd\lfw\Aaron_Eckhart\Aaron_Eckhart_0001.jpg")
q = tuple(face_recognition.face_encodings(picture)[0])
lres = list(idx.nearest(coordinates=q, num_results=10))
print("El vecino mas cercano de Aaron_Eckhart: ", lres)
```
Los 10 vecinos mas cercano de Aaron_Eckhart:  [0, 8510, 5823, 1578, 3541, 10244, 1969, 10035, 8167, 8836]

Range_seach con radio 0.6 de Aaron_Eckhart_0001.jpg
```python
picture = face_recognition.load_image_file("fotos_bd\lfw\Aaron_Eckhart\Aaron_Eckhart_0001.jpg")
query = np.array(face_recognition.face_encodings(picture)[0])
result = []
nrows = len(vectores)
radio = 0.6
for i in range(nrows):
    dist = ED(np.array(vectores[i]), query)
    if(dist < radio):
        result.append(i)
print(result)
```
[0, 1130, 1578, 1969, 3541, 5823, 6884, 8167, 8510, 8836, 10035, 10244]

