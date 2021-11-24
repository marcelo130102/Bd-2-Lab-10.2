# Base de datos lab 10.2
## Integrantes
* Marcelo Surco
* Miguel Alvarado

## Contenido
* [Pregunta 1](#p1:-funcionamiento-de-la-carga-de-imágenes)

# Pregunta 1
### Funcionamiento de la carga de imágenes
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
# P2: Adaptación del Rtree de python
La adaptación del código es la siguiente:
```python
def Rtree_face_recognition(file, mode = 'RGB'):
```

# P3: Implementación de la búsqueda por rango y la KNN

## Comparación de resultados
