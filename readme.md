# MNIST Web Digit Recognizer (CNN)

Este proyecto me ha servido para aprender a desarrollar una red neuronal que usando TensorFlow, entrenada en una máquina de Google Colab, pueda predecir un número.

Aparte le he diseñado una interfaz web “agradable” que se comunica con el modelo mediante una API.

Este README sirve para demostrar (sobre todo a mí mismo) que no he copiado y pegado de una IA, sino que he entendido cada paso requerido para llevar a cabo el proyecto. Obviamente se explicará cómo se ha entrenado el modelo, el porqué según lo que yo —una mente inexperta— puedo deducir, y cómo se ha montado la interfaz web, que también tiene su gracia.

---

# queeee..?? CNN (Convolutional Neural Network)

### Qué son
Redes neuronales diseñadas para procesar imágenes o datos con estructura espacial.

### Para qué sirven
Detectan patrones como bordes, formas y texturas para clasificación, detección o segmentación de imágenes. Y luego, con todo eso, comparan su “resumen” interno y se hacen una idea bastante decente de qué número puede ser (en este caso).

> sobra decir que no hemos descrito ni un 1% de lo que es porque esto tiene mil variantes y se puede usar para muchas cosas… pero bueno, con esto tiramos por ahora.

**¿Por qué una CNN y no otra cosa?**
- Porque sin CNN no hacierta ni a la de 3 y para hacer que un modelo entienda una imagen que son numeros con vecinos lo mejor es una forma de entrenar el modelo que sea capaz de reconocer patrones.


**Alternativas que podría haber usado (y por qué no)**
- **TensorFlow.js en el navegador**: habría sido muy guapo (todo local, sin backend), pero:
  - en clase no lo hemos visto así :P.
  - complica bastante el setup
  - no quería pelearme con compatibilidades y exportaciones al principio
  - SOY NOVATO, tenía que empezar por el inicio.
- **Modelos enormes / transfer learning**: Bueno casi ni lo he investigado porque me parece matar moscas a cañonazos.

<br><br>

# MNIST03

- Es un dataset de números escritos a mano del 0 al 9.

### que es un DATASET
- un conjunto de datos para entrenamiento :P.

### que es un MODELO
- es algo que aprende de ejemplos (como de un dataset :P) para dar respuesta.

### entonces un modelo MNIST03
- es algo que ha aprendido de esos datos para dar una respuesta (y normalmente lo hace bastante bien).

**Por qué MNIST (y no otro dataset)**
- Porque es el “hola mundo” de visión por ordenador y porque de eso iba esta clase xd.
- Además, como el objetivo era acabar con una app web, MNIST es perfecto para montar todo el pipeline sin morir en el intento.

**Qué limitación tiene MNIST (y por qué me importa)**
- MNIST es “limpio”: muchos dígitos están centrados, con contraste decente, etc.
- Un canvas de navegador no es tan limpio: el trazo puede ser gordo, fino, torcido, y encima el usuario dibuja como le da la gana.
- Por eso meto augmentation y me tomo en serio el preprocesado, porque si no, el modelo acierta en MNIST pero en mi web se vuelve medio tonto.

<br><br>

# ENTRENAMIENTO MNIST03

- ### que es entrenar un modelo
- entrenar un músculo = más fuerza, entrenar un modelo = más acierto (o al menos, esa es la idea).

- ### ¿por qué colab? 

- Entrenar un modelo consume recursos, y Colab te da una máquina decente (a veces con GPU) gratis.
- Al ser por celdas, puedo probar cambios sin ejecutar todo el notebook cada vez.

- ### Alternativas y por qué no
- Mi PC: sin GPU es más lento y además te comes la instalación/configuración y la nube de pago para esto es pasarse; Colab va sobrado.


- ### ¿COMO ENTRENARLO?
- al cuaderno de colab le subes un archivo `NOMBRE.ipynb` que tiene formato JSON y genera una serie de celdas que puedes ejecutar independientemente.
- al final te devolverá un modelo entrenado que ya podrás usar para tus menesteres (guardar, cargar, predecir, etc).

---

## EXPLICACIÓN DEL ENTRENAMIENTO PASO POR PASO

- ### librerías
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import logging
```

- #### tensorflow
lo que vamos a usar para entrenar a la IA, el gimnasio.

- #### tensorflow_datasets
facilita la gestión de datasets, las pesas.

- #### matplotlib.pyplot
para dibujar gráficos.

- #### numpy
para operaciones matemáticas (y porque al final siempre acaba saliendo).

- #### logging
para mostrar mensajes de estado y depuración.

---

- ### LOGGER
```python
logger = tf.get_logger()
# el logger solo muestra errores
logger.setLevel(logging.ERROR)
```

---

- ### CARGAR DATASET
Aquí ya empieza el lío. Estamos cargando desde `tensorflow_datasets` el dataset como entrenamiento y como test. Esto lo indicamos con `split=['train','test']`. Mezcla los archivos con `shuffle_files=True`. Con `as_supervised=True` indica que los archivos tienen que tener imagen y label obligatoriamente. En `ds_info` se guardan metadatos del dataset y esto es por usar `with_info=True`.

```python
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
```
---

- ### NORMALIZAR DATASET
Normalizar significa convertir los píxeles (0–255) a `float32` y escalarlos a un rango cómodo para la red: **0–1**.  
Así, `0 -> 0.0` (negro) y `255 -> 1.0` (blanco). Esto ayuda a que el entrenamiento sea más estable y a que el optimizador no vaya “a trompicones” por tener valores muy grandes.

Además añadimos un canal con `tf.expand_dims(images, axis=-1)` porque aunque MNIST venga como `(28, 28)`, una CNN normalmente espera **(alto, ancho, canales)**, o sea `(28, 28, 1)` en blanco y negro.

Devolvemos las imágenes ya adaptadas junto con sus etiquetas (labels), porque la etiqueta no se toca: sigue siendo el número correcto (0–9).

```python
def normalize(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = tf.expand_dims(images, axis=-1)
    return images, labels
```

Aquí aplicamos con `map()` la función `normalize` a cada elemento del dataset.  
Y con `num_parallel_calls=tf.data.AUTOTUNE`  ejecutamos varios en paralelo y con AUTOTUNE le decimos a tensorFlow que decida el cuantos.

```python
train_dataset = ds_train.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = ds_test.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
```
---

- ### DATA AUGMENTATION
Con `data_augmentation` estamos creando datos modificando los que ya teníamos, generando variaciones de ellos (rotados, desplazados, con zoom), para que los números no tengan que ser perfectos, sino más “humanos”, más caóticos.

Usamos `tf.keras.Sequential` para transformar una imagen detrás de otra, no solo una cosa.

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),          # ~ +/- 14 grados
    tf.keras.layers.RandomTranslation(0.10, 0.10), # desplazar hasta 10%
    tf.keras.layers.RandomZoom(0.10),              # zoom leve
], name='data_augmentation')
```

- ### MODELO CNN “MEJORADO” 

Aquí ya montamos la CNN de verdad. La idea es simple: **entrar una imagen 28×28**, sacar **probabilidades 0–9**.

---

#### Input + augmentation dentro del modelo
```python
inputs = tf.keras.layers.Input(shape=(28, 28, 1))

# Augmentation solo afecta durante training
x = data_augmentation(inputs)
```

**Por qué esto así**
- `Input(shape=(28,28,1))`: con esto dices al modelo que la forma de los inputs es de 28x28 y blanco y negro.

- **porque el `data_augmentation` aquí?** bueno pues fué por que no sabía muy bien que pasaría si lo metia en el map así que hice el map para normalizar y ahora que se está definiendo el entrenamiento pues retuerzo los números... alomejor baja la optimización pero asi no moldeamos todo el dataset.
---

## Bloque 1 (primeros “detectores”)
```python
x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.20)(x)
```

**Qué hace**
- `Conv2D(32, 3)`: crea 32 “filtros” 3×3 que van aprendiendo cosas básicas (bordes, curvas simples…).
- `padding='same'`: mantiene el tamaño (no quiero perder info demasiado pronto).
- `use_bias=False`: porque con BatchNorm el bias suele sobrar (menos parámetros inútiles).
- `BatchNormalization`: estabiliza el entrenamiento, hace que sea menos “random” y suele permitir que aprenda mejor.
- `relu`: mete no-linealidad, si no, sería un modelo muy limitado.
- `MaxPooling2D`: reduce tamaño (28→14), se queda con lo importante y hace al modelo más tolerante a pequeños movimientos.
- `Dropout(0.20)`: apaga neuronas al azar para que no memorice demasiado (anti-overfitting).

**Por qué dos conv seguidas**
- En vez de hacer una conv enorme, metes dos pequeñas y el modelo puede aprender combinaciones más finas con menos coste.

---

## Bloque 2 (patrones más complejos)
```python
x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.25)(x)
```

**Qué cambia respecto al bloque 1**
- Subo a **64 filtros** porque aquí ya no busco “rayas sueltas”, busco **partes de números** (como el “gancho” del 2 o el círculo del 0).
- El resto es el mismo patrón: conv + BN + ReLU, pooling y dropout.

**Por qué va subiendo dropout**
- Cuanto más profundo, más capacidad → más riesgo de memorizar → subo un poco el dropout.

---

## Bloque 3 (resumen final de features)
```python
x = tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.30)(x)
```

**Qué hace**
- 128 filtros para capturar cosas más “globales”.
- Otro pooling (14→7 aprox. desde el bloque anterior), o sea: cada vez vamos de detalle a resumen.
- Dropout sube otra vez para que no se confíe.

**Alternativa que podría haber usado**
- Meter otro bloque más o más filtros, pero para MNIST es fácil pasarte: ganas poco y te arriesgas a sobreajustar o a tardar más entrenando.

---

## Clasificador (convertir “mapas” en decisión 0–9)
```python
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.35)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
```

**Qué hace**
- `Flatten()`: convierte los mapas 2D en un vector para poder clasificar.
- `Dense(256, relu)`: capa que mezcla toda la info y aprende combinaciones finales.
- `Dropout(0.35)`: aquí es donde más fácil es memorizar, porque es denso y tiene mucha libertad → le meto más dropout.
- `Dense(10, softmax)`: me devuelve 10 probabilidades (una por cada número).

**Alternativa típica (y por qué no la usé aquí)**
- En vez de `Flatten`, usar `GlobalAveragePooling2D()` que reduce parámetros y a veces generaliza mejor.  
  No lo usé porque para MNIST esto ya iba bien y quería mantenerlo “clásico” y entendible.

---

## Construcción del modelo
```python
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist_cnn_aug_bn')
model.summary()
```

**Por qué `tf.keras.Model` y no `Sequential`**
- Porque ya estoy conectando cosas como “inputs → augmentation → bloques → outputs” y me gusta verlo claro (además, si luego cambio algo, la API funcional es más flexible).

`model.summary()` lo uso para comprobar:
- shapes entre capas (si algo está mal, aquí canta)
- número de parámetros (para no hacer un monstruo sin querer)