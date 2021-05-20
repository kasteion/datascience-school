# Curso Práctico de Regresión Lineal con Python

# Introducción a ML y los algoritmos

## Regresión lineal y machine learning

Curso enfocado en Machine Learning para que el modelo pueda ser entrenado y aprender utilizando un módelo de regresión lineal.

El proceso general del modelo machine learner sería:

1. Tener observaciones etiquetadas (Labeled observations)
2. Sería tener un set de entrenamiento (Datos que utilizamos para entrenar nuestro modelo)

- Tener un test set, un set de pruebas para validar si nuestro modelo se creó de forma eficiente o no.

3. Machine Learner. El modelo que agarra los datos de entrenamiento y aprende en base a ellos. Utilizando un modelo de predicción

4. Y luego con las estadisticas (stats) evaluamos la eficiencia de nuestro modelo porque puede estar over fitted o under fitted. Para reentrenar en caso de ser necesario.

**ML supervisado**

1. Regresión: Forma parte de la gran familia de machine learning supervisado. Nos sirve para agarrar datos de alguna variable respecto al tiempo y predecir el futuro. Nos puede servir para:

- Estimar el crecimiento de la población
- Predicción del clima
- Predicción del mercado

2. Clasificación: Nos puede servir para:

- Clasificar clientes, retención de clientes.
- Generar Diagnósticos
- Clasificación de imágenes

**ML supervisado - Algoritmos**

Resumen: - Algoritmo de regresión lineal - Regresión. - Algoritmo de regresión logística - Clasificación. - Algoritmo Naive Bayes - Clasificación. - K-nearest neighbors - Regresión y Clasificación. - Decision Tree - Regresión y Clasificación. - Random Forest - Regresión y Clasificación.

## Explicación matemática de la regresión lineal

La regresión lineal es un término estadistico que se conoce como un módelo matemático que nos ayuda a encontrar la relación entre una variable independiente y las variables dependientes que tiene un modelo.

Y = Constante + Pendiente(X)

# Entendiendo el algoritmos de regresión lineal

## Método de mínimos cuadrados: ecuación

Tengo una tabla con valores de X y Y (Nuestro Dataset)

Y tengo una grafica en que voy graficando los puntos de X,Y
X = [1, 2, 3, 4, 5]
y = [2, 3, 5, 6, 5]

Esta es la formula que debemos encontrar... Y = b0 + b1(X)

Esta sería la formula para encontrar la pendiente:

(Sumatoria de (x - mean(x))(y - mean(y))) / (Sumatoria de (x - mean(x))^2)

MEAN(X) = 3
MEAN(Y) = 4.2

X - MEAN(X) = [-2, -1, 0, 1, 2]
Y - MEAN(Y) = [-2.2, -1.2, 0.8, 1.8, 0.8]
(X - MEAN(X))^2 = [4, 1, 0, 1, 4]
(X - MEAN(X))(Y - MEAN(Y)) = [4.4, 1.2, 0, 1.8, 1.6]

SUM((X - MEAN(X))(Y - MEAN(Y))) = 9
SUM((X - MEAN(X))^2) = 10

Bsub1 = 9/10 = 0.9 => La pendiente es 0.9

Y = Bsub0 + 0.9(X)

Nuestra línea de regresión siempre debe pasar por (MEAN(X), MEAN(Y)) que sería (3, 4.2) es el único punto que conocemos de la linea de regresión. Entonces:

Y = Bsub0 + 0.9(X)
4.2 = Bsub0 + 0.9(3)
Bsub0 = 4.2 - 0.9(3)
Bsub0 = 1.5

Entonces la formula de la linea de regresión lineal sería:

Y = 1.5 + 0.9(X)

# Proyecto del curso

## Llevando nuestro algoritmo a Python

```python
import numpy as np

def estimate_b0_b1:
  n = np.size(x)
  # Obtenemos los promedios de X y de Y
  m_x, m_y = np.mean(x), np.mean(y)

  # Calcular sumatoria de XY y mi sumatoria de XX
  sum_xy = np.sum((x-m_x)*(y-m_y))
  sum_xx = np.sum(x*(x-m_x))

  # Coeficientes de regresión
  b_1 = sum_xy/sum_xx
  b_0 = m_y - (b_1 * m_x)

  return(b_0, b_1)
```

## Creando nuestra función de graficación

```python
import matplotlib.pyplot as plt

def plot_regression(x, y, b):
  plt.scatter(x, y, color = "g", marker="o", s=30)

  y_pred = b[0] + b[1]*x
  plt.plot(x, y_pred, color="b")

  # Etiquetado
  plt.xlabel('x-independiente')
  plt.ylabel('y-dependiente')

  plt.show()
```

## Código main y probando nuestro código

```python
# Codigo Main
def main():
  # Dataset
  x = np.array([1, 2, 3, 4, 5])
  y = np.array([2, 3, 5, 6, 5])

  # Obtenemos b1 y b2
  b = estimate_b0_b1(x, y)
  print("Los valores b_0 = {}, b_1 = {}".format(b[0], b[1]))

  # Graficamos nuestra linea de regresión
  plot_regression(x, y, b)

if __name__ == "__main__":
  main()
```

## Conclusiones

Esta en Google Colab...

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_b0_b1(x, y):
  n = np.size(x)
  # Obtenemos los promedios de X y de Y
  m_x, m_y = np.mean(x), np.mean(y)

  # Calcular sumatoria de XY y mi sumatoria de XX
  sum_xy = np.sum((x-m_x)*(y-m_y))
  sum_xx = np.sum(x*(x-m_x))

  # Coeficientes de regresión
  b_1 = sum_xy/sum_xx
  b_0 = m_y - (b_1 * m_x)

  return(b_0, b_1)

def plot_regression(x, y, b):
  plt.scatter(x, y, color = "g", marker="o", s=30)

  y_pred = b[0] + b[1]*x
  plt.plot(x, y_pred, color="b")

  # Etiquetado
  plt.xlabel('x-independiente')
  plt.ylabel('y-dependiente')

  plt.show()

# Codigo Main
def main():
  # Dataset
  x = np.array([1, 2, 3, 4, 5])
  y = np.array([2, 3, 5, 6, 5])

  # Obtenemos b1 y b2
  b = estimate_b0_b1(x, y)
  print("Los valores b_0 = {}, b_1 = {}".format(b[0], b[1]))

  # Graficamos nuestra linea de regresión
  plot_regression(x, y, b)

if __name__ == "__main__":
  main()
```
