# EmployeeTurnover

## Employee turnover: Predecir la probablidad de que un empleado renuncie usando Machine Learning

¿Cómo podemos  retener nuestro talento?

Es una pregunta que toda organización del mundo debería hacerse. El talento es el insumo de mayor valor para cada organización, dado que es el mayor generador de valor: No importan los recursos si no hay gente habilidosa, motivada detrás de ellos. 

Para poder retener el talento, es necesario entender la razón de la salida de los empleados: si se sabe quién está en riesgo de abandonar la organización, se pueden tomar medidas preventivas.

En el presente Jupyter notebook, se ilustrará el proceso para la obtenición de un modelo preliminar de Machine Learning para la predicción de la probabilidad de que un empleado abandone una organización. Para esto, se utilizará un dataser de Kaggle que contiene una serie de features asociadas a los empleados, además de una variables binaria que determina si el empleado abandonó o no la organización. 

A grandes rasgos, los pasos que se seguirán serán:
1. **Carga y validación de los datos:** Se verifica que los datos a usar sean integrales, que no haya faltantes, que sus valores sean coherentes con la realidad.
2. **Análisis Exploratorio:** Se exploran los datos para detectar posibles interacciones de interés. En el caso de dataset con muchos features, se usaría como base para empezar con los modelos.
3. **Preparación de los datos:** Se realiza preparación y limpieza de datos para los modelos (Normalizar, Estandarizar, Upsampling/Downsampling, generación de datos sintéticos, rellenado de faltantes, entre otras posibilidades). Adicionalmente, se separa un fragmento de los datos para posteriormente probar los modelos en ellos.
4. **Modelado inicial:** Se prueba con algunos modelos básicos. 
5. **Optimización de hiper parámetros**: Se aproxima a la mejor combinación de hiper parámetros del modelo seleccionado para los datos que se tienen. 
6. **Selección del mejor modelo** Con los datos separados en el punto 3, se prueban todos los modelos y se elige aquel con un mejor desempeño.

En el repositorio se encuentra:
1. data folder: Contiene los datos originales y los separados en el proceso para la prueba
2. models folder: Contiene los Pipelines de los GridSearchCV utilizados.
3. scripts folder: Contiene un .py con las funciones creadas en el notebook.
4. images folder: Contiene plots generados en el proceso
4. EmployeeTurnover.ipynb: Un JupyterNotebook con todo el ejercicio

## Resultados finales
### Comparación de modelos (Plot ROC-AUC) (usando los datos en "data/Test_data.csv":
![Plot ROC - AUC modelos](https://github.com/JECaballeroR/EmployeeTurnover/blob/main/images/ROC_AUC_final.png)
### Resultados del mejor modelo (sobre datos en "data/Test_data.csv")

              precision    recall  f1-score   support

           0       0.97      0.95      0.96      2255
           1       0.85      0.92      0.89       745

    accuracy                           0.94      3000
    macro avg       0.91      0.94      0.92      3000
    weighted avg       0.94      0.94      0.94      3000

![Matriz de confusión y curva ROC-AUC](https://github.com/JECaballeroR/EmployeeTurnover/blob/main/images/ConfMatr_ROC_AUC_Final_plot.png)
### Feature Importance del Random Forest final:
![Feature Importance plot](https://github.com/JECaballeroR/EmployeeTurnover/blob/main/images/Feature_Importance.png)


## To do: 
* Optimizar con una mayor variedad de hiper parámetros en el GridSearchCV (por ejemplo, probar el efecto del optimizador en la Red Neuronal)
* Migrar el KerasClassifier a [SciKeras](https://github.com/adriangb/scikeras), para poder ser exportado con joblib
* Probar otros modelos como SVM, otras redes neuronales, otros ensemble models
* Hacer pruebas del efecto de otros métodos para el manejo del inbalance en el dataset (podría incluirse en el pipeline del Grid Search)

## Conclusiones:

1. Para la predicción de la probablidad de que un empleado renuncie, para los datos usados, tienen mayor importancia:
    - satisfaction_level
    - time_spend_company
    - average_montly_hours
    - number_project
    - last_evaluation

    Dado eso, es de interés para la compañia recolectar esta información con mayor prioridad.

2. Todos los modelos probados tienen un desempeño relativamente aceptable, superior al 80%. Cualquiera es "suficientemente bueno".
    - El mejor modelo es el Random Forest. Al ser un modelo ensemble, no es de sorprender que se desempeñe mejor que otros algoritmos probados.
3. La metodología usada puede aplicarse a otros problemas de clasificación binaria, pero es importante aclarar:
    1. Para datasets con mas features, sería ideal aplicar algún método de selección de aquellos más relevantes (por ejemplo, PCA). El mapa de calor del Predictive Power Score puede ser un buen primer acercamiento.
    2. Otros datasets puede que requieran un mayor cuidado para su manejo (por ejemplo, ser limpiados o aplicar alguna transformación específica a los datos). Una opción para la realización de estar tareas es generalizarlas como funciones y agregarlas en el Pipeline
