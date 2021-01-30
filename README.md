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
3. EmployeeTurnover.ipynb: Un JupyterNotebook con todo el ejercicio
