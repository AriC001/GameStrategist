# Steam AI Game Strategist
## Sistema inteligente de predicción, búsqueda semántica y análisis estratégico de videojuegos

**Asignatura:** Inteligencia Artificial II  
**Carrera:** Licenciatura en Ciencias de la Computación  
**Universidad:** Universidad Nacional de Cuyo  
**Estudiante:**  Cohen Ari  
**Fecha:** 17/11/2026

---

## 1. Descripción del proyecto

El proyecto *Steam AI Game Strategist* propone el desarrollo de un sistema inteligente orientado a asistir a desarrolladores de videojuegos independientes en el análisis de sus proyectos y de su entorno competitivo.

La solución integrará distintas técnicas de Inteligencia Artificial para ofrecer tres capacidades principales: predicción de éxito comercial mediante modelos de Deep Learning, recuperación de videojuegos similares mediante búsqueda semántica y estructurada, y un agente conversacional capaz de utilizar estas funcionalidades para elaborar análisis y recomendaciones.

El sistema estará organizado en tres módulos interconectados:

1. **Predictive Intelligence:** módulo de predicción que utilizará modelos de aprendizaje automático y Deep Learning para estimar la probabilidad de que un videojuego alcance la categoría de éxito comercial (*hit*).
2. **Semantic Game Intelligence:** módulo de recuperación de información que permitirá encontrar videojuegos similares a partir de consultas en lenguaje natural, combinando búsqueda vectorial y filtros estructurados.
3. **AI Strategic Consultant:** agente inteligente basado en un modelo de lenguaje (LLM), encargado de integrar las funcionalidades anteriores mediante el uso de herramientas (*Function Calling*) y proporcionar análisis comparativos de videojuegos.

El objetivo final es desarrollar una herramienta que combine capacidades predictivas, recuperación contextual de información e interacción mediante lenguaje natural.

## 2. Planteamiento del problema

El desarrollo y lanzamiento de un videojuego implica tomar numerosas decisiones relacionadas con su género, características, estrategia de comercialización y posicionamiento frente a otros títulos del mercado.

Para los desarrolladores independientes, estas decisiones pueden resultar particularmente complejas debido a la limitada disponibilidad de recursos y a la dificultad de evaluar anticipadamente el potencial comercial de sus proyectos.

Si bien existen plataformas que proporcionan información sobre videojuegos, sus características y desempeño comercial, transformar esos datos en información útil para la toma de decisiones requiere combinar diferentes mecanismos de análisis.

En este contexto, se plantea desarrollar un sistema que permita:

- Estimar el potencial de éxito comercial de un videojuego a partir de sus características.
- Identificar videojuegos con características similares para utilizarlos como referencias comparativas.
- Recuperar información relevante mediante consultas expresadas en lenguaje natural.
- Integrar los resultados de los modelos predictivos y los mecanismos de búsqueda mediante un agente inteligente.

El problema central consiste, por lo tanto, en investigar cómo combinar técnicas de aprendizaje automático, representaciones vectoriales y modelos de lenguaje para construir un sistema de asistencia estratégica basado en datos.

## 3. Objetivos

### 3.1. Objetivo general

Desarrollar un sistema inteligente que integre técnicas de Deep Learning, búsqueda semántica y agentes basados en modelos de lenguaje para analizar características de videojuegos, estimar su potencial de éxito comercial y proporcionar información comparativa que facilite la toma de decisiones estratégicas.

### 3.2. Objetivos específicos

- Diseñar e implementar un modelo de clasificación basado en redes neuronales multicapa (MLP), utilizando representaciones aprendidas para variables categóricas mediante *Entity Embeddings*.
- Experimentar con técnicas de normalización y regularización, particularmente LayerNorm y Dropout, para mejorar el entrenamiento y la capacidad de generalización del modelo.
- Comparar el desempeño de la arquitectura neuronal propuesta con modelos tradicionales de aprendizaje automático.
- Incorporar técnicas de explicabilidad para analizar los factores que influyen en las predicciones y evaluar las probabilidades estimadas.
- Implementar un mecanismo de generación de embeddings y almacenamiento vectorial para representar las características de los videojuegos.
- Desarrollar un sistema de búsqueda semántica que permita recuperar videojuegos relevantes a partir de consultas en lenguaje natural.
- Integrar mecanismos de recuperación estructurada y búsqueda vectorial mediante una estrategia de ranking híbrido.
- Desarrollar un agente conversacional basado en un LLM que utilice herramientas externas mediante *Function Calling*.
- Integrar los módulos predictivo y de recuperación para permitir el análisis comparativo de videojuegos mediante consultas conversacionales.

## 4. Alcance y módulos del sistema

### 4.1. Módulo 1: Predictive Intelligence

Este módulo tendrá como finalidad estimar la probabilidad de que un videojuego alcance la categoría de *hit*, definida a partir de su desempeño en términos de ingresos.

Se utilizará un conjunto de datos tabular que contiene información sobre distintos videojuegos, incluyendo variables como género, año de lanzamiento, tamaño del equipo, precio, cantidad de listas de deseos previas al lanzamiento y visualizaciones de tráileres.

Se implementará una arquitectura neuronal multicapa (MLP). Para representar las variables categóricas se emplearán *Entity Embeddings*, mientras que las variables numéricas serán normalizadas durante el preprocesamiento.

La arquitectura incorporará capas de normalización LayerNorm y mecanismos de regularización Dropout.

Además, se realizará una comparación experimental entre el modelo neuronal desarrollado y diferentes algoritmos tradicionales de aprendizaje automático, con el propósito de evaluar las ventajas y limitaciones de cada enfoque.

Se incorporarán técnicas de explicabilidad para identificar la influencia de las distintas características en las predicciones y analizar la confiabilidad de las probabilidades estimadas.

### 4.2. Módulo 2: Semantic Game Intelligence

Este módulo permitirá realizar búsquedas y recuperar videojuegos relevantes a partir de consultas expresadas en lenguaje natural.

Para ello, los metadatos de cada videojuego serán transformados en representaciones vectoriales mediante modelos de embeddings. Estas representaciones serán almacenadas en una base de datos vectorial que permitirá realizar búsquedas por similitud semántica.

El sistema también contará con mecanismos de recuperación estructurada para aplicar condiciones específicas sobre atributos como género, año de lanzamiento, precio y otras características cuantificables.

Se combinarán ambos mecanismos mediante una estrategia de ranking híbrido, con el objetivo de obtener resultados que contemplen tanto la similitud semántica como el cumplimiento de criterios específicos.

De esta manera, será posible realizar consultas como:

*"Buscar videojuegos independientes de características similares, lanzados después de 2020 y con indicadores elevados de interés previo al lanzamiento."*

### 4.3. Módulo 3: AI Strategic Consultant

Este módulo consistirá en un agente conversacional basado en un modelo de lenguaje (LLM), cuya función será interpretar las consultas de los usuarios y utilizar las herramientas proporcionadas por los módulos anteriores.

Mediante mecanismos de *Function Calling*, el agente podrá invocar funcionalidades específicas para:

- Estimar la probabilidad de éxito de un videojuego mediante el modelo predictivo.
- Buscar videojuegos similares mediante el sistema de recuperación semántica.
- Aplicar filtros estructurados sobre el conjunto de datos.
- Recuperar información de videojuegos relevantes para realizar análisis comparativos.

A partir de los resultados obtenidos, el agente podrá elaborar respuestas contextualizadas y fundamentadas en la información recuperada y en las predicciones del modelo.

El propósito de este módulo será integrar las capacidades desarrolladas previamente en una interfaz conversacional que facilite su utilización.

## 5. Datos de entrada al modelo

El sistema utilizará inicialmente un conjunto de datos tabular de videojuegos compuesto por aproximadamente 8.000 registros y 22 columnas, almacenado en formato CSV.

Cada registro representa un videojuego e incluye características descriptivas, información comercial y métricas relacionadas con su desempeño.

### 5.1. Variables del modelo predictivo

Para el módulo Predictive Intelligence se utilizarán variables categóricas, numéricas y binarias que describan las características de un videojuego.

Las variables de entrada previstas son:

| Variable | Tipo | Descripción |
|---|---|---|
| `genre` | Categórica | Género principal del videojuego (12 categorías). |
| `release_year` | Numérica | Año de lanzamiento. |
| `is_indie` | Binaria | Indica si el juego es independiente. |
| `team_size` | Numérica | Cantidad de desarrolladores. |
| `price_usd` | Numérica | Precio de lanzamiento en dólares. |
| `has_multiplayer` | Binaria | Indica si posee multijugador. |
| `early_access` | Binaria | Indica si fue lanzado en acceso anticipado. |
| `controller_support` | Binaria | Indica si posee soporte completo para controles. |
| `languages_supported` | Numérica | Cantidad de idiomas disponibles. |
| `dlc_count` | Numérica | Cantidad de contenidos descargables. |
| `file_size_gb` | Numérica | Tamaño de instalación en GB. |
| `wishlists_k` | Numérica | Cantidad de listas de deseos previas al lanzamiento, en miles. |
| `trailer_views_k` | Numérica | Visualizaciones de tráileres, en miles. |

Se excluirán las variables que representen resultados posteriores al lanzamiento, como `num_reviews`, `owners_k`, `peak_concurrent_players`, `revenue_musd` y `steam_positive_pct`, para evitar fuga de información (*data leakage*).

También se excluirá `game_id`, por tratarse de un identificador, y `title`, debido a que corresponde a un título sintético y no constituye una característica predictiva definida para el modelo.

La variable objetivo será `hit`, una etiqueta binaria que indica si el videojuego pertenece al 20% superior de ingresos según `revenue_musd`.

### 5.2. Preprocesamiento y representación

Las variables categóricas serán transformadas mediante *Entity Embeddings*, permitiendo que la red neuronal aprenda representaciones densas durante el entrenamiento.

Las variables numéricas serán normalizadas mediante `StandardScaler`, mientras que las variables binarias serán conservadas como indicadores numéricos.

El conjunto de datos será dividido en subconjuntos de entrenamiento, validación y prueba, procurando mantener la proporción de las clases mediante estratificación. El preprocesamiento será ajustado exclusivamente con los datos de entrenamiento.

### 5.3. Datos para la búsqueda semántica

El módulo Semantic Game Intelligence utilizará los metadatos descriptivos de los videojuegos para construir documentos estructurados que serán transformados en embeddings mediante un modelo de representación textual.

Los documentos incluirán información como género, año de lanzamiento, características del equipo, precio y métricas disponibles. Los metadatos originales se conservarán para permitir filtros estructurados y mostrar los resultados recuperados.

Los embeddings serán almacenados en una base de datos vectorial y se utilizarán para recuperar videojuegos mediante similitud semántica.

### 5.4. Datos de entrada del agente

El agente recibirá consultas expresadas en lenguaje natural por parte del usuario. Estas consultas podrán contener características de un videojuego, criterios de búsqueda o solicitudes de análisis comparativo.

Según la consulta, el agente utilizará las herramientas de predicción y recuperación de información para obtener resultados y elaborar una respuesta contextualizada.

### 5.5. Fuente y disponibilidad

Se utilizará inicialmente el conjunto de datos CSV de 8.000 registros mencionado en la propuesta. Su procedencia, licencia de uso y enlace de descarga serán documentados en la versión definitiva del anteproyecto.

Debido a que el conjunto de datos posee características sintéticas, los resultados experimentales serán interpretados dentro de las limitaciones de dicha fuente.

## 6. Tipo de sistema a implementar

Se desarrollará un sistema híbrido de Inteligencia Artificial compuesto por modelos de aprendizaje automático supervisado, mecanismos de representación y recuperación vectorial de información, y un agente conversacional basado en modelos de lenguaje.

El sistema combinará diferentes paradigmas y técnicas de IA de acuerdo con las responsabilidades de cada módulo.

### 6.1. Aprendizaje supervisado: clasificación binaria

El módulo Predictive Intelligence implementará un problema de clasificación binaria supervisada, cuyo objetivo será estimar si un videojuego pertenece o no a la categoría de éxito comercial definida por la variable `hit`.

Se utilizará una red neuronal multicapa (MLP) implementada en PyTorch, compuesta por:

- Capas de *Embedding* para representar la variable categórica `genre`.
- Capas lineales para procesar las características numéricas y binarias.
- Capas de normalización LayerNorm.
- Funciones de activación GELU o ReLU.
- Mecanismos de regularización Dropout.
- Una capa de salida que producirá un logit para la clasificación binaria.

El entrenamiento utilizará una función de pérdida `BCEWithLogitsLoss`.

Como referencia experimental se implementarán modelos tradicionales, particularmente Regresión Logística y Random Forest, para comparar el desempeño de la arquitectura neuronal propuesta.

### 6.2. Representación vectorial y recuperación de información

El módulo Semantic Game Intelligence utilizará modelos de embeddings para transformar las descripciones de los videojuegos en vectores numéricos de dimensión determinada por el modelo seleccionado.

Se aplicará un enfoque de recuperación híbrida que combinará:

- Búsqueda vectorial mediante similitud semántica.
- Consultas estructuradas sobre los metadatos de los videojuegos.
- Un mecanismo de ranking para combinar y ordenar los resultados obtenidos.

Este módulo no requerirá entrenar una red neuronal propia, sino utilizar modelos de representación textual y mecanismos de indexación y recuperación.

### 6.3. Agente inteligente basado en LLM

El módulo AI Strategic Consultant utilizará un modelo de lenguaje con capacidad de invocación de herramientas (*Function Calling*).

El agente interpretará las solicitudes del usuario y decidirá qué herramientas utilizar para obtener información del sistema.

Entre las herramientas disponibles se encontrarán las funciones de predicción, búsqueda semántica y filtrado estructurado.

El LLM será responsable de interpretar las consultas, coordinar las herramientas y generar respuestas fundamentadas en los resultados obtenidos.

### 6.4. Justificación de la arquitectura

La combinación de estos enfoques permitirá abordar diferentes aspectos del problema mediante técnicas especializadas.

El aprendizaje supervisado permitirá construir un modelo predictivo evaluable cuantitativamente; los embeddings y la recuperación híbrida permitirán encontrar información relevante en el conjunto de videojuegos; y el agente conversacional facilitará la interacción con las funcionalidades desarrolladas.

De esta manera, se propone una arquitectura modular que integra modelos predictivos y generativos, evitando depender de un único modelo para resolver todas las tareas.

## 7 Diagrama de arquitectura
El siguiente diagrama representa el flujo completo. Las dimensiones indicadas son simbólicas y se terminarán de determinar al implementar el modelo, ya que dependen de la cantidad de características seleccionadas y de los hiperparámetros.

![alt text](mermaid-diagram.png)

### 7.2. Explicación del flujo
El flujo comienza con la adquisición del dataset y su análisis exploratorio. Luego se seleccionan las variables pertinentes, se realiza el preprocesamiento y se generan las particiones de entrenamiento, validación y prueba.

A partir de estos datos se desarrollan dos procesos independientes:
- El módulo predictivo procesa las variables tabulares, aprende representaciones categóricas mediante Entity Embeddings y entrena una red neuronal multicapa. Su desempeño se compara con modelos tradicionales.
- El módulo semántico transforma los metadatos de los videojuegos en embeddings, los almacena en una base vectorial y construye un índice para consultas estructuradas. Ambos mecanismos alimentan el recuperador híbrido.
Una vez disponibles los modelos y los índices, el agente conversacional recibe consultas del usuario e invoca las herramientas correspondientes. Los resultados obtenidos son integrados por el LLM para generar una respuesta que puede incluir predicciones y análisis comparativos de videojuegos.

## 8. Métricas y metodología de evaluación

La evaluación del sistema se realizará de manera modular, utilizando métricas específicas para cada componente y pruebas de integración para verificar el funcionamiento conjunto.

### 8.1. Evaluación del modelo predictivo

Se evaluará el desempeño del clasificador mediante las siguientes métricas:

- **Accuracy:** proporción de predicciones correctas sobre el total de ejemplos.
- **Precision:** proporción de predicciones positivas que corresponden efectivamente a videojuegos clasificados como *hit*.
- **Recall:** proporción de videojuegos *hit* correctamente identificados.
- **F1-Score:** media armónica entre Precision y Recall.
- **ROC-AUC:** capacidad del modelo para distinguir entre ambas clases a diferentes umbrales.
- **PR-AUC:** evaluación del equilibrio entre Precision y Recall, especialmente relevante para la clase positiva.
- **Matriz de confusión:** análisis de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.

Se realizará una comparación entre Regresión Logística, Random Forest y la arquitectura MLP propuesta.

La comparación utilizará las mismas particiones de entrenamiento, validación y prueba. Los hiperparámetros serán ajustados mediante el conjunto de validación, reservando el conjunto de prueba para la evaluación final.

También se analizará la calibración de las probabilidades mediante métricas como Brier Score, para determinar si las probabilidades estimadas representan adecuadamente la frecuencia observada de la clase positiva.

### 8.2. Evaluación de explicabilidad

Se utilizarán técnicas de interpretabilidad, como SHAP, para identificar las características que tienen mayor influencia en las predicciones.

Se analizará la importancia global de las variables y su contribución a predicciones individuales, con el objetivo de facilitar la interpretación de los resultados.

### 8.3. Evaluación del módulo de búsqueda semántica

Se evaluará la calidad de recuperación mediante consultas de prueba previamente definidas y resultados relevantes identificados como referencia.

Se utilizarán métricas de recuperación de información:

- **Precision@K:** proporción de resultados relevantes entre los primeros K recuperados.
- **Recall@K:** proporción de los resultados relevantes que aparecen entre los primeros K recuperados.
- **NDCG@K:** calidad del ordenamiento de los resultados, considerando su relevancia.

Se comparará la búsqueda vectorial, la recuperación estructurada y el mecanismo híbrido para analizar las diferencias de rendimiento y relevancia.

### 8.4. Evaluación del agente conversacional

Se realizarán pruebas con consultas representativas para comprobar:

- La correcta interpretación de las solicitudes.
- La selección de herramientas apropiadas.
- La correcta ejecución y utilización de los resultados devueltos por las herramientas.
- La coherencia y fundamentación de las respuestas generadas.
- La capacidad de reconocer consultas que no pueden resolverse con la información disponible.

Se verificará que las respuestas relacionadas con predicciones y videojuegos recuperados correspondan a resultados efectivamente proporcionados por los módulos especializados.

### 8.5. Evaluación de integración

Finalmente, se realizarán pruebas de extremo a extremo utilizando escenarios de consulta que involucren uno o varios módulos.

Se analizará el correcto intercambio de información entre componentes, la consistencia de los resultados y la capacidad del sistema para generar respuestas útiles a partir de datos recuperados y predicciones.

## 10. Consideraciones éticas y uso responsable

El desarrollo de Steam AI Game Strategist contempla distintas consideraciones relacionadas con la calidad de los datos, la confiabilidad de las predicciones, la privacidad y el uso responsable de los modelos de Inteligencia Artificial.

### 10.1. Sesgos en los datos y las predicciones

El conjunto de datos utilizado presenta características sintéticas, por lo que sus distribuciones y relaciones entre variables podrían no representar adecuadamente las condiciones del mercado real de videojuegos.

Además, el modelo predictivo podría presentar diferencias de desempeño entre distintos géneros, años de lanzamiento o categorías de videojuegos independientes y no independientes.

Para analizar estas posibles diferencias se evaluarán las métricas de clasificación de manera desagregada por subgrupos, siempre que la cantidad de registros disponibles permita obtener resultados estadísticamente interpretables.

Se evitará interpretar las relaciones aprendidas por el modelo como relaciones causales o recomendaciones comerciales garantizadas.

### 10.2. Privacidad y protección de datos

El conjunto de datos inicial no requiere incorporar información personal de desarrolladores ni usuarios.

Las consultas realizadas al agente estarán orientadas a características de videojuegos y no requerirán información personal identificable.

En caso de incorporar fuentes externas o funcionalidades adicionales, se evaluará la procedencia de los datos, sus condiciones de utilización y las medidas necesarias para proteger información potencialmente sensible.

### 10.3. Riesgos de uso indebido y limitaciones

Las predicciones generadas por el sistema podrían ser interpretadas erróneamente como garantías de éxito comercial.

Para evitarlo, las respuestas deberán presentar las predicciones como estimaciones experimentales y explicitar las limitaciones del modelo y de los datos utilizados.

El agente conversacional también podría generar información incorrecta o interpretaciones no respaldadas por los resultados recuperados. Por ello, se priorizará la utilización de herramientas especializadas y la fundamentación de las respuestas en información efectivamente disponible.

El sistema no estará diseñado para reemplazar el criterio de los desarrolladores ni para garantizar decisiones comerciales.

### 10.4. Uso de herramientas de Inteligencia Artificial generativa

Durante el desarrollo del proyecto se contempla la utilización de herramientas de IA generativa como asistentes para tareas de investigación, consulta técnica, generación y revisión de código, y asistencia en la redacción y organización de la documentación.

Su utilización será complementaria al trabajo. Las decisiones de diseño, implementación, experimentación y validación serán revisadas y verificadas.

## 11. Plan de trabajo y riesgos

### 11.1. Cronograma de actividades

Se propone organizar el desarrollo en 6 semanas de trabajo, distribuidas en etapas incrementales.

| Etapa | Actividad | Período estimado |
|---|---|---|
| 1 | Análisis del dataset, exploración y preprocesamiento | Semana 1 |
| 2 | Implementación de modelos tradicionales de referencia | Semana 1 |
| 3 | Implementación, entrenamiento y evaluación del MLP | Semanas 2 |
| 4 | Implementación de embeddings y base de datos vectorial | Semana 3 |
| 5 | Desarrollo de búsqueda semántica y recuperación híbrida | Semana 4 |
| 6 | Implementación del agente y Function Calling | Semana 5 |
| 7 | Integración, pruebas, evaluación y documentación final | Semana 6 |

Las etapas podrán solaparse parcialmente cuando las dependencias técnicas lo permitan.

### 11.2. Identificación de riesgos y planes de contingencia

| Riesgo | Impacto | Plan alternativo |
|---|---|---|
| Dataset sintético poco representativo | Limita la validez de las conclusiones | Delimitar las conclusiones al entorno experimental y, si es viable, contrastar con datos reales. |
| Cantidad insuficiente de registros | Puede afectar la generalización del modelo | Simplificar la arquitectura, aplicar validación cruzada y utilizar regularización. |
| Falta de recursos de GPU | Puede aumentar los tiempos de entrenamiento | Utilizar CPU, reducir el tamaño de la red y optimizar los experimentos. |
| El MLP no supera a los modelos tradicionales | Reduce la ventaja práctica del Deep Learning | Mantener la comparación como resultado experimental y analizar las diferencias. |
| Problemas de calidad en los embeddings | Recuperación semántica poco relevante | Probar otros modelos de embeddings y ajustar la representación de documentos. |
| Dificultades de integración del LLM | Fallos en la invocación de herramientas | Implementar un flujo controlado con funciones determinísticas y reducir la complejidad del agente. |
| Respuestas incorrectas del LLM | Información no fundamentada | Validar las salidas de las herramientas y limitar las respuestas a la información disponible. |
| Retrasos en el desarrollo | Riesgo de no completar la integración | Priorizar el predictor, la búsqueda y la integración básica del agente antes de incorporar mejoras adicionales. |

El desarrollo priorizará la implementación funcional de los tres módulos principales. Las funcionalidades complementarias podrán reducirse en caso de restricciones de tiempo o recursos, sin comprometer los objetivos centrales del proyecto.

## 12. Referencias 

### 12.1. Herramientas y tecnologías

- PyTorch: https://pytorch.org/
- Scikit-learn: https://scikit-learn.org/
- ChromaDB: https://docs.trychroma.com/
- FAISS: https://github.com/facebookresearch/faiss

### 12.2. Conjunto de datos

- Dataset de videojuegos utilizado en el proyecto: [What Makes a Steam Game a Hit? 8,000 Games](https://www.kaggle.com/datasets/vedauv/what-makes-a-steam-game-a-hit-8000-games)