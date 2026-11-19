# Experimento 2 — Semantic Game Intelligence: buscador híbrido de videojuegos en lenguaje natural

**Documentación del proceso experimental · Módulo de búsqueda semántica, filtros y ranking**

## Resumen ejecutivo

El segundo experimento aborda un problema de **recuperación y ranking de videojuegos a partir de consultas en lenguaje natural**. A diferencia del Experimento 1, que clasificaba juegos según su éxito comercial (`hit`), este módulo busca recuperar y ordenar juegos de acuerdo con las condiciones y preferencias expresadas por el usuario.

Se implementó un sistema compuesto por tres elementos:

- Un intérprete de consultas basado en reglas, sin utilizar un LLM.
- Representaciones semánticas de los juegos mediante embeddings de `intfloat/multilingual-e5-base`.
- Un mecanismo de recuperación y ranking que combina filtros obligatorios, similitud semántica, preferencias y una puntuación estructural.

El sistema se evaluó con un benchmark de 20 consultas y cuatro estrategias de ranking: estructurada, semántica, estática e híbrida.

La estrategia híbrida obtuvo el mejor resultado promedio, con **NDCG@10 = 0,800**, seguida de la estática (0,792), la estructurada (0,713) y la semántica (0,305). Sin embargo, la diferencia entre la estrategia híbrida y la estática fue pequeña (+0,008), por lo que los resultados no muestran una mejora generalizada importante por adaptar dinámicamente los pesos.

El análisis también identificó limitaciones en la representación semántica, la construcción del benchmark y la implementación de algunas funciones del parser. Las conclusiones se circunscriben al dataset sintético utilizado y no deben extrapolarse directamente a un catálogo real de Steam.

## 1. Introducción

### 1.1 Problema

El objetivo es recibir una consulta en lenguaje natural —por ejemplo, «juegos baratos, preferentemente independientes» o «similares a Iron Signal, pero más baratos»— y devolver un ranking de videojuegos acompañado de una explicación de los criterios utilizados.

Un sistema de búsqueda de este tipo debe resolver dos clases de necesidades:

- **Condiciones objetivas:** precio máximo, género específico, ausencia de acceso anticipado o soporte para controles.
- **Criterios de similitud y preferencias:** juegos parecidos a otro, económicos, populares o bien valorados.

Los filtros estructurados permiten aplicar condiciones objetivas con precisión, pero no representan por sí solos las nociones de similitud. Los embeddings permiten comparar representaciones semánticas, aunque no garantizan el cumplimiento de condiciones obligatorias. La hipótesis de trabajo fue que una arquitectura híbrida podía combinar ambas capacidades.

### 1.2 Diferencias con el Experimento 1

| Característica | Experimento 1 | Experimento 2 |
|---|---|---|
| Tarea | Clasificación binaria (`hit`) | Recuperación y ranking de juegos |
| Enfoque | Modelos entrenados (MLP, regresión logística y Random Forest) | Reglas y embeddings preentrenados |
| Resultado | Probabilidad de pertenecer a la clase `hit` | Lista ordenada de juegos |
| Métricas | Precision, Recall, F1, ROC-AUC y Average Precision | P@k, R@k y NDCG@k |
| Uso de `hit` | Variable objetivo | No se utiliza como entrada ni como etiqueta de relevancia |

## 2. Dataset y auditoría

### 2.1 Características generales

Se utilizó el mismo dataset sintético del Experimento 1, `steam_game_hits.csv`.

| Característica | Valor |
|---|---:|
| Juegos | 8.000 |
| Columnas | 22 |
| Géneros | 12 |
| Títulos únicos | 484 |
| Títulos que aparecen una sola vez | 0 |
| Apariciones de «Iron Signal» | 19 |

### 2.2 Hallazgos y decisiones sobre los datos

La auditoría mostró que ningún título aparece una sola vez. Por lo tanto, el título no permite identificar de manera unívoca un videojuego: **la identidad de cada registro está dada por `game_id`**. Esta característica condicionó el tratamiento de las consultas que hacen referencia a un juego concreto.

Se excluyeron `revenue_musd` y `hit` para evitar que el buscador utilizara información asociada al resultado comercial que se había estudiado en el primer experimento. En particular, `hit` no se utilizó como etiqueta de relevancia.

Los documentos semánticos se construyeron únicamente con `title`, `genre`, `is_indie`, `has_multiplayer`, `early_access` y `controller_support`. No se incorporaron descripciones narrativas inventadas.

También se calcularon percentiles de variables numéricas para traducir expresiones cualitativas —como «barato», «bien valorado» o «popular»— en condiciones cuantificables. Estos umbrales funcionan como referencias para la interpretación de consultas y no como reglas universales.

Las variables de popularidad posteriores al lanzamiento, como `owners_k`, `num_reviews` y `peak_concurrent_players`, se utilizaron en la puntuación estructural, no como parte de la representación semántica general.

## 3. Representación semántica

### 3.1 Construcción de documentos

Cada videojuego se transformó en un documento textual mediante una plantilla que concatena el título, el género y cuatro atributos booleanos expresados en lenguaje natural.

La representación incluye información como el título del juego, su género, si es independiente, si cuenta con multijugador, si está en acceso anticipado y si ofrece compatibilidad con controles.

No se agregaron descripciones sobre mecánicas, ambientación o experiencia de juego que no estuvieran respaldadas por los datos disponibles.

### 3.2 Alineación entre juegos y embeddings

La tabla `base` se ordenó para que cada fila, documento y embedding compartieran la misma posición. La implementación incluye una verificación explícita de esa correspondencia y un diccionario que relaciona `game_id` con la posición de su embedding.

Esta alineación corrige un error de desplazamiento presente en una versión anterior del notebook y evita asociar un vector semántico con un juego incorrecto.

### 3.3 Colapso semántico

La comparación de documentos reveló una limitación de la representación:

| Medición | Cantidad |
|---|---:|
| Documentos totales | 8.000 |
| Documentos únicos con título | 7.302 |
| Documentos únicos sin título | 192 |

Los 192 documentos distintos sin título corresponden a las combinaciones posibles de 12 géneros y cuatro atributos booleanos:

\[
12 \times 2^4 = 192
\]

Por lo tanto, cuando se omite el título, juegos con el mismo género y los mismos atributos booleanos generan documentos idénticos. En esos casos, el embedding no dispone de información para distinguirlos. La puntuación estructural se utiliza como señal adicional para ordenar candidatos con representaciones semánticas iguales o muy similares.

### 3.4 Modelo de embeddings

| Parámetro | Valor |
|---|---|
| Modelo | `intfloat/multilingual-e5-base` |
| Dimensión | 768 |
| Normalización | Sí |
| Prefijo de documentos | `passage:` |
| Prefijo de consultas | `query:` |
| `max_seq_length` | 256 |
| Batch size | 128 |
| Archivos generados | `embeddings_e5_base.npy`, `embedding_game_ids.npy`, `embedding_config.json` |

La generación de los embeddings de los 8.000 documentos tomó aproximadamente 7 segundos y produjo un conjunto de vectores de unos 23 MB.

Como los embeddings están normalizados, el producto punto entre vectores equivale a la similitud coseno.

El principal límite observado no está en la dimensión o el tamaño del modelo, sino en la información utilizada para construir los documentos. Un modelo de embeddings más grande no permitiría recuperar distinciones que no están presentes en el texto de entrada. Para enriquecer esta representación serían necesarias descripciones, etiquetas o reseñas reales.

### 3.5 ChromaDB

Se creó una colección persistente llamada `videojuegos_e5`, configurada con distancia coseno y cargada en lotes de 1.000 registros. Los identificadores se almacenaron como cadenas de texto.

En la implementación evaluada, ChromaDB no participa directamente en el ranking: las búsquedas calculan la similitud mediante el producto punto entre los embeddings almacenados en memoria. Para un conjunto de 8.000 juegos, este procedimiento resultó suficiente y el tiempo promedio de búsqueda se mantuvo alrededor de 0,13 segundos.

## 4. Intérprete de consultas

### 4.1 Enfoque

La consulta se procesa mediante un parser basado en reglas y expresiones regulares, sin utilizar un LLM. Esta decisión prioriza el determinismo, la trazabilidad y la posibilidad de auditar cómo se interpreta cada expresión.

La principal limitación de este enfoque es que las reglas deben contemplar explícitamente las distintas formas de expresar una misma intención.

### 4.2 Información extraída

El parser construye una estructura `Intencion` que puede contener:

| Campo | Función |
|---|---|
| `generos` | Géneros solicitados como condición de filtro |
| `genero_semantico` | Géneros mencionados en un contexto comparativo |
| `atributos_semanticos` | Atributos booleanos mencionados en modo comparativo |
| `obligatorios` | Condiciones que los candidatos deben cumplir |
| `preferencias` | Condiciones blandas utilizadas en el ranking |
| `titulo` | Título de referencia detectado |
| `advertencias` | Interpretaciones que pueden requerir aclaración |

### 4.3 Condiciones obligatorias y preferencias

El parser distingue entre condiciones que deben cumplirse y preferencias que influyen en el orden de los resultados.

Expresiones como «prefiero», «idealmente», «de ser posible» y «preferentemente» se interpretan como preferencias. En cambio, expresiones como «solo», «exclusivamente», «debe», «necesito», «sin», «con», «menos de», «hasta» y «entre» se interpretan como condiciones obligatorias, de acuerdo con el contexto local de la consulta.

Las condiciones obligatorias se aplican antes del ranking. Las preferencias se incorporan como una señal de puntuación.

### 4.4 Interpretación de expresiones cualitativas

Los términos cualitativos se traducen a umbrales basados en los percentiles del dataset.

| Expresión | Variable | Condición |
|---|---|---|
| Muy barato / regalado | `price_usd` | ≤ P10 |
| Barato / económico / asequible | `price_usd` | ≤ P25 |
| Caro / premium | `price_usd` | ≥ P75 |
| Muy bien valorado / excelentes valoraciones | `steam_positive_pct` | ≥ P90 |
| Bien valorado / buenas valoraciones | `steam_positive_pct` | ≥ P75 |
| Gran cantidad de reseñas | `num_reviews` | ≥ P90 |
| Bastantes reseñas | `num_reviews` | ≥ P75 |
| Muy popular / los más jugados | `peak_concurrent_players` | ≥ P90 |
| Popular / muchos jugadores | `peak_concurrent_players` | ≥ P75 |
| Pequeño / liviano / poco espacio | `file_size_gb` | ≤ P25 |

También se interpretan condiciones numéricas con unidades, como «menos de 15 dólares», «hasta 10 GB», «a partir de 2022» y «entre X e Y», además de condiciones booleanas negadas, como «sin multijugador» o «sin acceso anticipado».

Cuando se solicita una valoración, el parser agrega una preferencia de cantidad de reseñas (`num_reviews ≥ P25`) con peso 0,5 para contextualizar el porcentaje positivo. De esta manera, una valoración alta con muy pocas reseñas no se trata igual que otra respaldada por un volumen mayor de opiniones.

### 4.5 Consultas comparativas y títulos ambiguos

En expresiones como «parecido a», «similar a» o «del mismo estilo», los géneros mencionados se interpretan como referencias semánticas y no necesariamente como filtros obligatorios. Del mismo modo, expresiones como «aunque no» o «sin importar» permiten reconocer atributos que no deben convertirse en condiciones excluyentes.

La ambigüedad de los títulos se resuelve utilizando `game_id`. Si un título corresponde a varios registros, el sistema debe solicitar una selección o información adicional —por ejemplo, género, año o precio— en lugar de asumir que existe una única referencia.

### 4.6 Correcciones incorporadas al parser

La versión documentada del notebook incorpora las siguientes correcciones:

| Cambio | Detalle |
|---|---|
| Negación de acceso anticipado | Regla específica para `early_access = 0` |
| Expresión «a partir de» | Interpretación como condición de mínimo (`>=`) |
| Cualitativas de reseñas y concurrencia | Reglas con P75/P90 para `num_reviews` y `peak_concurrent_players` |
| «Exclusivamente» | Incorporación entre las marcas de condición obligatoria |
| Falso patrón `car\w*` | Reemplazo por `\bcar[oa]s?\b` para evitar confundir «características» con «caro» |

Se verificaron especialmente las consultas Q03, Q07, Q10, Q13, Q14, Q18, Q19 y Q20.

La extracción de `genero_semantico` y `atributos_semanticos` se verificó para las consultas comparativas. Sin embargo, esos campos no se utilizan directamente en el ranking de la versión evaluada, por lo que su extracción no implica que tengan un efecto efectivo sobre la recuperación.

### 4.7 Consultas del benchmark

| ID | Consulta | Tipo |
|---|---|---|
| Q01 | Busco exclusivamente juegos RPG. | Filtro de género |
| Q02 | Quiero juegos que tengan multijugador. | Booleano obligatorio |
| Q03 | Juegos que no tengan acceso anticipado. | Negación |
| Q04 | Busco juegos que cuesten menos de 15 dólares. | Numérica |
| Q05 | Juegos de menos de 10 GB que tengan soporte para controles. | Numérica + booleano |
| Q06 | Busco juegos baratos, preferentemente independientes. | Cualitativa + preferencia |
| Q07 | Quiero juegos con muy buenas valoraciones en Steam. | Cualitativa |
| Q08 | Prefiero juegos con muchos jugadores concurrentes. | Cualitativa |
| Q09 | Busco juegos pequeños, que ocupen poco espacio, y con buenas valoraciones. | Cualitativas combinadas |
| Q10 | Quiero juegos económicos, pero priorizando aquellos con muchas reseñas. | Cualitativa + prioridad |
| Q11 | Busco juegos parecidos a los RPG, pero estoy abierto a otros géneros. | Semántica (género) |
| Q12 | Quiero juegos similares a Iron Signal. | Referencia a título |
| Q13 | Busco videojuegos independientes con características similares a los juegos de estrategia. | Semántica + filtro |
| Q14 | Quiero juegos parecidos a los multijugador, aunque no necesariamente tengan ese modo. | Semántica (atributo) |
| Q15 | Busco juegos del mismo estilo que Iron Signal, pero que sean más baratos. | Referencia + preferencia |
| Q16 | Busco RPG multijugador por menos de 20 dólares. | Combinada |
| Q17 | Quiero juegos independientes, baratos y con buenas valoraciones. | Combinada |
| Q18 | Busco juegos populares que no ocupen demasiado espacio. | Combinada |
| Q19 | Quiero juegos lanzados a partir de 2022, preferentemente económicos y con soporte para controles. | Combinada |
| Q20 | Busco juegos con muchas reseñas y una gran cantidad de jugadores concurrentes, sin importar el género. | Combinada |

## 5. Filtros, preferencias y ranking

### 5.1 Generación de candidatos

El procedimiento comienza interpretando la consulta y generando el conjunto de candidatos.

- **Estrategia semántica:** utiliza los 8.000 juegos, sin aplicar los filtros estructurados detectados en la consulta.
- **Estrategias estructurada, estática e híbrida:** aplican primero los filtros obligatorios y utilizan únicamente los registros que los cumplen.

Cuando la consulta contiene una referencia a un juego, se utiliza el embedding de ese juego como vector de consulta. En los demás casos, se genera un embedding a partir del texto de la consulta.

### 5.2 Componentes del ranking

La puntuación final combina tres componentes:

\[
S(g,q)=
\frac{
w_{sem}S_{sem}(g,q)+
w_{pref}S_{pref}(g,q)+
w_{base}S_{base}(g,q)
}{
w_{sem}+w_{pref}+w_{base}
}
\]

Donde:

- \(S_{sem}\): similitud semántica entre la consulta y el juego.
- \(S_{pref}\): cumplimiento de las preferencias detectadas.
- \(S_{base}\): puntuación estructural calculada a partir de atributos del juego.
- \(w_{sem}\), \(w_{pref}\) y \(w_{base}\): pesos de cada componente.

Las condiciones obligatorias no forman parte de esta fórmula: se aplican previamente para determinar qué juegos pueden participar del ranking.

### 5.3 Estrategias comparadas

Las cuatro estrategias comparten la misma estructura de puntuación, pero utilizan diferentes pesos.

| Estrategia | Peso semántico | Peso preferencias | Peso base |
|---|---:|---:|---:|
| Estructurada | 0 | 2,0 si hay preferencias; 0 en caso contrario | 0,5 |
| Semántica | 1,0 | 0 | 0,001 |
| Híbrida | 1,0 o 2,0 | Entre 1,0 y 3,0, según las preferencias | 0,5 |
| Estática | 1,0 | 1,0 si hay preferencias; 0 en caso contrario | 0,5 |

En la estrategia híbrida, el peso semántico aumenta a 2,0 cuando la consulta contiene una referencia a un juego y permanece en 1,0 para consultas generales. El peso de preferencias se calcula según la cantidad de preferencias detectadas:

\[
w_{pref}=\min(1+0,5\,n_{pref},3)
\]

Si no hay preferencias, su peso es cero.

La estrategia estática mantiene pesos fijos para los componentes semántico y de preferencias. La híbrida, en cambio, adapta esos pesos al contenido de la consulta.

La estrategia semántica asigna un peso de 0,001 al componente base, por lo que la similitud semántica domina el ranking y la señal estructural tiene una influencia prácticamente nula.

La estrategia estructurada asigna peso cero a la similitud semántica. No obstante, conserva el componente base con peso 0,5: los filtros determinan los candidatos y la puntuación estructural contribuye a ordenarlos.

### 5.4 Satisfacción de preferencias

Las preferencias se evalúan mediante una función de satisfacción. Cuando un valor cumple el umbral solicitado, obtiene una puntuación de 1,0; no se recompensa adicionalmente cuánto supera ese umbral.

| Operador | Comportamiento |
|---|---|
| `=` | 1 si coincide; 0 si no |
| `<=` / `<` | 1 hasta el umbral; luego decae linealmente hasta 0 en un tope definido por la variable |
| `>=` / `>` | 1 desde el umbral; por debajo, puntuación proporcional (`x / umbral`) |
| `between` | 1 dentro del rango; decae según la distancia relativa al ancho del intervalo |
| Valor faltante | 0,5 en modo neutral o 0 en modo penalizar |

### 5.5 Ajuste de la preferencia de precio

En una versión inicial, la recompensa lineal para consultas de precio bajo concentraba los primeros resultados en juegos gratuitos. Aunque esos juegos cumplen literalmente una condición de precio bajo, esa concentración no necesariamente representa la intención de quien busca juegos económicos.

Para evitarlo, se estableció una banda de preferencia entre P10 y P25 (USD 2,99–4,99). Los precios inferiores a P10 reciben una satisfacción de 0,90, en lugar de la recompensa máxima.

El ajuste se revisó en Q06 y Q17. Las métricas P@10 y NDCG@10 no cambiaron, aunque la inspección de los resultados mostró una distribución de precios más acorde con la intención de encontrar juegos económicos y no solamente gratuitos.

### 5.6 Puntuación estructural

El componente `score_base` combina dos grupos de variables:

- **Popularidad:** `owners_k`, `num_reviews` y `peak_concurrent_players`.
- **Calidad:** `steam_positive_pct` y `metacritic_score`.

Cada grupo aporta la mitad de la puntuación estructural. Los valores se calculan en relación con los candidatos de la consulta. Las variables que ya se utilizan en la consulta se excluyen de este componente para evitar contabilizarlas dos veces.

Esta señal permite ordenar juegos que tienen representaciones semánticas iguales o muy similares, una situación frecuente debido al colapso semántico identificado en la sección 3.3.

### 5.7 Explicaciones de los resultados

Cada resultado puede incluir un campo `por_que`, construido a partir de las reglas aplicadas y las puntuaciones calculadas. Por ejemplo, puede indicar que un juego cumple una condición de precio determinada y mostrar la satisfacción asociada.

La explicación tiene un alcance limitado: no describe en profundidad por qué dos juegos son semánticamente similares y puede quedar vacía en consultas puramente semánticas.

## 6. Diseño del benchmark

### 6.1 Construcción del conjunto de relevancia

Se construyó un benchmark de 20 consultas con un conjunto de relevancia (*gold*) completo para cada una. Cada juego recibe un grado de relevancia de 0, 1 o 2.

Los grados se asignaron mediante máscaras sobre los datos y umbrales basados en percentiles. En términos generales, el grado 2 representa un criterio más estricto y el grado 1 uno más amplio.

Las consultas Q11, Q13 y Q14 incorporan alternativas semánticas. Por ejemplo, para una consulta sobre juegos parecidos a RPG, también pueden considerarse relevantes juegos de Strategy o Adventure con buenas valoraciones.

Las consultas Q12 y Q15 utilizan `G00001` como referencia explícita para «Iron Signal», debido a que el título corresponde a varios registros. El juego de referencia se excluye del conjunto de relevancia.

El gold fue construido mediante reglas, no mediante juicios manuales independientes. Esta característica debe tenerse en cuenta al interpretar las métricas.

### 6.2 Estrategias evaluadas

| Estrategia | Candidatos | Configuración principal |
|---|---|---|
| Estructurada | Juegos que cumplen los filtros | Sin similitud semántica; preferencias con peso 2,0 cuando existen; base 0,5 |
| Semántica | Los 8.000 juegos, sin filtros | Similitud con peso 1,0; preferencias 0; base 0,001 |
| Híbrida | Juegos que cumplen los filtros | Pesos semánticos y de preferencias adaptados a la consulta; base 0,5 |
| Estática | Juegos que cumplen los filtros | Semántica 1,0; preferencias 1,0 cuando existen; base 0,5 |

### 6.3 Métricas

Se calcularon las siguientes métricas para \(k=5\) y \(k=10\):

- **Precision@k (P@k):** proporción de resultados recuperados que son relevantes.
- **Recall@k (R@k):** proporción del conjunto relevante que aparece entre los primeros k resultados.
- **NDCG@k:** calidad del ordenamiento, considerando tanto la relevancia como la posición de cada resultado.

También se registraron la cantidad de candidatos y el tiempo de ejecución por consulta. Para Precision y Recall se considera relevante todo juego cuyo grado en el gold sea mayor que cero.

### 6.4 Tamaño del conjunto de relevancia

Los conjuntos de relevancia tienen tamaños muy diferentes: van de 14 a 4.752 juegos.

Esta variación afecta especialmente a Recall@10. Su valor máximo teórico depende del tamaño del gold: con 4.752 juegos relevantes, incluso recuperar diez relevantes produce un recall muy bajo. Por ese motivo, Recall@10 se presenta como métrica complementaria y no como criterio principal para comparar estrategias.

Las consultas con conjuntos de relevancia pequeños también pueden producir resultados más sensibles a cambios en unos pocos elementos.

## 7. Resultados del benchmark

### 7.1 Métricas agregadas

La siguiente tabla presenta los promedios de las 20 consultas.

| Estrategia | P@5 | P@10 | R@5 | R@10 | NDCG@5 | NDCG@10 | Candidatos promedio | Tiempo promedio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Estática | 0,890 | 0,870 | 0,009 | 0,013 | 0,816 | 0,792 | 6.326,35 | 0,134 s |
| Estructurada | 0,820 | 0,815 | 0,004 | 0,009 | 0,714 | 0,713 | 6.326,35 | 0,134 s |
| **Híbrida** | **0,910** | **0,880** | 0,013 | 0,017 | **0,825** | **0,800** | 6.326,35 | **0,132 s** |
| Semántica | 0,470 | 0,415 | **0,022** | **0,035** | 0,334 | 0,305 | 8.000,00 | 0,140 s |

La estrategia híbrida obtuvo los mejores valores de Precision y NDCG. La semántica alcanzó los valores más altos de Recall, pero esto no implica un mejor ranking: sus resultados de Precision y NDCG fueron considerablemente inferiores.

La cantidad promedio de candidatos es igual para las estrategias que aplican filtros, mientras que la estrategia semántica evalúa el catálogo completo.

Los tiempos corresponden a la ejecución registrada en el benchmark y muestran diferencias pequeñas entre estrategias.

### 7.2 Comparación de NDCG@10 por consulta

La tabla permite observar la variabilidad que queda oculta al considerar solamente los promedios.

| Consulta | Estática | Estructurada | Híbrida | Semántica |
|---|---:|---:|---:|---:|
| Q01 | 1,000 | 1,000 | 1,000 | 0,405 |
| Q02 | 1,000 | 1,000 | 1,000 | 0,059 |
| Q03 | 1,000 | 1,000 | 1,000 | 0,507 |
| Q04 | 0,512 | 0,390 | 0,512 | 0,162 |
| Q05 | 0,638 | 0,588 | 0,638 | 0,219 |
| Q06 | 1,000 | 1,000 | 1,000 | 0,628 |
| Q07 | 1,000 | 1,000 | 1,000 | 0,166 |
| Q08 | 1,000 | 1,000 | 1,000 | 0,120 |
| Q09 | 0,958 | 1,000 | 1,000 | 0,069 |
| Q10 | 1,000 | 1,000 | 1,000 | 0,000 |
| Q11 | 0,471 | 0,037 | 0,471 | 0,208 |
| Q12 | 0,359 | 0,000 | 0,387 | 1,000 |
| Q13 | 0,640 | 0,050 | 0,640 | 1,000 |
| Q14 | 0,501 | 0,650 | 0,501 | 0,498 |
| Q15 | 0,230 | 0,000 | 0,319 | 0,643 |
| Q16 | 0,522 | 0,537 | 0,522 | 0,097 |
| Q17 | 1,000 | 1,000 | 1,000 | 0,110 |
| Q18 | 1,000 | 1,000 | 1,000 | 0,000 |
| Q19 | 1,000 | 1,000 | 1,000 | 0,118 |
| Q20 | 1,000 | 1,000 | 1,000 | 0,098 |

La mejora promedio de NDCG@10 de la híbrida frente a la estructurada es de **+0,087**. Esta diferencia se concentra principalmente en Q11, Q12, Q13 y Q15. En esas consultas, la similitud semántica aporta información que los filtros estructurados no pueden representar por sí solos.

También hay diferencias en otras consultas, como Q04, Q05, Q14 y Q16, pero sus efectos positivos y negativos se compensan en gran medida. Por lo tanto, la mejora global no es uniforme.

### 7.3 Comparación entre la estrategia híbrida y la estática

La diferencia promedio entre la estrategia híbrida y la estática es de apenas **+0,008 NDCG@10**.

Las diferencias positivas más relevantes aparecen en:

| Consulta | Estática | Híbrida | Diferencia |
|---|---:|---:|---:|
| Q09 | 0,958 | 1,000 | +0,042 |
| Q12 | 0,359 | 0,387 | +0,028 |
| Q15 | 0,230 | 0,319 | +0,089 |

En el resto de las consultas, las diferencias son nulas o se compensan. Esto indica que la estrategia estática ya captura gran parte del beneficio de incorporar similitud semántica y preferencias. La adaptación dinámica de los pesos aporta una mejora promedio pequeña en este benchmark.

### 7.4 Casos en los que la búsqueda semántica obtiene mejores resultados

La estrategia semántica supera a la híbrida en algunas consultas concretas:

| Consulta | Semántica | Híbrida | Diferencia a favor de la semántica |
|---|---:|---:|---:|
| Q12 | 1,000 | 0,387 | 0,613 |
| Q13 | 1,000 | 0,640 | 0,360 |
| Q15 | 0,643 | 0,319 | 0,324 |

En Q12 y Q15 se utiliza un juego de referencia. En Q13 se busca similitud con juegos de estrategia, combinada con una condición sobre juegos independientes.

Estos resultados muestran que la búsqueda semántica puede ser especialmente efectiva en determinadas consultas comparativas. Sin embargo, su rendimiento global es bajo porque no aplica los filtros obligatorios detectados en las consultas y depende de la información limitada de los documentos semánticos.

### 7.5 Distribución de NDCG@10

Se calcularon estadísticas descriptivas sobre los resultados de las 20 consultas.

| Estrategia | Media | Mediana | Desviación estándar | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|
| Estática | 0,7916 | 1,0000 | 0,2707 | 0,2298 | 1,0000 |
| Estructurada | 0,7126 | 1,0000 | 0,4011 | 0,0000 | 1,0000 |
| Híbrida | **0,7996** | 1,0000 | **0,2607** | **0,3188** | 1,0000 |
| Semántica | 0,3053 | 0,1643 | 0,3099 | 0,0000 | 1,0000 |

La estrategia híbrida presenta la mayor media y la menor desviación estándar entre las estrategias estructurada, estática e híbrida. También tiene el mayor valor mínimo, lo que sugiere un comportamiento más consistente dentro de este conjunto de consultas.

La mediana de 1,0 en las tres estrategias que aplican filtros indica que al menos la mitad de las consultas obtuvo el máximo NDCG@10. Sin embargo, las diferencias entre medias, desviaciones estándar y mínimos muestran que el comportamiento de las consultas restantes no es uniforme.

La estrategia estructurada presenta la mayor variabilidad, con una desviación estándar de 0,4011 y un mínimo de cero. La estrategia semántica tiene una mediana de 0,1643, lo que refleja que sus buenos resultados se concentran en un grupo reducido de consultas.

### 7.6 Limitaciones de la evaluación

Los resultados deben interpretarse teniendo en cuenta las características del benchmark:

1. **Efecto techo.** En 14 de las 20 consultas, al menos una estrategia alcanza NDCG@10 = 1,0. En 11 consultas, las tres estrategias que aplican filtros —estructurada, estática e híbrida— alcanzan ese valor simultáneamente. Esto reduce la capacidad del benchmark para distinguir entre estrategias en esas consultas.

2. **Recall depende del tamaño del gold.** Los conjuntos relevantes varían entre 14 y 4.752 juegos. Por esa razón, Recall@10 no es directamente comparable entre consultas y puede favorecer en el promedio a las consultas con gold pequeño. La semántica obtiene el mayor Recall@10 promedio (0,035), a pesar de tener los peores resultados de P@10 y NDCG@10.

3. **Gold construido mediante reglas.** Los grados de relevancia se asignaron mediante máscaras y umbrales, no mediante evaluaciones humanas independientes. Además, algunos criterios del gold son más estrictos que los umbrales utilizados por el parser. Por ejemplo, en Q01–Q03 el grado 2 exige condiciones de calidad que no aparecen explícitamente en las consultas. Esto puede generar resultados perfectos sin que el benchmark discrimine suficientemente la calidad del ranking.

4. **La métrica no refleja todos los ajustes de diseño.** La modificación de la preferencia de precio mejoró la distribución perceptible de los resultados de Q06 y Q17, pero no produjo cambios en P@10 ni NDCG@10. El gold considera igualmente relevantes distintos juegos que cumplen el umbral de precio, por lo que no captura esa diferencia de calidad.

En consecuencia, P@10 y NDCG@10 son las métricas más útiles para describir los resultados de este benchmark. Aun así, deben interpretarse junto con el análisis por consulta y las limitaciones del gold.

## 8. Pruebas de robustez y análisis de la implementación

### 8.1 Datos faltantes

Se evaluó el efecto de ocultar aleatoriamente el 10 % de los valores de `steam_positive_pct`, utilizando `default_rng(7)`. Se compararon dos tratamientos: considerar los valores faltantes como neutrales (0,5) o penalizarlos (0,0).

Ambos modos obtuvieron el mismo resultado: **NDCG@10 = 0,808**. El valor es ligeramente superior al resultado de 0,800 obtenido en el benchmark sin esa perturbación, pero esto no significa que introducir valores faltantes mejore el sistema. Indica que, en esta prueba, la métrica no mostró sensibilidad a la diferencia entre ambos tratamientos.

La prueba cubrió una variable utilizada como preferencia. No evaluó el comportamiento ante valores faltantes en condiciones obligatorias. En la implementación, si un campo obligatorio está vacío, la comparación puede devolver `False` y excluir el juego.

### 8.2 Títulos ambiguos

El título «Iron Signal» corresponde a 19 juegos. Sus embeddings presentan similitudes muy altas, entre 0,98 y 0,999.

Elegir automáticamente uno de esos registros como referencia sería arbitrario. Por eso, el sistema requiere un `game_id` o información adicional para resolver la ambigüedad. Esta decisión respeta la identidad real de los registros en el dataset y evita asumir que un título identifica un único juego.

### 8.3 Limitaciones de la implementación

| Limitación | Descripción |
|---|---|
| ChromaDB no participa en el ranking | La colección se construyó, pero la búsqueda evaluada utiliza los embeddings almacenados en memoria. |
| Explicaciones incompletas | `explicar()` presenta géneros y reglas aplicadas, pero no explica en profundidad la similitud semántica ni el desempate. En algunas consultas puramente semánticas puede devolver una explicación vacía. |
| Atributos semánticos sin efecto directo | `genero_semantico` y `atributos_semanticos` se extraen, pero no se utilizan directamente en el ranking. |
| Peso semántico condicionado por el título | La implementación aumenta `w_sem` cuando hay una referencia a un título. Las consultas comparativas sin título no reciben ese aumento, aunque tengan una intención semántica. |
| Dependencia de reglas explícitas | El parser es determinista y auditable, pero requiere incorporar reglas para reconocer nuevas formas de expresión. |
| Variables de popularidad agrupadas | `owners_k`, `num_reviews` y `peak_concurrent_players` se agrupan en una dimensión de popularidad, sin que se haya medido su correlación en este experimento. |

Las estrategias estructurada, estática e híbrida utilizan `score_base` con peso 0,5, mientras que la estrategia semántica lo reduce a 0,001. Por ello, las diferencias entre estrategias no pueden atribuirse exclusivamente a la similitud semántica o a la adaptación dinámica de pesos: también interviene la puntuación estructural.

## 9. Conclusiones generales

### 9.1 Aprendizajes del experimento

1. **La estrategia híbrida obtuvo el mejor resultado global**, con NDCG@10 = 0,800, aunque su diferencia respecto de la estrategia estática fue pequeña (+0,008).

2. **La similitud semántica aporta valor en consultas comparativas.** La mejora de la híbrida frente a la estructurada fue de +0,087 NDCG@10 y se concentró principalmente en Q11, Q12, Q13 y Q15.

3. **La adaptación dinámica de pesos tuvo un beneficio promedio limitado.** La estrategia estática alcanzó NDCG@10 = 0,792, muy cerca de la híbrida. En este benchmark, gran parte de la mejora proviene de incorporar un componente semántico, más que de ajustar sus pesos dinámicamente.

4. **La búsqueda semántica también puede superar a la híbrida en casos concretos.** Esto ocurrió en Q12, Q13 y Q15. Por lo tanto, ninguna estrategia domina en todas las consultas.

5. **La representación semántica está limitada por la información disponible.** Los documentos contienen metadatos y no descripciones narrativas. Sin el título, solo existen 192 representaciones textuales distintas para los 8.000 juegos.

6. **La evaluación presenta un efecto techo y depende de un gold basado en reglas.** Los resultados deben leerse como una comparación dentro del benchmark construido, no como una medición definitiva de la calidad de un buscador en producción.

7. **La identidad de los juegos debe resolverse mediante `game_id`.** Los títulos repetidos, como «Iron Signal», no permiten seleccionar una referencia inequívoca.

### 9.2 Limitaciones generales

- El dataset es sintético. Sus títulos repetidos, precios y distribuciones no representan necesariamente un catálogo real de Steam.
- El benchmark contiene solo 20 consultas y el gold fue construido mediante reglas, no mediante juicios humanos independientes.
- Algunos conjuntos de relevancia son muy pequeños, mientras que otros contienen miles de juegos.
- Los documentos semánticos se construyen a partir de metadatos y no incluyen descripciones, etiquetas ni reseñas.
- El parser basado en reglas requiere mantenimiento para reconocer nuevas expresiones.
- La prueba de robustez ante valores faltantes cubrió únicamente una variable utilizada como preferencia.
- Las estrategias comparadas comparten algunos componentes de puntuación, por lo que sus diferencias no aíslan completamente el efecto de cada señal.

En conjunto, el experimento demuestra la viabilidad de combinar interpretación de consultas, filtros estructurados y embeddings para construir un buscador de videojuegos en lenguaje natural. Los resultados favorecen a la estrategia híbrida dentro del benchmark utilizado, pero también muestran que su ventaja frente a una estrategia estática es reducida y que la calidad de la recuperación depende fuertemente de la representación de los datos y de la construcción del conjunto de relevancia.

## Anexo A. Registro de experimentos

| ID | Experimento | Configuración | Criterio o métrica | Resultado |
|---|---|---|---|---|
| E2-01 | Auditoría del dataset | 8.000 juegos, 22 columnas | Unicidad de `game_id` y títulos | 484 títulos; ninguno aparece una sola vez |
| E2-02 | Documentos semánticos | Título, género y cuatro atributos booleanos | Documentos únicos | 7.302 con título; 192 sin título |
| E2-03 | Embeddings | `multilingual-e5-base`, 768 dimensiones, normalizados | Tiempo y tamaño | Aproximadamente 7 s y 23 MB |
| E2-04 | Parser por reglas | Obligatorios, preferencias y expresiones cualitativas | Verificación de 20 consultas | Correcciones en Q03, Q07, Q10, Q13, Q14, Q18, Q19 y Q20 |
| E2-05 | Preferencia de precio | Recompensa lineal frente a banda P10–P25 | Inspección de Q06 y Q17 | Sin cambios en métricas; distribución de precios más adecuada |
| E2-06 | Benchmark de cuatro estrategias | 20 consultas, gold 0/1/2 | P@k, R@k y NDCG@k | Híbrida: NDCG@10 = 0,800 |
| E2-07 | Análisis por consulta | Comparación pareada de estrategias | NDCG@10 | +0,087 híbrida vs. estructurada; +0,008 vs. estática |
| E2-08 | Robustez ante datos faltantes | 10 % de `steam_positive_pct` oculto | NDCG@10 | 0,808 en modo neutral y penalización |
| E2-09 | Títulos ambiguos | «Iron Signal», 19 registros | Similitud entre candidatos | Entre 0,98 y 0,999 |

## Anexo B. Reproducibilidad

| Elemento | Valor |
|---|---|
| Notebook | `semanticGameIntelligence2.ipynb` |
| Archivo de datos | `./data/steam_game_hits.csv` |
| Modelo de embeddings | `intfloat/multilingual-e5-base` |
| Dimensión y normalización | 768; normalizados |
| `max_seq_length` / batch size | 256 / 128 |
| Archivos generados | `embeddings_e5_base.npy`, `embedding_game_ids.npy`, `embedding_config.json` |
| ChromaDB | Colección `videojuegos_e5`, distancia coseno, lotes de 1.000 |
| Semilla de la prueba de datos faltantes | `np.random.default_rng(7)` |
