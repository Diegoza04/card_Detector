# Memoria técnica — Detector de cartas (card_Detector)

Fecha: 2025-11-26  
Autor: Proyecto refactorizado (Diegoza04)

Índice
1. Resumen ejecutivo
2. Hardware: listado de elementos usados en el proyecto y justificación de uso
3. Software: características técnicas y requisitos. Justificación de uso.
4. Hoja de ruta del desarrollo
5. Solución:
   - Arquitectura y explicación
6. Diagrama de decisión para la clasificación de cartas
7. Secuencialización del pipeline de imagen (operaciones, parámetros y justificación)
8. Otras tareas realizadas
9. Código fuente documentado — organización de módulos y funciones clave
10. Cómo generar el PDF
11. Anexos / referencias

---

1) Resumen ejecutivo
--------------------
El proyecto "card_Detector" es un sistema para detectar y clasificar cartas (palo y valor) a partir de imágenes fijas y video en vivo (p. ej. DroidCam). Se ha refactorizado el código original para separar responsabilidades, mejorar mantenibilidad y velocidad en vivo sin cambiar la lógica de clasificación. Esta memoria recoge requisitos hardware/software, la hoja de ruta seguida, la solución técnica y una secuencia detallada de operaciones y parámetros utilizados.

Objetivos principales:
- Detectar cartas en imagen/video.
- Extraer y normalizar la región de carta (warp).
- Extraer símbolos (rank y suit) de la esquina superior izquierda.
- Clasificar rank (2..A) y suit (Corazones, Diamantes, Picas, Treboles) con heurísticas y matching por templates.
- Mantener/ mejorar FPS en detección en vivo a través de optimizaciones no invasivas (tracking por optical-flow, threading, cache de templates).

---

2) Hardware: listado de elementos usados en el proyecto y justificación de uso
-------------------------------------------------------------------------------

Elementos usados
- Fuente de luz: luz blanca (lampara/led)
  - Rol: proporcionar iluminación homogénea y estable sobre la carta para reducir sombras, reflejos y variaciones de exposición entre frames.
  - Justificación técnica: una iluminación blanca y constante mejora la consistencia de los histogramas de color, facilita la binarización por Otsu y la extracción de contornos, y reduce falsos negativos/positivos en la segmentación de símbolos (rank/suit). Además permite fijar la exposición del teléfono sin que la cámara intente compensar cambios de luz.

- Soporte para móvil (tripode / soporte mecánico)
  - Rol: mantener el iPhone fijo y con orientación estable durante las capturas o el modo en vivo.
  - Justificación técnica: la estabilidad física evita desenfoques por movimiento y simplifica el tracking entre frames; además permite usar optical-flow de forma más robusta (menos errores de seguimiento). También facilita mantener la distancia y el encuadre constantes, lo cual reduce la necesidad de re-detecciones frecuentes.

- Dispositivo de captura: iPhone 16
  - Rol: fuente de video (por ejemplo usando DroidCam, aplicación de streaming) o como cámara para fotos de plantilla/ pruebas.
  - Especificaciones relevantes (generales para iPhone 16):
    - Cámara de alta resolución, buena sensibilidad en baja luz, estabilización óptica y procesamiento de imagen avanzado.
    - Salida de video de calidad suficiente (1080p / 4K según configuración).
  - Justificación técnica: la alta calidad óptica del iPhone facilita obtener imágenes nítidas y con buen contraste, lo que mejora la extracción de contornos y símbolos. Además, su capacidad para mantener enfoque y exposición constantes (si se configuran manualmente) ayuda a reproducir condiciones de prueba y reduce variabilidad en los templates.

- Equipo de procesamiento: Asus ROG Zephyrus (ordenador portátil)
  - Rol: ejecutar el código de detección y clasificación (procesamiento en tiempo real y trabajo de desarrollo).
  - Características relevantes (típicas de la gama Zephyrus):
    - CPU de alto rendimiento (múltiples núcleos/threads), GPU dedicada (NVIDIA), 16+ GB RAM o más.
  - Justificación técnica: el pipeline (findContours, warpPerspective, matchTemplate, distanceTransform, Canny, operaciones morfológicas y heurísticas) es intensivo en CPU y se beneficia de una GPU y de una CPU rápida para mantener FPS aceptables. La gama ROG Zephyrus ofrece suficiente potencia para ejecutar el detector en vivo con optimizaciones (threading, optical flow) y para compilar/usar builds optimizados de OpenCV si es necesario.

Configuración práctica y recomendaciones durante pruebas
- Cámara (iPhone):
  - Resolución sugerida en vivo: 1920x1080 (1080p) a 30 fps es un buen compromiso entre calidad y consumo de CPU.
  - Ajustes recomendados: fijar exposición y balance de blancos (si la app o el teléfono lo permite), desactivar HDR/auto-enhance para evitar cambios dinámicos entre frames.
- Iluminación:
  - Colocar la luz blanca en ángulo que minimice reflejos directos en el recubrimiento brillante de la carta.
  - Evitar fuentes mixtas (luz cálida + luz fría) para no introducir dominantes de color.
- Soporte:
  - Rígido y sin vibraciones; alejar la fuente de vibraciones (por ejemplo, mesa estable).
- Equipo (Asus):
  - Usar conexión por cable o Wi-Fi robusta si el iPhone transmite video por la red.
  - Si se dispone de GPU compatible y OpenCV con CUDA, activar aceleración en operaciones críticas.

Resumen breve de la justificación:
- Estos elementos permiten controlar condiciones de captura (iluminación, encuadre, estabilización) y disponer de suficiente capacidad de cómputo para ejecutar el pipeline y sus optimizaciones (tracking + threading + posible aceleración por GPU). Usar hardware de calidad (iPhone 16 + Zephyrus) mejora precisión y fps sin cambiar la lógica algorítmica.

---

3) Software: características técnicas y requisitos
-------------------------------------------------
Requisitos
- Python 3.9+ (preferible 3.10/3.11).  
- Paquetes pip:
  - opencv-python (o idealmente OpenCV compilado con contrib, TBB y CUDA si hay GPU)  
  - numpy  
  - matplotlib (solo para visualización offline)  
  - scipy (opcional, mejora suavizado y detección de picos)  
- Herramientas de desarrollo: git, editor (VSCode), pandoc (si se quiere convertir Markdown→PDF).

Recomendación de OpenCV:
- Para máxima velocidad en vivo: compilar OpenCV con TBB/IPP, habilitar optimizaciones SSE/AVX y (si se dispone) soporte CUDA. Si no se compila, instalar `opencv-python` o `opencv-python-headless` funciona, pero con menor rendimiento.

Justificación:
- OpenCV provee primitives optimizadas para procesamiento de imágenes; NumPy es la base de cálculos matriciales; SciPy es útil para filtros de suavizado (gaussiano) con implementación eficiente. Matplotlib se usa solo para debugging o modo interactivo; en producción usar `cv2.imshow`.

---

4) Hoja de ruta del desarrollo
------------------------------
Fases (cronograma recomendado)
Fase 0 — Levantamiento / análisis (1-2 días)
- Evaluar código existente, identificar módulos pesados y cuellos de botella.

Fase 1 — Refactorización en módulos (2-4 días)
- Separar utilidades (utils.py), detección de contornos (contours.py), transformaciones (transforms.py), extracción de símbolos y matching (symbols.py), features (features.py), pipeline (processing.py), clasificadores de palos (suit_classifier.py), y live detector (live_detector.py).
- Resolver import cycles (mover imports dentro de funciones si es necesario).

Fase 2 — Optimización sin cambiar lógica (2-4 días)
- Implementar tracking por optical-flow entre frames redetectados.
- Implementar ThreadPoolExecutor para clasificación asíncrona.
- Cache/preprocesado de templates si se desea.

Fase 3 — Robustez de clasificación (2-3 días)
- Añadir heurísticas para números problemáticos (2,3,8,10,5,6).
- Mejorar extracción y multi-threshold strategy.

Fase 4 — Tests y benchmark (2 días)
- Medir timings, ajustar redetect_interval, min_card_area, pool size.
- Validación con imágenes reales y video en diferentes condiciones.

Fase 5 — Documentación y empaquetado (1-2 días)
- Documentar código, README y generar la memoria técnica (este documento).
- Preparar script de despliegue o Dockerfile (opcional).

---

5) Solución: arquitectura y explicación
---------------------------------------
Componentes principales (refactorizados)
- utils.py: funciones auxiliares (mostrar imagen, cargar imagen).
- contours.py: find_contours, find_card_contour_from_binary, find_all_card_contours_from_binary.
- transforms.py: order_points, four_point_transform, extract_top_left_corner.
- symbols.py: extracción de símbolos, multi-template scoring, enhanced rank classification.
- features.py: shape metrics, color stats, heart/clover/spade/diamond features, resegment.
- templates.py: builder de templates desde directorios.
- processing.py: pipeline para procesar una imagen (load -> binarize -> find contour -> warp -> extract corner -> extract symbols).
- suit_classifier.py: classify_suit_v7 (lógica compuesta con heurísticas y overrides).
- live_detector.py: captura en vivo optimizada (optical-flow tracking, ThreadPool, redetección periódica).
- card_dectector.py: orquestador (CLI, main, interactive_process_and_classify, run_live_mode).

Buenas prácticas aplicadas:
- Separación de responsabilidades.
- Import dinámico para evitar ciclos.
- Operaciones pesadas ejecutadas de forma asíncrona y/o en un subconjunto de frames.
- Mismos algoritmos de clasificación (no se cambia la lógica de decisión), solo optimizaciones y heurísticas adicionales.

---

6) Diagrama de decisión para clasificación de las cartas
--------------------------------------------------------
Se presenta un diagrama de decisión secuencial en texto (equivalente lógico):

1. Entrada (símbolo suit binario + corner_rgb)
2. Template matching (enhanced_match_symbol_v2) → obtener best_template_name y template_score
3. Extraer color_stats = extract_symbol_color_stats_v3
4. Extraer shape = compute_shape_metrics
5. Extraer features por palo:
   - heart_feats = compute_heart_features
   - diamond_feats = compute_diamond_features
   - clover_feats = compute_clover_features
   - spade_feats = compute_spade_features
6. Chequeo degenerado:
   - Si shape indicia degenerado (baja circularity, baja solidity o aspect_ratio extremo) → intentar resegment_symbol_if_degenerate
   - Recalcular features si se aplicó resegment.
7. Definir color_group: "red" si color_stats["is_red"] True, else "black"
8. Candidate suits:
   - Si red: ["Diamantes","Corazones"]
   - Si black: ["Picas","Treboles"]
9. Para cada candidato calcular:
   - base = template_score (o penalizado si template difiere)
   - heur = suma de heurísticas dependientes de features (por ejemplo diamond_feature_score * peso, heart_lobes_score * peso, clover_score * peso, spade_score * peso)
   - ajustar heur por prototipo de color si aplica, penalizaciones por características contradictorias (ej. diamond fuerte penaliza corazón)
   - si degenerate_fix_applied y template fuerte, añadir bonificaciones
   - final_score = base + heur
10. chosen = max(final_score), final_score clamp a 1.0
11. Overrides:
    - Si template indica Diamantes fuerte y diamond_feats suficiente → forzar Diamantes
    - Si template indica Treboles fuerte y clover_feats suficiente → forzar Treboles
    - Si clover_feats extremadamente fuerte (3 lóbulos) → forzar Treboles
12. Resultado: suit elegido, score, debug_info (detalles de features y scores)

Este flujo garantiza que templates (evidencia visual inmediata) y features geométricos y de color (evidencia estructural) se combinen mediante reglas heurísticas y reglas de override seguras.

---

7) Secuencialización de las operaciones sobre las imágenes
----------------------------------------------------------
A continuación se describe la secuencia exacta de operaciones realizadas para procesar una imagen o un frame y las decisiones de configuración. Para cada operación se indica la función principal, sus parámetros relevantes (y por qué se eligieron).

Pipeline principal (procesamiento de una sola imagen)
1. Carga imagen
   - function: utils.load_image_rgb(path)
   - justificación: cv2.imread devuelve BGR; convertimos a RGB para coherencia en procesamiento/visualización.

2. Conversión a gris + binarización
   - function: cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
   - function: cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   - parámetros: Otsu automático — robusto a iluminación variable en la carta.
   - justificación: obtener máscara binaria para detectar contornos externos.

3. Detección de contornos candidatos
   - function: contours.find_all_card_contours_from_binary(binary, min_area=10000 o configurado)
   - cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   - filtro: area >= min_area; approxPolyDP eps = 0.02 * arcLength; exigir approx len == 4
   - justificación:
     - RETR_EXTERNAL para evitar contornos internos (cartas típicamente tienen un contorno notable).
     - CHAIN_APPROX_SIMPLE para reducir puntos guardados.
     - min_area evita ruido; 10000 es para imágenes de alta resolución (ajustar según captura).
     - eps 0.02 — equilibrio entre aproximación precisa y tolerancia.

4. Transformación perspectiva (warp)
   - function: transforms.four_point_transform(image_rgb, pts, width=300, height=420)
   - pasos y parámetros:
     - Ordenación de puntos: order_points usando centroid + ángulos.
     - Cálculo de longitudes laterales y promedio para inferir orientación.
     - Si carta horizontal → rotación de puntos.
     - target_aspect_ratio ~ 1.4 (estándar de póker).
     - final_width/final_height calculados manteniendo aspecto o usando defaults width/height.
     - cv2.getPerspectiveTransform + cv2.warpPerspective con (final_width, final_height).
   - justificación:
     - Normalizar dimensiones de carta para extracción consistente de símbolos.
     - Mantener orientación vertical.

5. Extracción esquina superior izquierda (rank & suit)
   - function: transforms.extract_top_left_corner(warped_card, w_ratio=0.28, h_ratio=0.40)
   - parámetros:
     - w_ratio/h_ratio adaptativos (aumentan si carta pequeña): default 0.28/0.40, límites mínimos (rw>=60, rh>=80).
   - justificación:
     - Permite capturar rank + suit en la esquina con margen fijo, robusto a distintos tamaños.

6. Extracción de símbolos de la esquina
   - function: symbols.extract_symbols_from_corner(corner_rgb, min_area=50, horizontal_gap=20)
   - secuencia interna y parámetros:
     - grayscale -> varios umbrales:
       - Otsu (THRESH_BINARY_INV + OTSU)
       - adaptiveThreshold Gaussian (ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=11, C=2)
       - adaptiveThreshold Mean (blockSize=11, C=2)
     - combinar umbrales con OR para robustez (cubre distintos contrastes).
     - medianBlur (k=3) para reducir ruido sal y pimienta.
     - morphologyEx(MORPH_CLOSE, kernel_small (2x2)) para unir trazos finos.
     - findContours en combined; filtrar por área adaptiva (adaptive_min_area = max(min_area, (h*w)//200)).
     - Filtrar cajas por proporciones (descartar cajas extremadamente anchas/altas).
     - Fusionar cajas cercanas horizontalmente (horizontal_gap default 20) y por alineamiento vertical (<25 px).
     - padding 3 px en crop y MORPH_OPEN para limpieza final.
   - justificación:
     - Múltiples métodos de umbral aumentan robustez frente a iluminación variable.
     - Operaciones morfológicas y filtros reducen falsos positivos.

7. Clasificación de rank (valor)
   - function: symbols.enhanced_rank_classification(rank_symbol, rank_templates)
   - pasos:
     - equalizeHist + bilateralFilter para mejorar contraste y reducir ruido.
     - enhanced_match_symbol_v2: multiprocesamiento de templates con multi_template_scores
       - multi_template_scores normaliza al tamaño base (W0=32x32), equalize, edge detection Canny (30,150), distance transforms y matchTemplate (TM_CCOEFF_NORMED).
       - compara escalas [0.9,1.0,1.1] por template.
       - combina métricas: correlación normalizada (corr_score), edge_score, chamfer_score (basado en distance transforms), cosine similarity (vectores normalizados) y ssim simplificado.
       - pesos: corr 0.30, edge 0.20, chamfer 0.20, cosine 0.15, ssim 0.15. (equilibrio entre correlación global y forma/contorno)
     - heurísticas extras para números problemáticos (2,3,8,10,5,6):
       - probar variantes (otsu, adaptive, thickened/ thinned, rotaciones pequeñas) y escalas adicionales.
       - contadores de huecos (connectedComponents sobre versión invertida) para distinguir 8 (espera 2 huecos).
       - proyección horizontal/inversa para detectar separación en '10' (dos componentes horizontales).
       - distribución izquierda/derecha para 3 vs 2 (3 más masa a la derecha).
       - fallback por shape_metrics (defects, circularity, vertices) si scores bajos.
   - justificación:
     - Matching por template es la base; heurísticas evitan ambigüedad en dígitos con formas similares.

8. Clasificación de suit (palo)
   - function: suit_classifier.classify_suit_v7(suit_symbol_binary, corner_rgb, suit_templates, suit_color_prototypes)
   - pasos:
     - enhanced_match_symbol_v2 para templates de suits (mismo multi-metric scoring).
     - compute color stats (extract_symbol_color_stats_v3) — HSV ranges amplios para rojo, canal a de LAB, Cr en YCrCb; raw_score combinado y sigmoid para red_confidence.
     - compute_shape_metrics, compute_heart_features, compute_diamond_features, compute_clover_features, compute_spade_features.
     - resegment_symbol_if_degenerate si shape sospechoso (baja circularity/solidity/aspect>3.0).
     - definir grupo de candidatos por color (red vs black).
     - combinar base (template_score) + heurísticas por palo (ej. diamond_feature_score*0.70, heart lobes*0.80, clover_feature*0.75, spade_feature*0.75) y bonificaciones/penalizaciones.
     - proto_dist: similitud a prototipos de color de templates rojos para ayudar a discriminar Corazones vs Diamantes.
     - Overrides finales en casos de evidencias muy fuertes de template+feature.
   - justificación:
     - combinación de evidencia visual (template), color y forma estructural es más robusta que solo template matching.

9. Optimización en modo Live (sin alterar lógica)
   - live_detector:
     - redetection completa cada `redetect_interval` frames (p. ej. 6).
     - tracking de esquinas por `calcOpticalFlowPyrLK` entre redetecciones (puntos: las 4 esquinas).
       - parámetros ópticos: winSize=(21,21), maxLevel=3, criteria=(EPS|COUNT, 30, 0.01)
     - clasificación asíncrona con ThreadPoolExecutor (max_workers configurable).
     - mantener historial por celda (deque len configurable) y votación para considerar resultado "stable".
   - justificación:
     - optical flow y threads reducen carga por frame manteniendo mismos pasos de clasificación (solo que se ejecutan con menor frecuencia o en background).

Parámetros clave recomendados (defaults)
- min_card_area (detección): 10000 (imágenes), 25000 (video 1280x720) — ajustar según resolución/cam.
- four_point_transform: width=300, height=420 (mantener aspect ~1.4).
- extract_top_left_corner: w_ratio=0.28, h_ratio=0.40, min rw=60, rh=80.
- threshold/adaptive: adaptive blockSize=11, C=2; Otsu combinado con adaptive para robustez.
- Canny: thresholds (30,150).
- multi_template scales: [0.9,1.0,1.1] (en rank mejora añadir 0.8 y 1.2 en evaluación extensiva).
- ThreadPoolExecutor max_workers: 3 (ajustable según núcleos).
- redetect_interval: 6 frames (ajustable).

---

8) Otras tareas realizadas
--------------------------
- Refactorización completa en módulos, manteniendo API pública y lógica.
- Resolución de import cycles (imports dinámicos en run_live_mode o mover imports).
- Implementación de optimización de live: optical-flow tracking + ThreadPool.
- Mejoras heurísticas para números problemáticos (2,3,8,10).
- Mejoras en extracción de símbolo (combinación de múltiples umbrales + morphología).
- Añadido resegmentación degenerate para shapes difíciles (especialmente diamantes).
- Documentación inline (docstrings y comentarios) y separación lógica para facilitar pruebas unitarias.

---

9) Código fuente documentado — organización y funciones clave
-------------------------------------------------------------
Estructura de archivos clave (ubicados en la raíz del repo tras refactor):
- card_dectector.py
  - main(), run_live_mode(), interactive_process_and_classify()
- utils.py
  - load_image_rgb(path)
  - show_img(img, title, figsize, cmap, mode)
  - wait_enter(enabled, message)
- contours.py
  - find_contours(binary)
  - find_card_contour_from_binary(binary, min_area=10000)
  - find_all_card_contours_from_binary(binary, min_area=10000)
- transforms.py
  - order_points(pts)
  - four_point_transform(image_rgb, pts, width=300, height=420)
  - extract_top_left_corner(warped_card, w_ratio=0.28, h_ratio=0.40)
- symbols.py
  - extract_symbols_from_corner(corner_rgb, min_area=50, horizontal_gap=20)
  - multi_template_scores(symbol_img, templates_list)
  - enhanced_match_symbol_v2(symbol_img, templates_dict, symbol_type="rank"/"suit")
  - enhanced_rank_classification(rank_symbol, rank_templates)
- features.py
  - extract_symbol_color_stats_v3(symbol_binary, corner_rgb, template_name)
  - compute_shape_metrics(symbol_binary)
  - compute_heart_features(symbol_binary)
  - compute_clover_features(symbol_binary)
  - compute_spade_features(symbol_binary)
  - compute_diamond_features(symbol_binary)
  - resegment_symbol_if_degenerate(symbol_binary, corner_rgb, template_name)
  - extract_symbol_color_vector(symbol_binary, corner_rgb)
- suit_classifier.py
  - classify_suit_v7(suit_symbol_binary, corner_rgb, suit_templates, suit_color_prototypes)
- processing.py
  - process_card_image(image_path, visualize=False)
- templates.py
  - build_templates_from_directory(template_base_dir="template/")
- live_detector.py
  - class LiveCardDetector: process_frame_multi, run, draw_info_panel_multi, optical flow tracking logic

Cada función principal contiene docstrings explicativos y parámetros documentados. Para buen nivel de mantenimiento, se recomienda añadir tests unitarios por función de features y por matching.

---

10) Anexos / referencias
------------------------
- Repositorio (ubicación de archivos referenciados): Diegoza04/card_Detector (raíz).
- Herramientas recomendadas: OpenCV (compilada con contrib/TBB/CUDA para mejorar rendimiento en vivo), Python 3.10+, NumPy, SciPy opcional.

---

