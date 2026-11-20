# Diagnóstico y corrección de la elasticidad

Revisión vigente: 8 de octubre de 2026, política de escenarios v4. Se conserva la base recibida por indicación de Leonardo; sustituirla es un último recurso. Excel SHA-256: `03b68f32303f5b1474561b0aa68db608a583793c95a2f000cb3ab78f104a9bb4`.

## Qué causaba el aumento de demanda en la app

La trayectoria queda reproducida: JU-021 tenía un coeficiente OLS **+0.0233883**. La app lo convertía automáticamente en el factor `1.10^0.0233883 = 1.0022316`. Subir precio 10% elevaba ventas previstas **0.223%**. Su intervalo agrupado por semana era **[−0.07035, +0.11712]**, p=0.6223. Ese aumento era la extrapolación de una asociación incierta, no evidencia de una respuesta económica positiva.

**La especificación también tenía defectos concretos.** La regresión combinaba un total de ventas semanal con medias de precio/stock y una bandera de promoción máxima. No distinguía semanas con distinta cantidad de días registrados ni cuánto duraba la promoción. La integración daba a su coeficiente una interpretación causal que el diseño no acreditaba. No apareció un error de signo ni una mezcla de SKU. Con el deslizador porcentual se conserva el mismo forecast base; esa trayectoria aumentaba por el OLS.

**Se verificó una segunda vía, distinta del deslizador:** el precio de referencia manual entraba directamente al forecast. En JU-021, SARIMAX producía **8 010.92 / 8 275.91 / 8 540.90** unidades al consultar referencias de **−10% / base / +10%**. El incremento de 3.2019% era otra asociación predictiva utilizada como intervención. La afirmación de que SARIMAX no participaba era válida solo para la trayectoria del deslizador, e incompleta para esta otra entrada. [Reproducción y corrección de ambas rutas](verificacion-rutas-precio.json), [script](verificar-rutas-precio.py).

## La misma fuente permitió verificar la transformación

Se recuperó la versión pública **16** del [dataset de Beata Faron](https://www.kaggle.com/datasets/beatafaron/fmcg-daily-sales-data-to-2022-2024), junto con sus notebooks y [repositorio primario](https://github.com/BeataFaron/fmcg_forecasting_repo). El CSV semanal público coincide con el Excel cargado en **las 24 columnas y las 31 027 filas**, con tolerancia numérica 1e−10. También confirma las 75 unidades recuperadas de MI-006 desde el lag consecutivo, sin modificar el Excel.

Los 190 757 registros diarios reconstruyen exactamente cantidades, precios, promociones, stock y entrega de esas semanas. No hay claves diarias duplicadas. Las huellas Git de los CSV dentro del ZIP coinciden con los archivos del repositorio en el commit `190925c87ae60b7ff22b26a8d643f900aae8b398`. No se cambió de caso ni se fabricaron registros.

La celda 16 del notebook público de EDA confirma esta agregación por SKU × semana × canal × región:

| Variable semanal | Construcción verificada | Consecuencia para el modelo |
| --- | --- | --- |
| Unidades | Suma de ventas diarias registradas | Aumentan con la cantidad de días representados |
| Precio | Media simple de precios diarios | No representa un único precio constante durante la semana |
| Promoción | Máximo de la bandera diaria | Un día de promoción y siete días producen la misma bandera |
| Stock | Media diaria | Puede ocultar días sin disponibilidad |
| Entrega | Media diaria | Su temporalidad operativa no queda identificada |

**21 084 de 31 027 celdas semanales (67.95%) contienen menos de siete días registrados.** Esto no significa que los días ausentes hayan tenido cero ventas. No se los completa con ceros ni se imputan ventas sin evidencia. **3 538 celdas** contienen días de stock cero, ocultos al mirar solo la media semanal. En los registros diarios, stock cero siempre coincide con ventas cero; entre días con stock positivo no hay ventas iguales al stock, por lo que no se observó una cota superior alcanzada. Esto no prueba ausencia general de demanda perdida.

La promoción diaria aparece en **14.92%** de registros, pero la bandera semanal vale uno en **61.34%** de celdas. Dentro de las semanas marcadas, solo **24.35%** de días registrados tienen promoción en promedio. El indicador recibido no es falso: era inadecuado interpretarlo como exposición promocional de toda la semana.

Evidencia: [trazabilidad](trazabilidad.json), [reconstrucción reproducible](trazar-datos.py), [exposición reconstruida](exposicion-semanal-reconstruida.csv) y [recibo público](fuentes-publicas/recepcion.json). Los notebooks se consultaron como fuentes; no se ejecutó código externo.

## Separar defectos de medición de falta de señal

Añadir únicamente log(días registrados) a la regresión original cambia ε de JU-021 de **+0.0234 a −0.0312**, todavía con IC que incluye cero. Modelar tasas y exposición promocional real lo aproxima a cero. Estos cambios prueban sensibilidad a la especificación; no permiten atribuir todo el coeficiente original a una sola variable.

Existe además un mecanismo de agregación plausible: log(media de precios) puede variar con la cantidad de días promediados, mientras el total de ventas aumenta con esos días. Una simulación de medición con precios independientes, ventas diarias constantes y la distribución observada de días produce una pendiente positiva media **+0.01835** en 200 repeticiones. **Es una demostración ilustrativa del sesgo posible, no una prueba del generador de esta base ni datos sustitutos.** [Contrastes y simulación](contrastes-exposicion.json).

La revisión diaria independiente, con PPML, promoción, calendario, presentación y efectos de combinación canal/región, también obtiene pendientes cercanas a cero. En JU-021: **−0.0136**, IC **[−0.0360, +0.0088]**. Ninguna de sus 30 pruebas de precio sobrevive Holm a 5%. Filtrar días con disponibilidad no demuestra que el precio sea exógeno. [Resultados diarios](elasticidad-diaria-contraste.csv).

La autora declara datos sintéticos destinados a forecasting. Se revisaron notebooks, repositorio e historial público disponible; **no se encontró el generador original que permitiría conocer la elasticidad estructural impuesta**. No se afirma que el precio haya sido excluido del generador, que su independencia sea demostrada o que los productos reales sean inelásticos. Lo comprobado es ausencia de una respuesta robusta utilizable con los diseños evaluados.

## Modelo semanal corregido

El objetivo continúa siendo **el total semanal recibido**, no una sustitución por ventas diarias. La información diaria se usa para construir sus regresores y exposición sin perder la mezcla de precios y promociones:

`μ_semana,celda = exp(X_semana,celda × β) × Σ_día_disponible exp(ε ln(P_día/5.25) + γ promo_día + κ presentación_día + δ día_de_semana)`

La suma incluye exclusivamente días registrados con stock positivo. No reconstruye días ausentes ni demanda no atendida. La presentación y el día de semana se representan mediante categorías, con una referencia omitida por grupo. X contiene constante, tendencia, Fourier anual K=2 y efectos de combinación canal × región. Se ajusta una media Poisson mediante máxima cuasiverosimilitud; no requiere que la varianza real sea idéntica a la media para interpretar su media. La inferencia del ajuste libre usa covarianza agrupada por semana, con sus límites temporales explícitos.

| Parámetro | Significado |
| --- | --- |
| ε | Elasticidad de la media de ventas registradas frente a un cambio proporcional común de todos los precios diarios, manteniendo las demás condiciones |
| γ | Efecto de promoción diaria en escala log; `exp(γ)` es el multiplicador promocional condicionado al modelo |
| Constante | Escala de ventas por día disponible para las categorías de referencia y el punto de calendario de referencia |
| Tendencia | Cambio de log(media) por año, dado el resto del diseño |
| K=2 | Dos pares seno/coseno: ciclo anual y medio ciclo, sin veinte o cincuenta parámetros estacionales |
| Efectos de celda | Diferencias persistentes entre cada combinación de canal/región y la referencia |
| κ y δ | Diferencias por presentación y día de semana respecto de sus referencias |
| 5.25 | Escala numérica de referencia para centrar precios; cambiarla traslada la constante y no altera ε |

Se comparan un modelo **sin precio** y uno **con ε ≤ 0**. La restricción no cambia el signo de un OLS: el optimizador reajusta conjuntamente todos los parámetros y puede elegir el límite ε=0. Se conserva también el ajuste libre para conocer la información que aporta la muestra. Sus intervalos no son intervalos del ajuste restringido ni efectos causales.

La restricción económica es un supuesto de demanda no creciente, apropiado para el ejercicio que definió Leonardo. La exogeneidad del precio no se consigue al imponerla. Precios y cantidades observados pueden reflejar variaciones conjuntas de oferta y demanda: [Cunningham, identificación de elasticidad](https://mixtape.scunning.com/07-instrumental_variables#elasticity-of-demand). Un instrumento necesita justificación; no se inventó uno usando stock, promoción o un rezago de precio. [Referencia del enlace log de Poisson](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Poisson.html).

## Validación de evidencia, separada de la sensibilidad

La regla confirmatoria v3 se definió antes de ejecutar la comparación: para respaldar precio, el coeficiente libre debe ser negativo con intervalo que excluya cero y prueba Holm sobre 30 SKU en la muestra anterior al holdout; los dos orígenes de validación deben estimar una pendiente negativa; incluir precio debe mejorar la deviance de validación. Son criterios de evidencia y estabilidad predictiva bajo el supuesto económico, no un diseño de identificación causal.

Se emplean **dos bloques H12 de validación y otro H12 posterior de comprobación**. La selección no usa las unidades del último bloque. Luego se reestima el modelo seleccionado con toda la historia disponible. Se validan ventas de celdas semanales con sus exposiciones observadas; estos errores no son una prueba del forecast operativo a doce semanas, que conserva sus propios modelos y artefactos.

| JU-021 | Resultado |
| --- | ---: |
| OLS anterior | +0.023388 |
| Modelo semanal corregido, libre | −0.009087 |
| IC 95% del corregido libre | [−0.075683, +0.057509] |
| Modelo corregido con restricción | −0.009088 |
| Ganancia en deviance al incluir precio | **−0.3239%**, empeora |
| Modelo seleccionado | Sin precio |
| WAPE del modelo semanal de celdas en holdout | 14.42% |

**Ninguno de los 30 SKU cumple conjuntamente los criterios confirmatorios.** Ese resultado se conserva como evidencia débil y no anula la sensibilidad del escenario v4. Existen coeficientes negativos e incluso intervalos nominales negativos en algunas muestras; no se ocultan ni se convierten por selección retrospectiva en efectos confirmados. El menor p ajustado preholdout es 0.2193. Precio mejora deviance en diez SKU y la empeora en otros; una pequeña mejora aislada no supera las demás condiciones. [Modelos y resultados](../artifacts/price_response.csv), [validación](validacion-respuesta-semanal.csv), [holdout](holdout-respuesta-semanal.csv), [preparación reproducible](../preparar_respuesta_precio.py).

La investigación previa también contrastó siete especificaciones y 180 formas adicionales, incluyendo PPML, diferencias, interacciones y efectos acumulados de rezagos. Los candidatos YO-009/MI-022/YO-024 fueron exploratorios y no sobrevivieron Holm en sus respectivas familias. Se mantienen [auditoría inicial](auditoria.json), [formas adicionales](formas-adicionales.json) y tablas asociadas como antecedentes; no se eligió una fórmula porque produjera el signo buscado.

## Integración vigente: sensibilidad condicionada v4

Leonardo confirmó que cambiar el precio debe cambiar la demanda del escenario. La v3 interpretaba falta de respaldo como omisión de precio, usando ε=0. Esto hacía inútil la sensibilidad automática; no era evidencia de inelasticidad. Se conserva esa comparación predictiva como diagnóstico, pero se separa de la construcción de escenarios.

La v4 usa el coeficiente libre semanal `b` y su error estándar agrupado `s` para calibrar **una distribución normal resumida**, condicionada al supuesto económico `ε ≤ 0`:

`b | ε ≈ Normal(ε, s²)` y `p(ε) ∝ 1 si ε ≤ 0; 0 si ε > 0`.

Por tanto, la distribución aproximada condicionada es `Normal(b,s²)` truncada en cero. La curva central usa su media:

`ε_escenario = b − s φ(−b/s) / Φ(−b/s)`.

φ y Φ son densidad y distribución acumulada normales estándar. `b` conserva la asociación estimada libremente; `s` describe su precisión; cero es el límite del supuesto de signo. No se fija una magnitud común, no se invierte el coeficiente, no se usa `−abs(b)` y no se confunde el optimizador restringido v3 con esta media. [Implementación y preparación](../respuesta_precio.py), [parámetros por SKU](../artifacts/price_scenario.csv), [manifiesto de supuestos](../artifacts/price_scenario_manifest.json). La [documentación de SciPy 1.17](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.truncnorm.html) establece cómo calcular momentos, cuantiles y sorteos con el límite estandarizado `−b/s`.

**Interpretación:** es sensibilidad calibrada y condicionada a una hipótesis económica. La distribución previa plana es impropia, pero su distribución resumida posterior es propia. Se usa una aproximación normal del estimador y la covarianza robusta, no un ajuste bayesiano completo de la cuasiverosimilitud ni una elasticidad causal identificada. El signo negativo de su intervalo proviene de la restricción, no de una nueva prueba estadística. Se siguen mostrando el intervalo libre y la validación original, incluidos los resultados no favorables.

| JU-021 | Valor y significado |
| --- | --- |
| Coeficiente libre b | −0.009087: resultado de la regresión semanal |
| Error estándar s | 0.033647: incertidumbre agrupada del coeficiente libre |
| IC libre 95% | [−0.075683, +0.057509], incluye cero |
| Media condicionada usada | **−0.030428**, depende del supuesto de signo |
| Intervalo condicionado 95% | [−0.081961, −0.001320], no prueba significancia negativa |
| Precio +10% | **Demanda central −0.289592%** |
| Precio −10% | **Demanda central +0.321108%**, aproximadamente |

Los treinta SKU tienen una media condicionada estrictamente negativa. Precio +10% produce descensos centrales entre 0.098% y 0.657%. Son efectos pequeños; no se amplifican para que el gráfico parezca más convincente. La app muestra unidades con un decimal, cambio absoluto, elasticidad aplicada y rango de sensibilidad por precio, además de la curva semanal.

**Límites del supuesto:** una distribución continua no incluye una masa de probabilidad puntual en ε=0, por lo que no estima la probabilidad de inelasticidad exacta. Con poca información de magnitud, mayor error puede aumentar la magnitud de la media negativa al condicionar por signo; eso no significa mayor evidencia. Debe contrastarse con ε=0 y otras magnitudes usando sensibilidad manual si se toman decisiones. Una señal positiva precisa llevaría la distribución condicionada cerca de cero; no justifica un efecto negativo grande. La normalidad y la restricción son supuestos visibles y revisables. La dependencia temporal, endogeneidad y demanda perdida siguen sin resolver.

El forecast se evalúa siempre al **ancla histórica Pa**, media de las últimas cuatro semanas. La referencia manual P0 no interviene en sus regresores: `Q_base(P0)=Q_ancla(P0/Pa)^ε` y `Q_escenario(P1)=Q_ancla(P1/Pa)^ε`. Equivale a `Q₁=Q₀(P₁/P₀)^ε`. La respuesta se aplica una sola vez a cada nivel; promoción, calendario y composición permanecen iguales. Se conserva el mismo Excel y los modelos de forecast.

Para incertidumbre automática se sortean elasticidades de esa misma distribución condicionada, compartiendo cada sorteo y los errores entre base y escenario. La trayectoria central usa la media de ε, y **no es exactamente la media de las demandas sorteadas** porque la exponenciación es no lineal. Los sorteos de elasticidad, bloques de errores y plazos se consideran independientes entre componentes; no se estimó su covarianza conjunta. La banda de simulación también incorpora errores externos y plazos; no tiene cobertura futura acreditada ni interpretación causal. La casilla de incertidumbre permite fijar ε para contrastar ambos cálculos. En modo manual se mantiene la lognormal de magnitud negativa y dispersión subjetiva.

La demanda ajustada alimenta inventario durante L+R. **El stock de seguridad no tiene que caer siempre al subir precio:** cambia también la incertidumbre del escenario; cantidades enteras pueden quedar iguales por redondeo. No se fuerza una variación artificial de stock. La demanda esperada protegida sí conserva la dirección del cambio de la demanda central.

La política nueva es `precio-sensibilidad-condicionada-v4`. Sus artefactos se guardan aparte de los ajustes v3 y verifican sus huellas; las ejecuciones y exportaciones previas se conservan como antecedentes. Se comunica el criterio confirmatorio mediante `price_supported`, sin usarlo como interruptor de la sensibilidad.

## Pruebas y límites de lo resuelto

[Prueba del estimador](verificacion-modelo-semanal.json): recupera una elasticidad conocida −1 como −0.9887 en datos de prueba con días ausentes y promociones. También recupera un efecto positivo conocido +0.7 como +0.7084 en ajuste libre y optimiza el restringido en el límite cero. Derivadas verificadas por diferencias finitas, error máximo inferior a 1e−7; escala semanal del cambio de precio comprobada independientemente. Estas simulaciones son exclusivamente pruebas y no entran a los resultados del curso.

Las pruebas v3 documentan el antecedente sin respuesta automática: [motor y rutas](verificacion-v3.json), [navegador](verificacion-navegador-v3.json). Las comprobaciones v4 suman 720 comparaciones de precio/referencia en treinta SKU y seis páginas AppTest; se registran en [motor, interfaz e integración](verificacion-v4.json) y [navegador real](verificacion-navegador-v4.json): respuesta automática distinta de cero, dirección coherente, precio neutro, dos rutas equivalentes, incertidumbre, demanda protegida, persistencia/exportación y huellas conservadas. El momento de la distribución se contrasta por integración numérica independiente de su densidad. [Script de integración](verificar-v4.py), [script de navegador](probar-ui-v4.cjs).

**No se ha identificado una elasticidad causal real por SKU.** La sensibilidad funciona bajo una hipótesis económica declarada. Si la rúbrica exige identificar un efecto causal, sigue faltando un diseño válido o evidencia adicional; no se cambia de base sin agotar y discutir esa necesidad con Leonardo.

Límites independientes: el benchmark Seasonal Naive figura no elegible con motivo `'Seasonal Naive'` en H12; no se reparó ni se alteraron ganadores durante esta revisión. El PBIX y el Word recibidos permanecen intactos. La demanda perdida, la completitud de días, el stock/plazo operativos y la cobertura futura de bandas siguen pendientes de verificación.

## Explicación para la sustentación

> Conservamos las ventas semanales y corregimos su exposición a precio, promoción y disponibilidad usando la misma fuente diaria. Los datos muestran una señal de precio débil. Para comparar escenarios, combinamos la estimación libre y su incertidumbre con el supuesto económico de demanda decreciente. Esto permite que un cambio de precio modifique la demanda y el inventario sin inventar una elasticidad común. Mostramos por separado la evidencia libre y el supuesto de signo: la simulación funciona, pero su respuesta no es un efecto causal demostrado.
