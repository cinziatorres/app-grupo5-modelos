# App integrada de precio, demanda e inventario · Grupo 5

## Novedades de esta versión (lista para publicar como web)

- **Tablero BI integrador**: réplica de las 4 páginas del reporte Power BI (forecast, inventarios, elasticidad y mapa ejecutivo) con sus segmentadores, más descarga de las tablas BI_*.
- **Inicio y objetivos**: 4 objetivos específicos con indicador, meta numérica y resultado calculado en vivo.
- **Datos y análisis descriptivo**: media, mediana, moda, desviación, mínimo, máximo; correlación; estandarización z-score; reducción de dimensiones con PCA.
- **Comparar escenarios**: evaluación prescriptiva de 3 o más escenarios de precio y servicio, por SKU o para los 10 SKU priorizados.
- **Método y fuentes**: fundamentación científica de cada algoritmo, explicación de la regularización y referencias.

### Versión 3 (lógica unificada y más precisa)

- **Forecast:** se agregó la **combinación** (promedio simple de los modelos admisibles) como quinto candidato, con la misma regla de selección (`combinar_modelos.py`). WAPE de comprobación promedio 14.40% → 11.55% (mediana 11.18% → 10.91%). Detalle por SKU en `artifacts/combination_update.csv`.
- **Inventarios:** una sola lógica para la app y el Power BI. SS = z·√((L+R)·σ² + d²·σL²), S = D(L+R) + SS, ROP = D(L) + z·√(L·σ² + d²·σL²). σ = RMSE del error de pronóstico fuera de muestra; L y σL desde delivery_days (≈ 0.43 y 0.08 semanas); stock inicial = stock_available de la última semana. La simulación solo verifica el servicio.
- **Control histórico (BI):** misma ecuación con el pronóstico disponible en cada semana (promedio de 4 semanas previas, σ = RMSE de las 12 previas). Cobertura histórica 95.5% (meta 95%) vs 97.9% de la lógica anterior, con 13.6% menos stock de seguridad (10 SKU priorizados).
- `power-bi/R_inventarios_unificado.R`: script R para reemplazar la «PARTE DE SOFIA» del PBIX con la misma lógica.

### Versión 2 (aditamento del profesor)

- **Inventarios** ahora tiene 3 gráficos: (1) ciclo de inventario con **punto de reorden**, emisión y llegada de pedidos y **tiempo de reabastecimiento L**; (2) **stock de seguridad** (de dónde sale en la distribución simulada y cómo cambia con el nivel de servicio); (3) **stock total objetivo** S = demanda en L + demanda en R + SS, con la posición de inventario.
- **Flechas de variación**: llevan signo explícito y no aparecen si no hay cambio (antes «+0.0%» mostraba flecha verde). En inventario se ocultan los cambios menores al error de la simulación Monte Carlo, que antes daban flechas en sentido contrario al esperado (p. ej. bajar el precio mostraba el stock de seguridad bajando).

### Correcciones

1. **Seasonal Naive nunca se evaluaba**: `modelos.py` llamaba a `fc.seasonal_naive_forecast`, que no existía; el benchmark quedaba «no elegible» con el motivo `'Seasonal Naive'`. Se agregó la función y `reparar_naive.py` lo evaluó con los mismos cortes y la misma regla de ganador. Cambios de ganador en `artifacts/naive_repair.csv`.
2. **Dependencias fuera de la carpeta**: los datos (`../anexos/...`) y `forecast_core` (`../forecast-30-skus`) estaban fuera de la app. Ahora están en `data/` y en la raíz.
3. **Modelos de 1.5 GB**: los `.pkl` de SARIMAX/DHR no se pueden subir a GitHub. Se guardan los parámetros estimados en `models/*.json` (364 KB) y el modelo se reconstruye con `filter(params)`, sin reestimar; Ridge guarda su escalador y coeficientes. Pronósticos idénticos a los `.pkl` (diferencia máxima 1e-4 unidades).
4. `server.address = "localhost"` en `.streamlit/config.toml` impedía el acceso web; se retiró.
5. Escenarios y exportaciones antiguas (calculados con el ganador anterior al arreglo del Naive) se retiraron para no mezclar versiones.

## Publicar como web (Streamlit Community Cloud)

1. Subir **todo el contenido de esta carpeta** a un repositorio de GitHub.
2. En https://share.streamlit.io → **Create app** → repositorio, rama `main`, archivo `app.py`.
3. En **Advanced settings** elegir **Python 3.12** → **Deploy**.

En la nube, los escenarios guardados y las exportaciones se escriben en el servidor y se pierden al reiniciarse la app; para conservarlos usar los botones de descarga.

## Ejecutar en local

```
pip install -r requirements.txt
streamlit run app.py
```

---

## Documentación original

Streamlit local para treinta SKU, escenarios manuales de precio, forecast y revisión periódica conectados. No busca un precio óptimo. Motor, interfaz y recorrido en navegador comprobados; los datos operativos iniciales siguen siendo supuestos académicos editables.

## Abrir

Doble clic en **abrir-app.bat**; abrir **http://localhost:8510**.

Fuente: `data/weekly_df_final_for_modeling.xlsx`. Modelos livianos en `models/`.

## Recorrido

1. Seleccionar SKU y periodo. Ver precio base y pronóstico del ganador.
2. Cambiar precio. La demanda cambia automáticamente según la sensibilidad de cada SKU, calibrada con el ajuste semanal y su incertidumbre bajo el supuesto de respuesta decreciente. La evidencia débil se muestra por separado. **Usar elasticidad manual** permite explorar una hipótesis ε ≤ 0, separada de la estimación. Promoción y composición permanecen constantes.
3. Abrir **Inventarios** y capturar stock, tránsito, atrasados, plazo, revisión, servicio y capacidad si se conoce. Comparar seguridad, stock objetivo y reposición con la base.
4. Consultar **Modelos y elasticidad** para parámetros, errores, controles e intervalos.
5. Guardar/descargar un escenario. **Escenarios guardados** conserva cifras y permite cargar parámetros para editar.
6. **Resumen de 30 SKU** calcula los últimos parámetros por SKU editados en esta sesión; los restantes usan supuestos iniciales. Exportar ZIP o guardar la exportación local para Power BI.

El cambio de precio ejecuta inferencia, sin reentrenar ni cambiar de campeón. Parámetros por SKU se conservan al navegar dentro de la sesión. Para retomarlos después de cerrar, guardar un escenario y recuperarlo desde su página.

Se pueden marcar las semanas festivas programadas; sin selección se supone que no hay semanas festivas. La base y el escenario comparten ese calendario. Con plazos fraccionarios, se supone demanda uniforme dentro de cada semana. La demanda central durante protección se promedia sobre los mismos plazos simulados, conservando la trayectoria estacional.

## Método

- **Ancla Pa:** media simple de precios semanales agregados de las cuatro últimas semanas; el forecast siempre se evalúa en ese contexto. **Referencia P0:** por defecto Pa o una entrada manual; se traduce mediante la misma respuesta de precio: Q_base=Q_ancla(P0/Pa)^ε. El precio propuesto usa Q_escenario=Q_ancla(P1/Pa)^ε, sin una segunda intervención a través de regresores predictivos. Supone cambio proporcional común entre canales/regiones. No es precio óptimo.
- **Forecast:** Naive anual, Fourier/Ridge, SARIMAX estacional y DHR. Ridge compara K=1/2/3, tendencia lineal/cuadrática y alpha=0.1/1/10/100. SARIMAX compara (1,0,0) o (1,0,1), con AR o MA anual. DHR compara tres órdenes ARMA con K=2. Calibración de cuatro semanas antes de selección H12; otras doce para comprobación. Precio/promoción se fijan desde el origen; calendario de prueba se trata como programado y conocido. WAPE/MAE de selección eligen entre admisibles; el error de comprobación no ordena candidatos. Convergencia/raíces en comprobación también restringen admisibilidad, por lo que esta no es completamente ajena a todos los criterios de selección.
- **Corte:** semanas futuras comunes. Si la historia completa del SKU termina antes del corte global, se pronostica un puente y se indica; no se fabrican ventas observadas.
- **Respuesta al precio:** regresión de totales semanales que suma la exposición de días disponibles con sus precios, promociones, presentación y día; tendencia/Fourier anual K2 y efectos de combinación canal/región. Fuente diaria pública verificada contra el mismo Excel. Se conservan las comparaciones con/sin precio en dos bloques H12 previos al holdout H12 como diagnóstico; ningún SKU supera su criterio confirmatorio. Para escenarios, el coeficiente libre y su error agrupado calibran una normal aproximada condicionada a ε ≤ 0; su media es la elasticidad automática. El signo procede del supuesto económico y no demuestra significancia ni causalidad. Hay respuesta por precio en los treinta SKU, pequeña y con incertidumbre. [Diagnóstico y parámetros](diagnostico-precio/DIAGNOSTICO.md).
- **OLS histórico:** se conserva como diagnóstico, con sus 30 IC agrupados que incluyen cero. No se interpreta como efecto causal ni se invierte su signo. **Escenario:** Q₁=Q₀(P₁/P₀)^ε con sensibilidad condicionada automática o hipótesis manual explícita, aplicada una sola vez.
- **Ingreso:** suma P×Q en semanas visibles; potencial si se atiende toda la demanda. Moneda pendiente: unidades monetarias. No calcula utilidad con costos inventados.
- **Inventario:** revisión periódica (R, S). SS = z·√((L+R)·σ² + d²·σL²), S = D(L+R) + SS, ROP = D(L) + z·√(L·σ² + d²·σL²), pedido = máx(0, S − posición). σ = RMSE del error fuera de muestra del campeón; L y σL desde delivery_days. La simulación (bootstrap de errores, plazo aleatorio) verifica el servicio de ciclo.
- **Incertidumbre:** bloques de cuatro errores externos centrados de selección/comprobación, 2 000 trayectorias, semilla 805. Elasticidad automática sorteada de la normal resumida condicionada al signo; usa el coeficiente libre y error agrupado semanales corregidos. Manual: lognormal negativa con media/desviación ingresadas, preservando ε < 0 en todas las trayectorias. No utiliza el error estándar del OLS anterior. Plazo lognormal con media/desviación configuradas, fijo si desviación cero. Base y escenario comparten números aleatorios. Error proporcional a demanda o absoluto constante: hipótesis visibles. Se computa horizonte que cubre todos los plazos simulados, aunque exceda el gráfico.

La curva central usa la media de ε; no equivale exactamente a la media de demandas simuladas. La banda 5–95% es de **simulación**, sin cobertura futura acreditada. Dos orígenes H12 aportan pocos errores; bootstrap no crea más evidencia. La autora declara la [fuente sintética para practicar forecasting](https://www.kaggle.com/code/beatafaron/fmcg-forecasting-ml-real-demand-signal-2/input?scriptVersionId=316098962). No se verificó un generador que imponga una elasticidad estructural. Los coeficientes positivos se conservan como resultados observacionales y se retiran del simulador automático; no se invierte su signo.

## Entradas operativas

Inicio académico: entrega media 2 semanas, desviación 0.5, revisión 1 y servicio de ciclo 95%, derivados del ejemplo del informe; **no parámetros empresariales verificados**. Físico/tránsito/atrasados inician en cero como entradas pendientes. Capacidad sin límite hasta capturarla. Seguridad depende de incertidumbre durante protección, además de media.

Capacidad se compara por SKU con el objetivo. No asigna presupuesto/almacén conjuntamente, no estima sustitución entre SKU ni envía compras. Una operación real requiere confirmar existencias, recepciones, restricciones, costos y abastecimiento.

## Persistencia y Power BI

Escenarios: `runs/<run_id>/`, con manifiesto/versiones, forecast visible, forecast de protección e inventario. Política actual: `precio-sensibilidad-condicionada-v4`; salidas nuevas registran sensibilidad, supuesto de signo, intervalos libres/condicionados y evidencia predictiva. Ajustes v3 conservados en `price_response.csv`; escenarios v4 en `price_scenario.csv`. Ejecuciones anteriores conservadas y marcadas como antecedentes en la app. Conjunto exportado: `exports/<run_id>/`. El puntero `latest.json` se reemplaza atómicamente después de escribir todos los archivos.

En Power BI, crear función `LeerExportacion` pegando `power-bi/Leer-exportacion.pq`. Crear parámetro de texto `RutaApp` a esta carpeta, y consultas `LeerExportacion(RutaApp, "summary")`, `LeerExportacion(RutaApp, "weekly")`, `LeerExportacion(RutaApp, "inventory")`. Tipar week como fecha, cantidades/precios como números y SKU/run_id como texto. Relacionar summary/weekly por SKU dentro de una ejecución; inventory tiene dos escenarios por SKU. Actualizar tras guardar otra exportación. El PBIX original no se sobrescribe ni se afirma probado en Desktop.

## Preparación y comprobación

Verificación vigente: [recibo v4](diagnostico-precio/verificacion-v4.json) y [navegador v4](diagnostico-precio/verificacion-navegador-v4.json), respuesta automática y rutas de precio en treinta SKU, incertidumbre, demanda protegida, seis páginas, persistencia/exportación y huellas conservadas. [Estimador semanal](diagnostico-precio/verificacion-modelo-semanal.json) comprobado con efectos conocidos; sus datos de prueba no sustituyen el curso. Los recibos v2/v3 documentan antecedentes, no la conducta actual.

Las [pruebas integradas](tests/verificacion_integrada.json), [de interfaz](tests/verificacion_ui.json) y [de navegador](tests/verificacion_navegador.json) anteriores documentan la versión previa; no acreditan automáticamente la política nueva. El benchmark Seasonal Naive figura no elegible con motivo `'Seasonal Naive'` en la evaluación H12: queda pendiente reparar y repetir su evaluación. No se cambió el ganador durante la revisión del precio.

```powershell
python entrenar_integrado.py
python preparar_elasticidad.py
python diagnostico-precio\recuperar-fuente.py
python diagnostico-precio\trazar-datos.py
python preparar_respuesta_precio.py
python respuesta_precio.py
python diagnostico-precio\probar-modelo-semanal.py
python diagnostico-precio\verificar-v4.py
```

El entrenamiento reanuda recibos compatibles por SKU. Antes de cambiar datos, conservar la versión vigente y regenerar entrenamiento, elasticidad y evidencia. Fuente/modelos deben tener intervalos y versiones compatibles; los manifiestos registran SHA-256.

Referencias: [validación temporal](https://otexts.com/fpp3/tscv.html), [regresión armónica](https://otexts.com/fpp3/dhr.html), [escenarios de regresión](https://otexts.com/fpp3/forecasting-regression.html), [inventarios y seguridad, MIT](https://ocw.mit.edu/courses/15-763j-manufacturing-system-and-supply-chain-design-spring-2005/e1800430fdccaf618b22bd551fc00fac_summary.pdf). La [guía del trabajo](../forecast-30-skus/GUIA-MODELOS-PARAMETROS.md) explica familias y distingue la versión previa H4 de esta evaluación H12.
