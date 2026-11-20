# =====================================================================
# INVENTARIOS · LÓGICA UNIFICADA (reemplaza "PARTE DE SOFIA")
# Paso R en Power Query. Entrada: 'dataset' (weekly_df_final_for_modeling).
# Columnas usadas: sku, week, channel, region, units_sold, stock_available, delivery_days
# Salida: 'output' = una fila por SKU y semana con SS, ROP, S y estado.
#
# Mismas ecuaciones que la app (página Inventarios):
#   d_t   = promedio de las 4 semanas PREVIAS (pronóstico disponible en t)
#   sigma = RMSE de ese pronóstico en las 12 semanas previas (mín. 8)
#   L, sigma_L = media y desviación de delivery_days del SKU / 7 ; R = 1 semana
#   SS  = z * sqrt((L+R)*sigma^2 + d^2*sigma_L^2)
#   ROP = d*L + z * sqrt(L*sigma^2 + d^2*sigma_L^2)
#   S   = d*(L+R) + SS
# Nivel: SKU total (suma de canal x región en el intervalo común), no por fila.
# =====================================================================
library(dplyr)

service_level <- 0.95
z <- qnorm(service_level)
R <- 1

num <- function(x) suppressWarnings(as.numeric(as.character(x)))
d <- dataset %>%
  mutate(week = as.Date(week), sku = as.character(sku),
         units_sold = num(units_sold), stock_available = num(stock_available),
         delivery_days = num(delivery_days))

# Plazo de entrega por SKU (semanas)
lt <- d %>% group_by(sku) %>%
  summarise(L = round(mean(delivery_days, na.rm = TRUE) / 7, 2),        # igual que la app
            sigma_L = round(sd(delivery_days, na.rm = TRUE) / 7, 2), .groups = "drop")

# Intervalo común: semanas en que existen las 9 combinaciones canal-región
span <- d %>% group_by(sku, channel, region) %>%
  summarise(first = min(week), last = max(week), .groups = "drop") %>%
  group_by(sku) %>% summarise(start = max(first), end = min(last), .groups = "drop")

roll_mean_prev <- function(x, k) {          # promedio de las k observaciones previas
  n <- length(x); out <- rep(NA_real_, n)
  if (n > k) for (i in (k + 1):n) out[i] <- mean(x[(i - k):(i - 1)])
  out
}
roll_rmse_prev <- function(e, k, kmin) {    # RMSE de los errores previos (ventana k)
  n <- length(e); out <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    if (i > 1) {
      w <- e[max(1, i - k):(i - 1)]; w <- w[!is.na(w)]
      if (length(w) >= kmin) out[i] <- sqrt(mean(w^2))
    }
  }
  out
}

output <- d %>%
  inner_join(span, by = "sku") %>%
  filter(week >= start, week <= end) %>%
  group_by(sku, week) %>%
  summarise(demanda = sum(units_sold, na.rm = TRUE),
            stock = sum(stock_available, na.rm = TRUE), .groups = "drop") %>%
  arrange(sku, week) %>%
  inner_join(lt, by = "sku") %>%
  group_by(sku) %>%
  mutate(
    pronostico = roll_mean_prev(demanda, 4),
    error = demanda - pronostico,
    sigma = roll_rmse_prev(error, 12, 8),
    Stock_Seguridad = z * sqrt((L + R) * sigma^2 + pronostico^2 * sigma_L^2),
    Punto_Reorden_ROP = pronostico * L + z * sqrt(L * sigma^2 + pronostico^2 * sigma_L^2),
    Inventario_Maximo_S = pronostico * (L + R) + Stock_Seguridad,
    demanda_real_LR = demanda + (L + R - 1) * lead(demanda),
    cubierto = as.numeric(demanda_real_LR <= Inventario_Maximo_S)
  ) %>%
  ungroup() %>%
  filter(!is.na(Stock_Seguridad)) %>%
  mutate(
    Stock_Seguridad = ceiling(Stock_Seguridad),
    Punto_Reorden_ROP = ceiling(Punto_Reorden_ROP),
    Inventario_Maximo_S = ceiling(Inventario_Maximo_S),
    estado = case_when(
      stock < Stock_Seguridad ~ "Agotamiento crítico",
      stock < Punto_Reorden_ROP ~ "Reordenar urgente",
      stock > Inventario_Maximo_S ~ "Sobre-stock",
      TRUE ~ "Óptimo"
    )
  ) %>%
  select(sku, week, demanda, pronostico, sigma, L, sigma_L, stock,
         Stock_Seguridad, Punto_Reorden_ROP, Inventario_Maximo_S, estado,
         demanda_real_LR, cubierto)

# =====================================================================
# VISUAL R 1 (página Inventarios): control de inventario
# Campos del visual: week, stock, Stock_Seguridad, Punto_Reorden_ROP, Inventario_Maximo_S, sku
# =====================================================================
# library(ggplot2); library(dplyr)
# n_sku <- length(unique(dataset$sku))
# df <- dataset %>% mutate(week = as.Date(week)) %>% group_by(week) %>%
#   summarise(n = n_distinct(sku), stock = sum(stock), SS = sum(Stock_Seguridad),
#             ROP = sum(Punto_Reorden_ROP), S = sum(Inventario_Maximo_S), .groups = "drop") %>%
#   filter(n == n_sku) %>% arrange(week)
# alerts <- df %>% filter(stock < ROP)
# ggplot(df, aes(week)) +
#   geom_ribbon(aes(ymin = 0, ymax = SS), fill = "#C0392B", alpha = .15) +
#   geom_line(aes(y = S, colour = "Stock objetivo (S)"), linetype = "dashed") +
#   geom_line(aes(y = ROP, colour = "Punto de reorden (ROP)"), linewidth = 1) +
#   geom_line(aes(y = SS, colour = "Stock de seguridad (SS)"), linetype = "dotted") +
#   geom_line(aes(y = stock, colour = "Stock disponible real"), linewidth = 1) +
#   geom_point(data = alerts, aes(y = stock), colour = "#D9534F", size = 2) +
#   scale_colour_manual(values = c("Stock objetivo (S)" = "#2B5C8F", "Punto de reorden (ROP)" = "#E67E22",
#                                  "Stock de seguridad (SS)" = "#C0392B", "Stock disponible real" = "#27AE60")) +
#   labs(title = "Control de inventario y alertas de reabastecimiento", x = NULL, y = "Unidades", colour = NULL) +
#   theme_minimal() + theme(legend.position = "bottom")

# =====================================================================
# VISUAL R 2 (página Inventarios): estado por SKU en su última semana
# =====================================================================
# library(ggplot2); library(dplyr)
# last <- dataset %>% mutate(week = as.Date(week)) %>% group_by(sku) %>% filter(week == max(week)) %>%
#   ungroup() %>% mutate(brecha = stock - Punto_Reorden_ROP)
# ggplot(last, aes(x = brecha, y = reorder(sku, brecha), fill = estado)) + geom_col() +
#   geom_vline(xintercept = 0, linetype = "dashed") +
#   scale_fill_manual(values = c("Agotamiento crítico" = "#D9534F", "Reordenar urgente" = "#F0AD4E",
#                                "Óptimo" = "#5CB85C", "Sobre-stock" = "#0275D8")) +
#   labs(title = "Estado operativo por SKU", x = "Stock disponible − ROP", y = "SKU", fill = "Estado") +
#   theme_minimal()

# =====================================================================
# MAPA EJECUTIVO (página 4): usar la tabla BI_Inventario.csv descargada de la app
# (pestaña Descargas del Tablero BI). Trae, por SKU: modelo campeón, forecast 12 semanas,
# SS, ROP, S, stock y estado con la misma lógica hacia adelante (forecast del campeón y
# su error fuera de muestra), que R no puede recalcular sin los modelos.
# =====================================================================
