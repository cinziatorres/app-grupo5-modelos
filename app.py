from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dataclasses import asdict
import json,math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import extras
from motor import Scenario,default_scenario,ROOT,PRICE_POLICY,PRICE_BASE_POLICY,load_registry,load_manifest,load_elasticity,load_price_response,load_meta,load_data,forecast_base,calculate,bundle,save,list_runs,restore,export_portfolio

st.set_page_config(page_title='Precio, demanda e inventario',page_icon='📈',layout='wide')
st.markdown('''<style>
:root{--ink:#123b5d;--base:#3978b3;--scenario:#cf6225;--canvas:#f3f7fc;}
.stApp{background:var(--canvas);color:#172f46;}
h1,h2,h3{font-family:Aptos,"Segoe UI",sans-serif;color:var(--ink);letter-spacing:-.025em;}
h1{font-size:2.4rem!important;font-weight:650!important;}
[data-testid="stSidebar"]{background:#e7eff8;border-right:1px solid #cbd9e8;}
[data-testid="stMetric"]{border-top:3px solid #3978b3;padding:14px 8px 10px;background:white;}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums;}
[data-testid="stCaptionContainer"] p{color:#46617a!important;}
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible{outline:3px solid #00877a;outline-offset:3px;}
.block-container{max-width:1400px;padding-top:2rem;}
@media(max-width:700px){h1{font-size:1.8rem!important;}.block-container{padding:1.2rem;}}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def tables(fingerprint):return load_registry(),load_elasticity(),load_price_response(),load_manifest()

@st.cache_data(show_spinner=False,max_entries=90)
def base_for(sku,horizon,lead,sd,review,samples,seed,holidays,version_token):
    config=Scenario(sku=sku,horizon=horizon,lead_weeks=lead,lead_sd=sd,review_weeks=review,samples=samples,seed=seed,holiday_weeks=holidays)
    return forecast_base(config,source_for(version_token[0]))

@st.cache_data(show_spinner=False)
def source_for(source_version):return load_data()

def run(config):
    m=load_manifest();version_token=(m['source_sha256'],(ROOT/f'training/{config.sku}_result.json').stat().st_mtime_ns,m['as_of'],PRICE_BASE_POLICY)
    frame,meta,info=base_for(config.sku,config.horizon,config.lead_weeks,config.lead_sd,config.review_weeks,config.samples,config.seed,tuple(config.holiday_weeks),version_token)
    info=dict(info)
    if config.reference_price is not None:info['price_reference']=float(config.reference_price)
    base=(frame,meta,info)
    return calculate(config,base)

def format_n(value):return f'{value:,.1f}'
def format_u(value):return f'{value:,.0f} u.'

def chart(result):
    df=result['weekly'];fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.week,y=df.scenario_high,mode='lines',line={'width':0,'color':'#cf6225'},showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df.week,y=df.scenario_low,mode='lines',line={'width':0,'color':'#cf6225'},fill='tonexty',fillcolor='rgba(207,98,37,.13)',name='Banda de simulación 90%',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df.week,y=df.base_demand,name='Base',line={'color':'#3978b3','width':3,'dash':'dot'},hovertemplate='%{x|%d/%m/%Y}<br>Base: %{y:,.1f} u.<extra></extra>'))
    fig.add_trace(go.Scatter(x=df.week,y=df.scenario_demand,name='Precio propuesto',line={'color':'#cf6225','width':3},hovertemplate='%{x|%d/%m/%Y}<br>Escenario: %{y:,.1f} u.<extra></extra>'))
    fig.update_layout(height=360,margin={'l':10,'r':10,'t':12,'b':10},paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white',legend={'orientation':'h','y':1.15},yaxis={'title':'Unidades por semana','rangemode':'tozero'},xaxis={'title':None},font={'family':'Segoe UI','color':'#172f46'})
    return fig

def comparison(result):
    s=result['summary'];iv=result['inventory'].set_index('scenario')
    rows=[('Precio por unidad',s['price_reference'],s['price_proposed']),('Ventas previstas del periodo',s['base_units'],s['scenario_units']),('Ingreso potencial del periodo',s['base_revenue'],s['scenario_revenue'])]
    for label,key in [('Stock de seguridad','safety_stock'),('Stock objetivo','target_stock'),('Pedido sugerido','suggested_order')]:rows.append((label,iv.loc['Base',key],iv.loc['Precio propuesto',key]))
    return pd.DataFrame([{'Resultado':label,'Base':float(a),'Precio propuesto':float(b),'Cambio':float(b-a)} for label,a,b in rows])

def show_result(result):
    st.caption('Respuesta al precio: '+result['info'].get('price_response_status','resultado anterior a la corrección; asociación OLS o supuesto antiguo'))
    s=result['summary'];c1,c2,c3=st.columns(3)
    c1.metric('Precio propuesto',f"{s['price_proposed']:.2f} u.m.",extras.delta(s['price_proposed']/s['price_reference']*100-100,1,'%','vs base'))
    c2.metric('Ventas previstas del periodo',f"{s['scenario_units']:,.1f} u.",extras.delta(s['demand_change_pct'],2,'%','vs base'))
    c3.metric('Ingreso potencial del periodo',f"{s['scenario_revenue']:,.0f} u.m.",extras.delta(s['revenue_change_pct'],2,'%','vs base'))
    st.caption(f"Elasticidad aplicada: {s['elasticity']:.4f} · cambio frente a la base: {s['scenario_units']-s['base_units']:+,.1f} unidades.")
    if 'demand_change_low_pct' in s:
        st.caption(f"Rango de sensibilidad 90% del cambio por precio: [{s['demand_change_low_pct']:+.3f}%, {s['demand_change_high_pct']:+.3f}%], condicionado al supuesto decreciente. La curva central usa la elasticidad media; este rango no incluye errores del forecast.")
    st.plotly_chart(chart(result),width='stretch')
    st.caption('La banda muestra sensibilidad y errores simulados; su cobertura futura no está acreditada. Ingreso potencial supone atender toda la demanda estimada.')
    st.dataframe(comparison(result),hide_index=True,width='stretch',column_config={c:st.column_config.NumberColumn(format='%.2f') for c in ['Base','Precio propuesto','Cambio']})

if not (ROOT/'artifacts/training_manifest.json').exists() and not list((ROOT/'training').glob('*_result.json')):
    st.title('Precio, demanda e inventario')
    st.info('La nueva evaluación de los treinta SKU está en curso. Al terminar, la app recuperará los modelos guardados.')
    statuses=[]
    for file in (ROOT/'training').glob('*_status.json'):
        record=json.loads(file.read_text());statuses.append({'SKU':record['sku'],'Estado':record['phase']})
    if statuses:st.dataframe(pd.DataFrame(statuses),hide_index=True)
    st.stop()

fingerprint=[(p.name,p.stat().st_mtime_ns) for p in (ROOT/'training').glob('*_result.json')]
fingerprint.append(('manifest',(ROOT/'artifacts/training_manifest.json').stat().st_mtime_ns if (ROOT/'artifacts/training_manifest.json').exists() else 0))
fingerprint.append(('price_response',(ROOT/'artifacts/price_response_manifest.json').stat().st_mtime_ns if (ROOT/'artifacts/price_response_manifest.json').exists() else 0))
fingerprint.append(('price_scenario',(ROOT/'artifacts/price_scenario_manifest.json').stat().st_mtime_ns if (ROOT/'artifacts/price_scenario_manifest.json').exists() else 0))
try:registry,elastic,responses,manifest=tables(fingerprint)
except Exception as exc:
    st.error(f'No se pudo abrir el registro de modelos: {exc}');st.stop()

if 'profiles' not in st.session_state:st.session_state.profiles={}
if 'pending_config' in st.session_state:
    pending=st.session_state.pop('pending_config');st.session_state['sku']=pending['sku'];st.session_state.profiles[pending['sku']]=pending
    if pending.get('elasticity_override') is not None and pending['elasticity_override']>0:
        pending.update(price_change_pct=0.,elasticity_override=None,elasticity_se_override=0.)
        st.session_state.legacy_notice='La elasticidad positiva antigua se retira al recalcular. Se aplicará el modelo semanal vigente; el resultado guardado conserva sus cifras originales.'
    for k,v in pending.items():
        if v is not None:st.session_state[f"{pending['sku']}_{k}"]=v
        else:st.session_state.pop(f"{pending['sku']}_{k}",None)
    st.session_state[f"{pending['sku']}_service_percent"]=pending['service']*100
    st.session_state[f"{pending['sku']}_manual_p0"]=pending['reference_price'] is not None
    st.session_state[f"{pending['sku']}_manual_epsilon"]=pending['elasticity_override'] is not None
    st.session_state[f"{pending['sku']}_limited"]=pending['capacity'] is not None
    st.session_state[f"{pending['sku']}_error_label"]='Proporcional a la demanda' if pending['error_policy']=='proportional' else 'Error absoluto constante'
    st.session_state['page']='Escenario'

st.sidebar.title('Planificación por SKU')
sku=st.sidebar.selectbox('SKU',registry.sku.sort_values().to_list(),key='sku')
page=st.sidebar.radio('Ver',['Inicio y objetivos','Datos y análisis descriptivo','Escenario','Inventarios','Modelos y elasticidad','Comparar escenarios','Tablero BI integrador','Escenarios guardados','Resumen de 30 SKU','Método y fuentes'],key='page')
st.sidebar.caption(f"Corte de información: {manifest['as_of']}\n\nFuente histórica del curso. Simulación académica.")
# Streamlit incorpora los valores iniciales a la identidad del widget. Mantener
# una semilla estable mientras se edita el SKU evita reiniciar el control en el
# siguiente recálculo. Al cambiar de SKU se recupera su perfil conservado.
if st.session_state.get('widget_seed_sku')!=sku:
    st.session_state.widget_seed_sku=sku
    st.session_state.widget_seed=dict(st.session_state.profiles.get(sku,asdict(default_scenario(sku))))
default=st.session_state.widget_seed
key=lambda name:f'{sku}_{name}'

st.title('Precio, demanda e inventario')
st.caption('Un escenario comercial. Una trayectoria de ventas. Un cálculo de reposición.')
if 'legacy_notice' in st.session_state:st.info(st.session_state.pop('legacy_notice'))
if not manifest['complete']:st.info(f"Entrenamiento en curso: {len(registry)}/30 SKU disponibles. Puedes probar los completados; la exportación del conjunto se habilita al terminar.")
e=elastic.set_index('sku').loc[sku];pr=responses.set_index('sku').loc[sku];meta=load_meta(sku)

with st.sidebar.expander('Periodo y precio',expanded=True):
    horizon=st.number_input('Semanas del periodo',1,26,int(default['horizon']),key=key('horizon'))
    st.caption('El precio ajusta la demanda usando la estimación semanal de este SKU y su incertidumbre, bajo el supuesto de respuesta decreciente.')
    manual=st.checkbox('Usar elasticidad manual',value=default['elasticity_override'] is not None,key=key('manual_epsilon'))
    epsilon=st.number_input('Elasticidad del escenario',-5.,0.,min(0.,float(default['elasticity_override'] if default['elasticity_override'] is not None else pr.elasticity_used)),.01,format='%.4f',key=key('elasticity_override')) if manual else None
    if manual:st.caption('Hipótesis tuya, no estimación del Excel. Por ejemplo, ε = −1 supone demanda inversamente proporcional al precio; ε = 0 supone ausencia de respuesta.')
    else:st.caption(f"ε aplicada = {pr.elasticity_used:.4f}. Respuesta pequeña y condicionada al supuesto económico; la evidencia de esta base es débil.")
    change=st.slider('Cambio de precio (%)',-40.,40.,float(default['price_change_pct']),.5,key=key('price_change_pct'))
    manual_p0=st.checkbox('Definir precio de referencia',value=default['reference_price'] is not None,key=key('manual_p0'))
    p0=st.number_input('Precio de referencia por unidad',min_value=.01,value=float(default['reference_price'] or e.p0_recent),step=.1,key=key('reference_price')) if manual_p0 else None
    st.caption('El forecast parte del precio medio de las últimas cuatro semanas. Una referencia manual se ajusta con la misma respuesta de precio; promoción y calendario se mantienen iguales.')
    dates=pd.date_range(pd.Timestamp(manifest['as_of'])+pd.Timedelta(weeks=1),periods=52,freq='W-MON').strftime('%Y-%m-%d').to_list()
    holidays=st.multiselect('Semanas festivas conocidas',dates,default=default.get('holiday_weeks',[]),key=key('holiday_weeks'))
    st.caption('Marca el lunes de cada semana programada. Sin selección se supone que no hay semanas festivas; el calendario se mantiene igual en base y escenario.')

with st.sidebar.expander('Stock y abastecimiento'):
    st.caption('Valores iniciales tomados de la base: stock = stock_available de la última semana (suma canal × región); L y σL = delivery_days del SKU. Todos son editables.')
    stock=st.number_input('Stock físico (unidades)',min_value=0.,value=float(default['stock']),step=10.,key=key('stock'))
    transit=st.number_input('Pedidos en tránsito (unidades)',min_value=0.,value=float(default['in_transit']),step=10.,key=key('in_transit'))
    backlog=st.number_input('Pedidos atrasados (unidades)',min_value=0.,value=float(default['backorders']),step=10.,key=key('backorders'))
    lead=st.number_input('Plazo medio de entrega L (semanas)',.05,12.,float(default['lead_weeks']),.05,key=key('lead_weeks'),help='Por defecto: promedio de delivery_days del SKU / 7 (≈ 3 días).')
    sd=st.number_input('Desviación del plazo σL (semanas)',0.,3.,float(default['lead_sd']),.01,key=key('lead_sd'),help='Por defecto: desviación de delivery_days del SKU / 7.')
    review=st.number_input('Tiempo entre revisiones (semanas)',.25,4.,float(default['review_weeks']),.25,key=key('review_weeks'))
    service=st.slider('Servicio de ciclo (%)',80.,99.,float(default['service']*100),.5,key=key('service_percent'))/100
    limited=st.checkbox('Definir capacidad por SKU',value=default['capacity'] is not None,key=key('limited'))
    capacity=st.number_input('Capacidad disponible (unidades)',min_value=0.,value=float(default['capacity'] or 1000.),step=50.,key=key('capacity')) if limited else None

with st.sidebar.expander('Sensibilidad e incertidumbre'):
    se=st.number_input('Dispersión del supuesto de elasticidad',0.,2.,float(default['elasticity_se_override']),.01,key=key('elasticity_se_override')) if manual and epsilon<0 else 0.
    include=st.checkbox('Incluir incertidumbre de elasticidad',value=default['include_elasticity_uncertainty'],key=key('include_elasticity_uncertainty'))
    label=st.selectbox('Comportamiento del error',['Proporcional a la demanda','Error absoluto constante'],index=0 if default['error_policy']=='proportional' else 1,key=key('error_label'))
    policy='proportional' if label=='Proporcional a la demanda' else 'absolute'
    st.caption('Automática: incertidumbre del ajuste semanal condicionada al signo no positivo. Manual: dispersión subjetiva ingresada. Las trayectorias mantienen el supuesto decreciente; no se estimó cómo cambia la varianza de ventas con el precio.')

config=Scenario(sku=sku,horizon=int(horizon),price_change_pct=change,reference_price=p0,lead_weeks=lead,lead_sd=sd,review_weeks=review,service=service,stock=stock,in_transit=transit,backorders=backlog,elasticity_override=epsilon,elasticity_se_override=se,include_elasticity_uncertainty=include,error_policy=policy,capacity=capacity,holiday_weeks=tuple(holidays))
st.session_state.profiles[sku]=asdict(config)
try:
    with st.spinner('Calculando el escenario y la reposición…'):result=run(config)
except Exception as exc:
    st.error(f'No se pudo calcular el escenario: {exc}');st.stop()

info=result['info']
response_label=info['price_response_status'] if epsilon is None else f"elasticidad supuesta: {result['summary']['elasticity']:.3f} ({info['elasticity_source']})"
st.caption(f"{sku} · {meta['winner']} · última semana completa: {meta['history_end']} · precio base: {info['price_reference']:.2f} u.m. · {response_label}")

if page=='Inicio y objetivos':
    extras.page_inicio()

elif page=='Datos y análisis descriptivo':
    extras.page_datos()

elif page=='Tablero BI integrador':
    extras.page_bi()

elif page=='Comparar escenarios':
    extras.page_escenarios(config,run)

elif page=='Escenario':
    st.subheader('De precio a ventas previstas')
    show_result(result)
    with st.expander('Supuestos y límites del escenario',expanded=True):
        for warning in result['warnings']:st.write('• '+warning)
        st.write('Cambio de precio aplicado una sola vez; composición comercial y promoción constantes. Ventas observadas son una aproximación de demanda.')
    with st.expander('Detalle semanal'):
        st.dataframe(result['weekly'][['week','base_demand','scenario_demand','scenario_price','base_revenue','scenario_revenue','scenario_low','scenario_high']],hide_index=True,width='stretch')
    c1,c2=st.columns([2,1]);name=c1.text_input('Nombre del escenario',value=f'{sku}: precio {change:+.1f}%')
    if c2.button('Guardar escenario',type='primary',width='stretch'):
        path=save(result,name);st.success(f'Escenario guardado: {path.name}')
    st.download_button('Descargar resultado completo',bundle(result),file_name=f'{sku}-escenario.zip',mime='application/zip')

elif page=='Inventarios':
    st.subheader('Reposición para la misma demanda')
    iv=result['inventory'].set_index('scenario');r=iv.loc['Precio propuesto']
    c1,c2,c3=st.columns(3)
    b=iv.loc['Base']
    # Variación vs. precio base calculada sobre los valores mostrados (redondeados), con signo
    # explícito; sin flecha si no cambia. Flecha roja = el escenario exige más inventario.
    d_ss,d_s,d_o=round(r.safety_stock)-round(b.safety_stock),round(r.target_stock)-round(b.target_stock),round(r.suggested_order)-round(b.suggested_order)
    c1.metric('Stock de seguridad',format_u(r.safety_stock),extras.delta(d_ss,0,' u.','vs base'),delta_color='inverse')
    c2.metric('Stock objetivo',format_u(r.target_stock),extras.delta(d_s,0,' u.','vs base'),delta_color='inverse')
    c3.metric('Pedido sugerido',format_u(r.suggested_order),extras.delta(d_o,0,' u.','vs base'),delta_color='inverse')
    st.caption('Flechas = precio propuesto frente al precio base: ↑ rojo = se necesita más inventario; ↓ verde = se necesita menos; sin flecha = sin cambio.')
    extras.formula_panel(result)
    st.write(f"La revisión protege **{lead+review:.2f} semanas**: entrega ({lead:.2f}) más revisión ({review:.2f}). Posición de inventario: **{r.inventory_position:,.0f} unidades**.")
    st.caption(f"Se calculan {info['computed_horizon']} semanas para cubrir todos los plazos simulados; el gráfico del escenario muestra {horizon}. Servicio elegido es probabilidad de cubrir un ciclo, no porcentaje de unidades atendidas.")
    if not r.capacity_feasible:st.error('El stock objetivo excede la capacidad definida para este SKU. Ajusta escenario o abastecimiento.')
    else:st.success('El stock objetivo cabe en la capacidad definida.' if limited else 'Cálculo sin límite de capacidad: confirma espacio y presupuesto antes de una compra.')
    view=result['inventory'].rename(columns={'scenario':'Escenario','expected_protection_demand':'Demanda esperada protegida','safety_stock':'Stock de seguridad','target_stock':'Stock objetivo','inventory_position':'Posición de inventario','suggested_order':'Pedido sugerido','reorder_point':'Punto de pedido (indicador)','simulated_cycle_service':'Servicio en simulación','capacity_feasible':'Cabe en capacidad'})
    st.dataframe(view[['Escenario','Demanda esperada protegida','Stock de seguridad','Stock objetivo','Posición de inventario','Pedido sugerido','Servicio en simulación','Cabe en capacidad']],hide_index=True,width='stretch')
    st.info('El pedido es una sugerencia de revisión periódica. El stock en tránsito se trata como comprometido para esta posición; su calendario de llegada no está confirmado. No se envían órdenes de compra.')
    st.write('SS = z·√((L+R)·σ² + d²·σL²), con σ = RMSE semanal del error del pronóstico fuera de muestra y L, σL desde delivery_days. S = demanda esperada en L+R + SS; ROP = demanda en L + z·√(L·σ² + d²·σL²); pedido = máx(0, S − posición). La simulación solo verifica el servicio logrado.')
    st.markdown('#### Visualización de la política de inventario')
    extras.inventory_charts(result,run)
    with st.expander('Trayectoria usada para protección'):
        st.dataframe(result['protection_forecast'][['week','base_demand','scenario_demand']],hide_index=True,width='stretch')

elif page=='Modelos y elasticidad':
    st.subheader('Evidencia del pronóstico')
    st.write(f"Ganador: **{meta['winner']}**. Selección WAPE: **{meta['selection_WAPE']:.2f}%**. Comprobación final WAPE: **{meta['holdout_WAPE']:.2f}%**, MAE: **{meta['holdout_MAE']:.1f} unidades**.")
    if 'Combinación (promedio)' in meta['params']:st.caption('Combinación (promedio) = promedio simple de los modelos admisibles: '+' + '.join(meta['params']['Combinación (promedio)']['members'])+'. Compite como quinto candidato con la misma regla (menor WAPE de selección).')
    metrics=pd.read_csv(ROOT/f'training/{sku}_metrics.csv')
    st.dataframe(metrics,hide_index=True,width='stretch')
    st.caption('Precio y promoción fijados con datos del origen. Calibración: 4 semanas; selección: 12; comprobación: otras 12. Los errores de comprobación no ordenan candidatos; su estabilidad numérica sí restringe admisibilidad.')
    with st.expander('Parámetros y alternativas evaluadas'):
        st.json(meta['params'])
        st.json(meta['final_diagnostics'])
        st.dataframe(pd.read_csv(ROOT/f'training/{sku}_calibration.csv'),hide_index=True,width='stretch')
    st.subheader('Regresión semanal corregida del mismo SKU')
    c1,c2,c3=st.columns(3);c1.metric('Coeficiente de precio libre',f'{pr.epsilon_corrected_free:.4f}');c2.metric('Intervalo 95% agrupado',f'[{pr.corrected_ci_low:.4f}, {pr.corrected_ci_high:.4f}]');c3.metric('Coeficiente no positivo',f'{pr.epsilon_corrected_nonpositive:.4f}')
    st.write('Se ajustan los totales semanales originales. Su media suma los días registrados con disponibilidad, cada uno con su precio, promoción, presentación y día de semana; incluye tendencia, estacionalidad anual y combinación canal/región. No trata una promoción de un día como una promoción de toda la semana.')
    st.write(f"Estado del escenario: **{pr.status}**. Diagnóstico predictivo separado: mejora de deviance al incluir precio **{pr.validation_price_gain_pct:+.2f}%**; WAPE de comprobación del modelo seleccionado de celdas **{pr.holdout_WAPE:.2f}%**.")
    st.subheader('Elasticidad aplicada en el escenario')
    st.write(f"**ε = {result['summary']['elasticity']:.4f}**. {result['info']['elasticity_source']}. La sensibilidad automática tiene intervalo condicional 95% **[{pr.scenario_elasticity_low:.4f}, {pr.scenario_elasticity_high:.4f}]**.")
    st.info('La estimación libre y su error estándar definen una distribución normal aproximada. El simulador la condiciona a ε ≤ 0 y usa su media. Así incorpora incertidumbre y el supuesto económico sin cambiar el signo del coeficiente ni asignar la misma elasticidad a todos los SKU. El signo negativo de ese intervalo proviene del supuesto; no prueba un efecto negativo en los datos.')
    st.write(f"Evidencia predictiva independiente: modelo **{pr.predictive_model_selected}**; cumple criterio confirmatorio: **{'sí' if bool(pr.price_supported) else 'no'}**. La falta de respaldo se comunica y no convierte la sensibilidad en cero. El WAPE anterior pertenece a ese modelo de celdas, no a la curva condicional del escenario.")
    st.caption('Estos modelos explican ventas registradas. No reconstruyen días ausentes ni demanda perdida; los intervalos son del ajuste libre y no identifican causalidad.')
    st.caption('Método, trazabilidad y pruebas: diagnostico-precio/DIAGNOSTICO.md. La covarianza agrupada por semana no corrige toda dependencia temporal.')
    with st.expander('Coeficientes del modelo predictivo seleccionado para comprobar evidencia'):
        st.dataframe(pd.read_csv(ROOT/'artifacts/price_response_coefficients.csv').query('sku == @sku'),hide_index=True,width='stretch')
    with st.expander('Regresión OLS anterior, conservada para comparar'):
        st.write(f"ε anterior: {e.elasticity:+.4f}; IC 95%: [{e.ci_low:.4f}, {e.ci_high:.4f}]. Era una asociación de log(ventas) con log(precio medio), promoción binaria, stock, entrega y calendario; no valida una intervención de precio.")
        coefficients=pd.read_csv(ROOT/'artifacts/elasticity_coefficients.csv');st.dataframe(coefficients[coefficients.sku.eq(sku)],hide_index=True,width='stretch')

elif page=='Escenarios guardados':
    st.subheader('Escenarios que puedes retomar')
    runs=list_runs()
    if not runs:st.info('Guarda un escenario desde la primera página para consultarlo aquí.')
    else:
        record=st.selectbox('Escenario guardado',runs,format_func=lambda x:f"{x['name']} · {x['sku']} · {pd.Timestamp(x['created_at']).tz_convert('America/Lima').strftime('%d/%m/%Y %H:%M')}")
        saved=restore(record['run_id']);show_result(saved)
        if saved['info'].get('price_policy_version')!=PRICE_POLICY or saved['info'].get('base_price_policy')!=PRICE_BASE_POLICY:st.warning('Resultado anterior a la corrección completa del precio. Se conserva como antecedente; no lo presentes como efecto causal. Al cargarlo, se aplicará la política actual.')
        st.caption('Estas cifras se recuperan del resultado guardado; no se recalculan con otros modelos.')
        if st.button('Cargar parámetros para editar'):
            st.session_state.pending_config=saved['config'];st.rerun()
        st.download_button('Descargar escenario guardado',bundle(saved),file_name=record['run_id']+'.zip',mime='application/zip')

elif page=='Resumen de 30 SKU':
    st.subheader('Escenarios del conjunto')
    st.write('Cada SKU usa sus últimos parámetros editados en esta sesión. Los que aún no editaste utilizan precio sin cambio y los supuestos académicos iniciales, con stock y tránsito en cero.')
    if st.button('Calcular los 30 SKU',type='primary',disabled=not manifest['complete']):
        rows=[];weekly=[];inventory_rows=[];configs=[];progress=st.progress(0)
        for i,s in enumerate(registry.sku.sort_values()):
            sc=Scenario(**st.session_state.profiles.get(s,asdict(default_scenario(s))))
            rr=run(sc)
            provenance={'elasticity_source':rr['info']['elasticity_source'],'price_response_status':rr['info']['price_response_status'],'price_policy_version':PRICE_POLICY,'base_price_policy':PRICE_BASE_POLICY}
            rows.append({'sku':s,'winner':rr['info']['forecast_model'],**rr['summary'],**provenance,'target_stock':float(rr['inventory'].iloc[1].target_stock),'safety_stock':float(rr['inventory'].iloc[1].safety_stock),'suggested_order':float(rr['inventory'].iloc[1].suggested_order),'as_of':rr['info']['as_of']})
            weekly.append(rr['weekly'].assign(sku=s,**provenance));inventory_rows.append(rr['inventory'].assign(sku=s,**provenance));configs.append(asdict(sc));progress.progress((i+1)/30)
        st.session_state.portfolio={'summary':pd.DataFrame(rows),'weekly':pd.concat(weekly,ignore_index=True),'inventory':pd.concat(inventory_rows,ignore_index=True),'configurations':configs,'price_policy_version':PRICE_POLICY,'base_price_policy':PRICE_BASE_POLICY}
    if 'portfolio' in st.session_state:
        pf=st.session_state.portfolio;st.dataframe(pf['summary'],hide_index=True,width='stretch')
        st.caption('Este resumen conserva la configuración de su última ejecución. Vuelve a calcular si editas otro escenario. No asigna recursos compartidos entre SKU.')
        import io,zipfile
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
            for table in ['summary','weekly','inventory']:archive.writestr(table+'.csv',pf[table].to_csv(index=False).encode('utf-8-sig'))
            archive.writestr('configurations.json',json.dumps(pf['configurations'],ensure_ascii=False,indent=2))
        st.download_button('Descargar los 30 SKU para Power BI',buf.getvalue(),file_name='escenarios-30-skus.zip',mime='application/zip')
        if st.button('Guardar exportación local para Power BI'):
            destination=export_portfolio(pf);st.success(f'Exportación completa guardada: {destination.name}')

else:
    st.subheader('Cómo se conectan los cálculos')
    st.write('Forecast al ancla histórica → precio definido por ti → sensibilidad de demanda → ingresos e inventarios → comparación con la base.')
    st.markdown('''
- **Base:** ventas semanales del modelo elegido, precio medio reciente o referencia manual y promoción reciente fija.
- **Respuesta:** regresión de totales semanales que suma la exposición diaria a precio/promoción/disponibilidad. Su coeficiente libre y error agrupado calibran una distribución aproximada, condicionada al supuesto ε ≤ 0. Su media es la sensibilidad automática por SKU.
- **Escenario:** Q₁ = Q₀ × (P₁/P₀)^ε. Precio mayor reduce unidades y precio menor las aumenta bajo ese supuesto. Una hipótesis manual puede reemplazar la sensibilidad. La evidencia débil se informa por separado y no anula la simulación.
- **Inventario:** revisión periódica (R, S); SS = z·√((L+R)σ² + d²σL²) con σ = RMSE fuera de muestra; L y σL de delivery_days; la simulación verifica el servicio; posición = stock + tránsito − atrasados.
- **Incertidumbre:** bootstrap de bloques de 4 errores externos; 2 000 trayectorias con semilla fija; plazo lognormal. Elasticidad automática: normal aproximada condicionada al signo; manual: lognormal negativa con dispersión subjetiva. Bandas condicionadas a esos supuestos, sin identificación causal.
- **Ámbito:** ventas sumadas del SKU; cambio proporcional común de precios, composición comercial estable y stock agrupado que atiende ese mismo ámbito.
- **Límites:** ventas no observan demanda perdida, muestras temporales pequeñas, bandas sin cobertura acreditada y semanas posteriores a doce con errores extrapolados.
''')
    st.write('La fuente de Kaggle se describe como sintética y destinada a practicar forecasting. Sus registros permiten un ejercicio académico; no acreditan una respuesta causal al precio. Stock operativo, tránsito, atrasados, capacidad y plazos actuales son supuestos editables hasta verificarlos. Las unidades monetarias no se etiquetan con una moneda sin confirmar.')
    st.markdown('[Descripción de la autora y archivo semanal](https://www.kaggle.com/code/beatafaron/fmcg-forecasting-ml-real-demand-signal-2/input?scriptVersionId=316098962)')
    st.caption('Las semanas futuras comienzan después del corte global. Un SKU cuya historia completa termina antes utiliza un pronóstico puente; no se rellenan ventas reales.')
    st.json(manifest)
    st.download_button('Descargar guía de uso y método',(ROOT/'README.md').read_bytes(),file_name='GUIA-APP-INTEGRADA.md',mime='text/markdown')
    st.divider()
    extras.section_fundamentacion()

