"""Un motor de escenarios, incertidumbre y reposición para la app y las exportaciones."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,io,json,math,pickle,zipfile,uuid
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy import stats
from modelos import ROOT,DATA,fc,NAMES,COMBO,predict,sarimax_spec,ridge_from_json
from respuesta_precio import VERSION as PRICE_POLICY,distribution as price_distribution

PRICE_BASE_POLICY='ancla-historica-v1'

@dataclass(frozen=True)
class Scenario:
    sku:str
    horizon:int=12
    price_change_pct:float=0.
    reference_price:float|None=None
    lead_weeks:float=2.
    lead_sd:float=.5
    review_weeks:float=1.
    service:float=.95
    stock:float=0.
    in_transit:float=0.
    backorders:float=0.
    elasticity_override:float|None=None
    elasticity_se_override:float=0.
    include_elasticity_uncertainty:bool=True
    error_policy:str='proportional'
    capacity:float|None=None
    samples:int=2000
    seed:int=805
    holiday_weeks:tuple[str,...]=()

from functools import lru_cache

@lru_cache(maxsize=1)
def _data_readonly():return load_data()

@lru_cache(maxsize=64)
def lead_time_from_data(sku):
    """Plazo de entrega del SKU desde la base: delivery_days medio y su desviación, en semanas."""
    d=_data_readonly();g=pd.to_numeric(d.loc[d.sku.astype(str)==sku,'delivery_days'],errors='coerce').dropna()
    return round(float(g.mean()/7),2),round(float(g.std()/7),2)

@lru_cache(maxsize=64)
def stock_from_data(sku):
    """Stock disponible del Excel en la última semana completa del SKU (suma canal × región)."""
    d=_data_readonly();end=pd.Timestamp(load_meta(sku)['history_end'])
    v=pd.to_numeric(d.loc[(d.sku.astype(str)==sku)&(d.week==end),'stock_available'],errors='coerce').sum()
    return float(round(v))

def default_scenario(sku,**kw):
    """Escenario inicial del SKU con supuestos tomados de la base (plazo y stock)."""
    L,sL=lead_time_from_data(sku)
    base=dict(sku=sku,lead_weeks=L,lead_sd=sL,stock=stock_from_data(sku));base.update(kw)
    return Scenario(**base)

def validate(config):
    if config.sku not in load_registry().sku.to_list():raise ValueError('SKU no registrado.')
    if not 1<=config.horizon<=26:raise ValueError('El periodo visible debe tener de 1 a 26 semanas.')
    numeric=[config.price_change_pct,config.lead_weeks,config.lead_sd,config.review_weeks,config.service,config.stock,config.in_transit,config.backorders,config.elasticity_se_override]
    if not np.isfinite(numeric).all():raise ValueError('Los parámetros deben ser finitos.')
    if config.price_change_pct<=-100:raise ValueError('El precio propuesto debe ser positivo.')
    if config.reference_price is not None and (not np.isfinite(config.reference_price) or config.reference_price<=0):raise ValueError('El precio de referencia debe ser positivo.')
    if config.lead_weeks<=0 or config.lead_sd<0 or config.review_weeks<0:raise ValueError('Plazos no válidos.')
    if not .5<config.service<1:raise ValueError('El servicio de ciclo debe estar entre 50% y 100%.')
    if min(config.stock,config.in_transit,config.backorders,config.elasticity_se_override)<0:raise ValueError('Stock, tránsito, atrasados y error estándar no pueden ser negativos.')
    if config.capacity is not None and (not np.isfinite(config.capacity) or config.capacity<0):raise ValueError('Capacidad no válida.')
    if config.error_policy not in ['proportional','absolute']:raise ValueError('Política de error no válida.')
    if not 100<=config.samples<=10000:raise ValueError('Cantidad de simulaciones fuera del rango permitido.')
    if config.elasticity_override is not None and not np.isfinite(config.elasticity_override):raise ValueError('Elasticidad manual no finita.')
    if config.elasticity_override is not None and config.elasticity_override>0:raise ValueError('La hipótesis de demanda debe tener elasticidad menor o igual a cero.')
    if config.elasticity_se_override>0 and (config.elasticity_override is None or config.elasticity_override==0):raise ValueError('La dispersión requiere una hipótesis de elasticidad estrictamente negativa.')
    for week in config.holiday_weeks:
        date=pd.Timestamp(week)
        if date.dayofweek!=0:raise ValueError('Las semanas festivas deben identificarse por su lunes.')

def ready_records():
    records=[]
    for file in (ROOT/'training').glob('*_result.json'):
        try:records.append(json.loads(file.read_text()))
        except (ValueError,OSError):continue
    return records

def load_registry():
    file=ROOT/'artifacts/model_registry.csv'
    if file.exists():return pd.read_csv(file)
    columns=['sku','winner','n_weeks','history_start','history_end','selection_WAPE','holdout_WAPE','holdout_MAE']
    return pd.DataFrame([{k:r[k] for k in columns} for r in ready_records()],columns=columns)
def load_manifest():
    file=ROOT/'artifacts/training_manifest.json'
    elastic=json.loads((ROOT/'artifacts/elasticity_manifest.json').read_text())
    if file.exists():manifest={**json.loads(file.read_text()),'complete':True}
    else:
        records=ready_records()
        if not records:raise ValueError('No hay modelos completos disponibles todavía.')
        manifest={'version':records[0]['version'],'source_sha256':records[0]['source_sha256'],'sku_count':len(records),'as_of':elastic['as_of'],'complete':False}
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=manifest['source_sha256']:raise ValueError('La fuente cambió: recalibra modelos y elasticidad antes de calcular.')
    if elastic['source_sha256']!=manifest['source_sha256']:raise ValueError('Elasticidad y forecast usan distintas versiones de datos.')
    response=json.loads((ROOT/'artifacts/price_scenario_manifest.json').read_text(encoding='utf8'))
    if response['source_sha256']!=manifest['source_sha256'] or response['version']!=PRICE_POLICY:raise ValueError('La respuesta semanal de precio debe recalibrarse para esta fuente/política.')
    for name,key in [('price_response.csv','response_csv_sha256'),('price_response_manifest.json','response_manifest_sha256'),('price_scenario.csv','scenario_csv_sha256')]:
        if not same_content(ROOT/'artifacts'/name,response[key]):raise ValueError('La estimación semanal cambió: ejecuta respuesta_precio.py antes de simular.')
    return manifest
def same_content(path,expected):
    """Compara la huella del archivo tolerando saltos de línea Windows/Linux: Git y
    GitHub pueden convertir CRLF↔LF al subir la app sin cambiar ningún dato."""
    raw=path.read_bytes();lf=raw.replace(b'\r\n',b'\n')
    return expected in {hashlib.sha256(x).hexdigest() for x in (raw,lf,lf.replace(b'\n',b'\r\n'))}

def load_data():return fc.load_data(DATA)
def load_elasticity():return pd.read_csv(ROOT/'artifacts/elasticity.csv')
def load_price_response():return pd.read_csv(ROOT/'artifacts/price_scenario.csv')
def load_meta(sku):return json.loads((ROOT/f'training/{sku}_result.json').read_text())
def load_model(sku,name,hist=None,params=None):
    """Modelos livianos: parámetros ya estimados en JSON (los .pkl pesaban 1.5 GB).
    SARIMAX/DHR se reconstruyen con filter(params), sin reestimar; Ridge con sus
    coeficientes y escalador guardados. El pronóstico coincide con el .pkl original."""
    if name==fc.MODEL_NAIVE:return None
    spec=json.loads((ROOT/f'models/{sku}_{NAMES.index(name)}.json').read_text())
    if name==fc.MODEL_RIDGE:return ridge_from_json(spec)
    if hist is None or params is None:hist=fc.aggregate_sku(load_data(),sku);params=load_meta(sku)['params'][name]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mod=sarimax_spec(name,hist,params)
        return mod.filter(pd.Series(spec['params'],index=spec['names']).reindex(mod.param_names).to_numpy(float))

def predict_named(sku,name,hist,future,meta):
    """Pronóstico de un modelo o de la combinación (promedio simple de sus miembros)."""
    if name==COMBO:
        members=meta['params'][COMBO]['members']
        return np.mean([predict_named(sku,m,hist,future,meta) for m in members],axis=0)
    model=load_model(sku,name,hist,meta['params'].get(name,{}))
    return predict(name,model,hist,future,meta['params'][name])

def lead_draws(config):
    rng=np.random.default_rng(config.seed+17)
    if config.lead_sd==0:return np.full(config.samples,config.lead_weeks)
    s=math.sqrt(math.log1p((config.lead_sd/config.lead_weeks)**2))
    return rng.lognormal(math.log(config.lead_weeks)-s*s/2,s,config.samples)

def required_horizon(config):
    n=max(config.horizon,math.ceil(float(lead_draws(config).max())+config.review_weeks))
    if n>104:raise ValueError('Los plazos requieren más de 104 semanas: reduce la dispersión/plazo o utiliza otro horizonte de planificación.')
    return n

def forecast_base(config,data=None):
    validate(config)
    manifest=load_manifest();meta=load_meta(config.sku)
    if meta['source_sha256']!=manifest['source_sha256']:raise ValueError('El modelo del SKU pertenece a otra versión de datos.')
    data=load_data() if data is None else data
    hist=fc.aggregate_sku(data,config.sku)
    if hist.week.iloc[-1].strftime('%Y-%m-%d')!=meta['history_end']:raise ValueError('Historia y modelos no tienen el mismo corte. Reentrena esta versión.')
    as_of=pd.Timestamp(manifest['as_of']);gap=int((as_of-hist.week.iloc[-1]).days//7)
    if gap<0:raise ValueError('El modelo contiene observaciones posteriores al corte global.')
    n=required_horizon(config)
    future=fc.future_scenario(hist,n+gap,source_df=data)
    anchor=float(hist.price_unit.tail(4).mean())
    p0=anchor if config.reference_price is None else float(config.reference_price)
    # El forecast se evalúa en su ancla histórica. Referencia y precio propuesto
    # se traducen después mediante una misma respuesta dentro de calculate().
    future['price_unit']=anchor
    future['is_holiday_week']=future.week.dt.strftime('%Y-%m-%d').isin(config.holiday_weeks).astype(int)
    name=meta['winner']
    y=predict_named(config.sku,name,hist,future,meta)
    frame=future.iloc[gap:].copy().reset_index(drop=True);frame['base_demand']=y[gap:]
    frame['anchor_demand']=frame.base_demand
    frame['horizon_step']=np.arange(1,len(frame)+1)
    if not np.isfinite(frame.base_demand).all():raise ValueError('Pronóstico no finito.')
    return frame,meta,{'as_of':as_of.strftime('%Y-%m-%d'),'history_end':meta['history_end'],'gap_weeks':gap,'price_reference':p0,'forecast_price_anchor':anchor,'base_price_policy':PRICE_BASE_POLICY,'forecast_model':name,'computed_horizon':n,'promotion_share':float(hist.promotion_flag.tail(4).mean()),'promotion_share_definition':'Media del indicador semanal de alguna promoción entre celdas, no proporción de días promocionales. Forecast original conservado.','data_version':manifest['source_sha256'],'training_version':manifest['version']}

def error_paths(errors,horizon,samples,seed):
    """Bootstrap de bloques de cuatro errores, centrados; preserva dependencia local."""
    e=np.asarray(errors,float)
    if len(e)<8 or not np.isfinite(e).all():raise ValueError('No hay suficientes errores fuera de muestra para calcular seguridad.')
    e=e-e.mean();block=min(4,len(e));rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(e)-block+1,size=(samples,math.ceil(horizon/block)))
    return e[starts[...,None]+np.arange(block)].reshape(samples,-1)[:,:horizon]

def cumulative_at(paths,durations):
    h=paths.shape[1];d=np.asarray(durations,float)
    if (d<0).any() or (d>h+1e-9).any():raise ValueError('La trayectoria no cubre todo el periodo de protección.')
    whole=np.floor(d).astype(int);fraction=d-whole
    cumul=np.column_stack([np.zeros(len(paths)),np.cumsum(paths,axis=1)])
    index=np.arange(len(paths))
    return cumul[index,whole]+fraction*paths[index,np.minimum(whole,h-1)]

def expected_at(demand,duration):
    duration=float(duration);whole=int(math.floor(duration));fraction=duration-whole
    return float(np.sum(demand[:whole])+(demand[whole]*fraction if fraction else 0))

def inventory(paths,demand,config,sigma):
    """Política de revisión periódica (R, S) con fórmulas analíticas y verificación por simulación.
    d = demanda semanal esperada (forecast), σ = RMSE semanal del error de pronóstico fuera de muestra,
    L y σL = plazo de entrega medio y su desviación (semanas), R = revisión, z = Φ⁻¹(servicio).
      SS_L = z·√(L·σ² + d_L²·σL²)          ROP = D_L + SS_L      (D_L = demanda esperada en L)
      SS   = z·√((L+R)·σ² + d_P²·σL²)      S   = D_P + SS        (D_P = demanda esperada en L+R)
      Pedido = max(0, S − posición de inventario)
    La simulación (errores reales remuestreados, plazo aleatorio e incertidumbre de elasticidad) no fija
    los parámetros: solo verifica qué servicio de ciclo se logra con ese S."""
    if len(paths)!=config.samples:raise ValueError('La cantidad de trayectorias no coincide con la simulación configurada.')
    L,sL,R=config.lead_weeks,config.lead_sd,config.review_weeks
    z=float(stats.norm.ppf(config.service))
    demand=np.asarray(demand,float)
    D_L=expected_at(demand,L);D_P=expected_at(demand,L+R)
    d_L=D_L/L;d_P=D_P/(L+R)
    ss_l=z*math.sqrt(L*sigma**2+d_L**2*sL**2)
    ss=z*math.sqrt((L+R)*sigma**2+d_P**2*sL**2)
    rop=int(math.ceil(D_L+ss_l));target_units=int(math.ceil(D_P+ss))
    lead=lead_draws(config);protection_totals=cumulative_at(paths,lead+R)
    ip=float(config.stock+config.in_transit-config.backorders)
    order=max(0,int(math.ceil(target_units-ip)))
    feasible=config.capacity is None or target_units<=config.capacity
    return {'expected_lead_demand':D_L,'expected_protection_demand':D_P,'safety_stock':ss,'safety_stock_lead':ss_l,'reorder_point':rop,'target_stock':target_units,'inventory_position':ip,'suggested_order':order,'cycle_service_target':config.service,'simulated_cycle_service':float(np.mean(protection_totals<=target_units)),'capacity_feasible':bool(feasible),'protection_weeks':L+R,'sigma_week':sigma,'z':z}

def calculate(config,base=None):
    validate(config)
    frame,meta,info=forecast_base(config) if base is None else base
    manifest=load_manifest()
    if info['data_version']!=manifest['source_sha256'] or info['training_version']!=manifest['version']:raise ValueError('El forecast base en memoria pertenece a otra versión.')
    if info.get('base_price_policy')!=PRICE_BASE_POLICY or 'anchor_demand' not in frame:raise ValueError('Recalcula el forecast con el ancla histórica de precio vigente.')
    frame=frame.copy()
    info=info.copy()
    er=load_elasticity().set_index('sku').loc[config.sku]
    pr=load_price_response().set_index('sku').loc[config.sku]
    # El coeficiente observacional se conserva como diagnóstico. No identifica
    # una intervención de precio y nunca se convierte en elasticidad causal.
    epsilon=float(pr.elasticity_used) if config.elasticity_override is None else float(config.elasticity_override)
    if epsilon>0 or not np.isfinite(epsilon):raise ValueError('La respuesta del escenario debe ser finita y no positiva.')
    automatic=config.elasticity_override is None
    se=float(pr.scenario_elasticity_sd) if automatic else config.elasticity_se_override
    conditional=price_distribution(pr.epsilon_corrected_free,pr.corrected_se) if automatic else None
    ratio=1+config.price_change_pct/100.;factor=float(ratio**epsilon)
    if not np.isfinite(factor) or factor>1e6:raise ValueError('El escenario produce un factor de demanda extremo.')
    p0=info['price_reference'];p1=p0*ratio;anchor=info['forecast_price_anchor']
    rebase_factor=float((p0/anchor)**epsilon)
    if not np.isfinite(rebase_factor) or rebase_factor>1e6:raise ValueError('La referencia produce un factor de demanda extremo.')
    q_anchor=frame.anchor_demand.to_numpy()
    frame['base_demand']=q_anchor*rebase_factor
    frame['scenario_demand']=frame.base_demand*factor
    frame['scenario_price']=p1;frame['base_revenue']=frame.base_demand*p0;frame['scenario_revenue']=frame.scenario_demand*p1
    errors=pd.read_csv(ROOT/f'training/{config.sku}_predictions.csv')
    errors=errors[errors.model.eq(meta['winner'])].sort_values('week').residual.to_numpy()
    shocks=error_paths(errors,len(frame),config.samples,config.seed)
    q=frame.base_demand.to_numpy()
    rng=np.random.default_rng(config.seed+31)
    eps_samples=np.full(config.samples,epsilon)
    if config.include_elasticity_uncertainty and se>0:
        if automatic:
            eps_samples=conditional.rvs(size=config.samples,random_state=rng)
        else:
            # Magnitud positiva para la hipótesis manual, con media/desviación
            # explícitas del usuario; su signo no proviene de una estimación.
            sigma=math.sqrt(math.log1p((se/epsilon)**2))
            eps_samples=-rng.lognormal(math.log(-epsilon)-sigma*sigma/2,sigma,config.samples)
    log_factors=np.log(ratio)*eps_samples
    log_rebase=np.log(p0/anchor)*eps_samples
    if not np.isfinite(log_factors+log_rebase).all() or (np.abs(log_factors+log_rebase)>20).any() or (np.abs(log_rebase)>20).any():raise ValueError('La incertidumbre de elasticidad produce escenarios extremos; reduce el cambio, la referencia o la dispersión manual.')
    factors=np.exp(log_factors)
    ref_factors=np.exp(log_rebase)
    if config.error_policy=='proportional':
        base_paths=np.maximum(q_anchor+shocks,0.)*ref_factors[:,None]
        scenario_paths=base_paths*factors[:,None]
    else:
        base_paths=np.maximum(q_anchor*ref_factors[:,None]+shocks,0.)
        scenario_paths=np.maximum(q_anchor*(ref_factors*factors)[:,None]+shocks,0.)
    for prefix,paths in [('base',base_paths),('scenario',scenario_paths)]:
        frame[f'{prefix}_low']=np.quantile(paths,.05,axis=0)
        frame[f'{prefix}_high']=np.quantile(paths,.95,axis=0)
    # σ semanal = RMSE de los errores fuera de muestra del modelo campeón (selección + comprobación).
    # Con error proporcional se escala con el nivel de demanda del escenario.
    sigma=float(np.sqrt(np.mean(errors**2)))
    scale=(lambda f:f) if config.error_policy=='proportional' else (lambda f:1.)
    inventories=pd.DataFrame([{'scenario':s,**inventory(paths,demand,config,sigma*scale(f))} for s,paths,demand,f in [('Base',base_paths,q,rebase_factor),('Precio propuesto',scenario_paths,frame.scenario_demand.to_numpy(),rebase_factor*factor)]])
    visible=frame.iloc[:config.horizon].copy()
    warnings=[]
    if automatic:
        warnings.append('Respuesta calibrada con el ajuste semanal y su incertidumbre, condicionada al supuesto de demanda decreciente. El signo se impone como hipótesis económica; no es un efecto causal demostrado.')
        if not bool(pr.price_supported):warnings.append('La evidencia de precio es débil: el criterio confirmatorio y predictivo no se cumple. El escenario conserva una sensibilidad condicional; esto no confirma su magnitud en la realidad.')
    else:warnings.append('Sensibilidad bajo una elasticidad supuesta por el usuario; su magnitud no fue identificada por esta base.')
    if config.reference_price is not None:warnings.append('La referencia manual se compara con el ancla histórica usando la misma respuesta de precio. No se introduce como intervención en los regresores del forecast.')
    if p1<er.price_min or p1>er.price_max or p0<er.price_min or p0>er.price_max:warnings.append('Precio fuera del rango histórico: extrapolación.')
    if config.horizon>12 or info['computed_horizon']>12:warnings.append('Hay semanas más allá del horizonte validado de doce; errores extrapolados por bloques.')
    if info['gap_weeks']:warnings.append(f"Este SKU tiene {info['gap_weeks']} semana(s) sin observación completa al corte global; se pronostica el puente sin inventar ventas históricas.")
    if meta['n_weeks']<104:warnings.append('Historia menor a dos ciclos anuales completos.')
    summary={'base_units':float(visible.base_demand.sum()),'scenario_units':float(visible.scenario_demand.sum()),'base_revenue':float(visible.base_revenue.sum()),'scenario_revenue':float(visible.scenario_revenue.sum()),'demand_change_pct':(factor-1)*100,'revenue_change_pct':(ratio*factor-1)*100,'elasticity':epsilon,'elasticity_se':se,'price_reference':p0,'price_proposed':p1}
    if automatic:
        bounds=np.sort((ratio**conditional.ppf([.05,.95])-1)*100)
        summary.update(demand_change_low_pct=float(bounds[0]),demand_change_high_pct=float(bounds[1]))
    info.update({'price_policy_version':PRICE_POLICY,'price_response_status':str(pr.status) if automatic else 'hipótesis explícita','elasticity_source':'Sensibilidad condicionada: ajuste semanal libre y supuesto de signo' if automatic else 'Supuesto manual','scenario_elasticity_low':float(pr.scenario_elasticity_low) if automatic else None,'scenario_elasticity_high':float(pr.scenario_elasticity_high) if automatic else None,'observational_elasticity':float(er.elasticity),'elasticity_p_cluster':float(er.p_cluster),'elasticity_ci_low':float(er.ci_low),'elasticity_ci_high':float(er.ci_high),'corrected_elasticity_free':float(pr.epsilon_corrected_free),'corrected_elasticity_ci_low':float(pr.corrected_ci_low),'corrected_elasticity_ci_high':float(pr.corrected_ci_high),'price_supported':bool(pr.price_supported),'price_validation_gain_pct':float(pr.validation_price_gain_pct),'error_points':len(errors),'error_policy':config.error_policy,'elasticity_uncertainty_included':bool(config.include_elasticity_uncertainty and se>0),'uncertainty_method':'Bootstrap de bloques 4; epsilon automática: aproximación normal condicionada a no positiva; epsilon manual: lognormal negativa subjetiva. No intervalo causal.','seed':config.seed,'samples':config.samples})
    # Distribuciones simuladas (para los gráficos de inventario): demanda durante el
    # plazo de entrega y durante la protección (plazo + revisión), base y escenario.
    lead=lead_draws(config)
    dist={s:{'lead':cumulative_at(p,lead),'protection':cumulative_at(p,lead+config.review_weeks)} for s,p in [('Base',base_paths),('Precio propuesto',scenario_paths)]}
    return {'config':asdict(config),'info':info,'summary':summary,'weekly':visible,'protection_forecast':frame,'inventory':inventories,'warnings':warnings,'distributions':dist}

def serialization(result):
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'configuration':result['config'],'info':result['info'],'summary':result['summary'],'warnings':result['warnings'],'data_version':result['info']['data_version'],'training_version':result['info']['training_version'],'scope':'Simulación académica de ventas e inventario; no causalidad ni compra enviada.'}
    tables={'weekly_forecast':result['weekly'],'protection_forecast':result['protection_forecast'],'inventory':result['inventory']}
    content={name+'.csv':table.to_csv(index=False).encode('utf-8-sig') for name,table in tables.items()}
    content['manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False).encode('utf8')
    return content

def bundle(result):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in serialization(result).items():archive.writestr(name,content)
    return buf.getvalue()

def save(result,name='Escenario'):
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    destination=ROOT/'runs'/run_id;destination.mkdir(parents=True)
    content=serialization(result)
    manifest=json.loads(content['manifest.json']);manifest.update({'run_id':run_id,'name':str(name)[:100]})
    content['manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf8')
    for filename,data in content.items():(destination/filename).write_bytes(data)
    return destination

def list_runs():
    records=[]
    for file in sorted((ROOT/'runs').glob('*/manifest.json'),reverse=True):
        try:
            m=json.loads(file.read_text(encoding='utf8'));records.append({'run_id':m['run_id'],'name':m['name'],'sku':m['configuration']['sku'],'created_at':m['created_at'],'price_change_pct':m['configuration']['price_change_pct']})
        except (KeyError,ValueError):continue
    return records

def restore(run_id):
    if not isinstance(run_id,str) or Path(run_id).name!=run_id:raise ValueError('Identificador no válido.')
    path=ROOT/'runs'/run_id
    m=json.loads((path/'manifest.json').read_text(encoding='utf8'))
    return {'config':m['configuration'],'info':m['info'],'summary':m['summary'],'warnings':m['warnings'],'weekly':pd.read_csv(path/'weekly_forecast.csv',parse_dates=['week']),'protection_forecast':pd.read_csv(path/'protection_forecast.csv',parse_dates=['week']),'inventory':pd.read_csv(path/'inventory.csv')}

def export_portfolio(portfolio):
    """Publicación local atómica: puntero a una ejecución completa para BI."""
    if portfolio.get('price_policy_version')!=PRICE_POLICY or portfolio.get('base_price_policy')!=PRICE_BASE_POLICY:raise ValueError('Recalcula el conjunto con la política y ancla de precio vigentes antes de exportar.')
    for configuration in portfolio['configurations']:validate(Scenario(**configuration))
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    root=ROOT/'exports';destination=root/run_id;destination.mkdir(parents=True)
    for table in ['summary','weekly','inventory']:
        portfolio[table].assign(run_id=run_id).to_csv(destination/f'{table}.csv',index=False,encoding='utf-8-sig')
    manifest={'run_id':run_id,'name':'Escenarios de treinta SKU','as_of':load_manifest()['as_of'],'data_version':load_manifest()['source_sha256'],'training_version':load_manifest()['version'],'price_policy_version':PRICE_POLICY,'base_price_policy':PRICE_BASE_POLICY,'configurations':portfolio['configurations'],'sku_count':int(portfolio['summary'].sku.nunique()),'scope':'Sensibilidad calibrada bajo supuesto de signo o hipótesis manual explícita; efecto causal no demostrado; stock cero si no fue capturado.'}
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    pointer=root/'latest.json';temporary=root/f'.{run_id}.json'
    temporary.write_text(json.dumps({'run_id':run_id},indent=2),encoding='utf8');temporary.replace(pointer)
    return destination
