"""Reestima y valida respuesta de precio sin sustituir el Excel del curso."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from modelos import ROOT,DATA,fc
from elasticidad_semanal import KEY,VERSION,build,fit,predict,poisson_deviance

def main():
    diagnostic=ROOT/'diagnostico-precio'
    archive=diagnostic/'fuentes-publicas/dataset-v16.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!='c2ad83e0964513932fbe5ee4f05669567adeed95ac901dd0b3c678bda716ca21':raise ValueError('Fuente diaria distinta de la versión v16 verificada; no se sustituye la base.')
    source=fc.load_data(DATA)
    with zipfile.ZipFile(archive) as z:
        raw=z.read('FMCG_2022_2024.csv')
        daily=pd.read_csv(z.open('FMCG_2022_2024.csv'),parse_dates=['date'])
    daily['week']=daily.date.dt.to_period('W').dt.start_time
    assert not daily.duplicated(['sku','date','channel','region']).any()
    assert daily.loc[daily.stock_available.eq(0),'units_sold'].eq(0).all()
    reconstructed=daily.groupby(KEY).units_sold.sum().reset_index()
    check=source[KEY+['units_sold']].merge(reconstructed,on=KEY,suffixes=('','_daily'),validate='one_to_one')
    assert len(check)==len(source) and np.allclose(check.units_sold,check.units_sold_daily)
    rows=[];folds=[];holdouts=[];parameters=[]
    for sku,g in source.groupby('sku'):
        hist=fc.aggregate_sku(source,sku);g=g[g.week.between(hist.week.min(),hist.week.max())].copy()
        weeks=sorted(g.week.unique());selection_end=weeks[-13]
        pre=g[g.week<=selection_end];design_pre=build(pre,daily[daily.sku.eq(sku)])
        prefree=fit(design_pre,'free');a=design_pre.X.shape[1];b=prefree['epsilon'];se=float(np.sqrt(prefree['covariance'][a,a]));G=prefree['weeks'];critical=t.ppf(.975,G-1)
        for end in [len(weeks)-24,len(weeks)-12]:
            cutoff=weeks[end-13]
            train=g[g.week<=cutoff];test=g[g.week.isin(weeks[end-12:end])]
            td=build(train,daily[daily.sku.eq(sku)]);vd=build(test,daily[daily.sku.eq(sku)])
            for mode in ['no_price','nonpositive']:
                result=fit(td,mode);mu=predict(result,vd)
                folds.append({'sku':sku,'validation_end':str(pd.Timestamp(weeks[end-1]).date()),'mode':mode,'epsilon_training':result['epsilon'],'deviance':poisson_deviance(vd.weekly.units_sold,mu),'WAPE':float(100*abs(vd.weekly.units_sold.to_numpy()-mu).sum()/vd.weekly.units_sold.sum()),'observations':len(vd.weekly)})
        vd=pd.DataFrame([r for r in folds if r['sku']==sku])
        mean_dev=vd.groupby('mode').deviance.sum()
        rows.append({'sku':sku,'epsilon_preholdout_free':b,'preholdout_se':se,'preholdout_ci_low':b-critical*se,'preholdout_ci_high':b+critical*se,'p_preholdout':float(2*t.sf(abs(b/se),G-1)),'validation_deviance_without_price':float(mean_dev['no_price']),'validation_deviance_with_price':float(mean_dev['nonpositive']),'validation_price_gain_pct':float(100*(1-mean_dev['nonpositive']/mean_dev['no_price'])),'folds_negative':bool((vd.loc[vd['mode'].eq('nonpositive'),'epsilon_training']<-1e-6).all()),'history_end':str(hist.week.max().date()),'n_weekly_cells':len(g)})
        print('Validado '+sku,flush=True)
    response=pd.DataFrame(rows)
    response['p_preholdout_holm']=multipletests(response.p_preholdout,method='holm')[1]
    # Regla explícita, sin elegir modelos por conseguir un signo conveniente.
    # Inferencia y dos validaciones anteriores al último bloque H12.
    response['price_supported']=response.preholdout_ci_high.lt(0)&response.p_preholdout_holm.lt(.05)&response.validation_price_gain_pct.gt(0)&response.folds_negative
    for i,row in response.iterrows():
        sku=row.sku;g=source[source.sku.eq(sku)];hist=fc.aggregate_sku(source,sku);g=g[g.week.between(hist.week.min(),hist.week.max())]
        weeks=sorted(g.week.unique());ds=daily[daily.sku.eq(sku)]
        mode='nonpositive' if row.price_supported else 'no_price'
        train=build(g[g.week<weeks[-12]],ds);test=build(g[g.week>=weeks[-12]],ds)
        selected=fit(train,mode);mu=predict(selected,test)
        for week,actual,forecast in zip(test.weekly.week,test.weekly.units_sold,mu):holdouts.append({'sku':sku,'week':str(week.date()),'actual':float(actual),'predicted':float(forecast),'selected_mode':mode})
        full=build(g,ds);free=fit(full,'free');constrained=fit(full,'nonpositive');final=fit(full,mode)
        a=full.X.shape[1];se=float(np.sqrt(free['covariance'][a,a]));crit=t.ppf(.975,free['weeks']-1)
        updates={'selected_model':mode,'elasticity_used':final['epsilon'],'epsilon_corrected_free':free['epsilon'],'epsilon_corrected_nonpositive':constrained['epsilon'],'corrected_se':se,'corrected_ci_low':free['epsilon']-crit*se,'corrected_ci_high':free['epsilon']+crit*se,'holdout_deviance':poisson_deviance(test.weekly.units_sold,mu),'holdout_WAPE':float(100*abs(test.weekly.units_sold.to_numpy()-mu).sum()/test.weekly.units_sold.sum()),'status':'asociación predictiva con restricción económica; causalidad no demostrada' if row.price_supported else 'respuesta al precio no respaldada; modelo seleccionado sin precio','converged':True,'gradient_max':max(free['gradient_max'],constrained['gradient_max'],final['gradient_max'])}
        for name,value in updates.items():response.loc[i,name]=value
        for p,value in zip(final['columns'],final['parameters']):parameters.append({'sku':sku,'parameter':p,'coefficient':float(value),'selected_model':mode})
    response['price_supported']=response.price_supported.astype(bool)
    response['converged']=response.converged.astype(bool)
    artifacts=ROOT/'artifacts'
    response.to_csv(artifacts/'price_response.csv',index=False)
    pd.DataFrame(parameters).to_csv(artifacts/'price_response_coefficients.csv',index=False)
    pd.DataFrame(folds).to_csv(diagnostic/'validacion-respuesta-semanal.csv',index=False)
    pd.DataFrame(holdouts).to_csv(diagnostic/'holdout-respuesta-semanal.csv',index=False)
    manifest={'version':VERSION,'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'daily_sha256':hashlib.sha256(raw).hexdigest(),'public_zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'dataset_version':16,'sku_count':len(response),'as_of':max(response.history_end),'weekly_source_unchanged':True,'model':'Media Poisson de unidades semanales; suma exacta de exposiciones diarias con disponibilidad, precio, promoción, pack y día; tendencia/Fourier anual K2 y efectos de canal×región. Sin stock/plazo contemporáneos como regresores.','constraint':'epsilon <= 0 en modelo no positivo, ajustando conjuntamente todos los parámetros; no inversión de signo.','selection':'Sin precio o no positivo. Dos bloques H12 anteriores al holdout H12; efecto negativo preholdout con Holm30, pendiente negativa en ambos orígenes y mejora de deviance; selección independiente del holdout final.','zero_meaning':'Modelo sin precio seleccionado: no efecto respaldado con estos datos. No demuestra demanda real inelástica; no identifica efectos causales.','supported_sku':response.loc[response.price_supported,'sku'].to_list(),'scope':'Respuesta predictiva bajo restricción económica. Disponibilidad describe ventas observadas; no reconstruye demanda perdida ni imputa días ausentes.'}
    (artifacts/'price_response_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    from respuesta_precio import prepare
    prepare()
    print(response[['sku','epsilon_corrected_free','epsilon_corrected_nonpositive','validation_price_gain_pct','price_supported','elasticity_used']].to_string(index=False),flush=True)

if __name__=='__main__':main()
