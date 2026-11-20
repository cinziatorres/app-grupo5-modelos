"""Calibración, selección H=12 y comprobación H=12 separadas para treinta SKU.
Reanudable por SKU; originales sin modificaciones. Dos procesos, un hilo BLAS.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import concurrent.futures,hashlib,json,pickle,time,traceback
import numpy as np
import pandas as pd
from modelos import ROOT,DATA,fc,fit,predict,diagnostics,origin_scenario,NAMES

ART=ROOT/'artifacts';MODELS=ROOT/'models';CHECK=ROOT/'training'
for path in [ART,MODELS,CHECK]:path.mkdir(exist_ok=True)
VERSION='integrado-h12-exante-v1'

def status(sku,phase,**extra):
    (CHECK/f'{sku}_status.json').write_text(json.dumps({'sku':sku,'phase':phase,'time':time.time(),**extra},indent=2),encoding='utf8')
    print(f'{sku}: {phase}',flush=True)

def score(actual,pred):return float(fc.wape(actual,pred)),float(fc.mae(actual,pred))

def train(sku):
    checksum=hashlib.sha256(DATA.read_bytes()).hexdigest()
    receipt=CHECK/f'{sku}_result.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get('version')==VERSION and saved.get('source_sha256')==checksum:return sku
    try:
        status(sku,'calibrating')
        data=fc.load_data(DATA);hist=fc.aggregate_sku(data,sku)
        assert len(hist)>=80,'Se necesitan 52+4+12+12 semanas.'
        dev=hist.iloc[:-24].copy();cal_train=dev.iloc[:-4].copy();cal_target=dev.iloc[-4:]
        future=origin_scenario(cal_train,cal_target,data)
        grids={
            fc.MODEL_NAIVE:[{}],
            fc.MODEL_RIDGE:[{'fourier_k':k,'quadratic':q,'ridge_alpha':a} for k in [1,2,3] for q in [False,True] for a in [.1,1.,10.,100.]],
            fc.MODEL_DHR:[{'fourier_k':2,'dhr_order':o} for o in [(1,0,0),(0,0,1),(1,0,1)]],
            fc.MODEL_SARIMAX:[{'sarimax_order':o,'sarimax_seasonal_order':s} for o in [(1,0,0),(1,0,1)] for s in [(1,0,0,52),(0,0,1,52)]],
        }
        params={};calibration=[]
        for name,grid in grids.items():
            options=[]
            for config in grid:
                try:
                    model=fit(name,cal_train,config);diag=diagnostics(model)
                    y=predict(name,model,cal_train,future,config);wape,mae=score(cal_target.units_sold,y)
                    row={'sku':sku,'model':name,'config':json.dumps(config),'WAPE':wape,'MAE':mae,**diag}
                    options.append((not diag['eligible'],wape,mae,config));calibration.append(row)
                except Exception as exc:
                    calibration.append({'sku':sku,'model':name,'config':json.dumps(config),'eligible':False,'reason':str(exc)})
            if options:params[name]=sorted(options,key=lambda x:x[:3])[0][3]
        status(sku,'selection_and_holdout')
        metrics=[];predictions=[];val_flags={}
        for scope,train_hist,target in [('selection',hist.iloc[:-24],hist.iloc[-24:-12]),('holdout',hist.iloc[:-12],hist.iloc[-12:])]:
            future=origin_scenario(train_hist,target,data)
            for name in NAMES:
                try:
                    config=params[name];model=fit(name,train_hist,config);diag=diagnostics(model)
                    y=predict(name,model,train_hist,future,config)
                    if not np.isfinite(y).all():raise ValueError('Predicciones no finitas')
                    if scope=='selection':val_flags[name]=diag['eligible']
                    metrics.append({'sku':sku,'model':name,'scope':scope,'WAPE':fc.wape(target.units_sold,y),'MAE':fc.mae(target.units_sold,y),'bias':float(np.mean(target.units_sold.to_numpy()-y)),**diag})
                    for h,(week,actual,value) in enumerate(zip(target.week,target.units_sold,y),1):
                        predictions.append({'sku':sku,'model':name,'scope':scope,'origin_week':train_hist.week.iloc[-1].strftime('%Y-%m-%d'),'week':week.strftime('%Y-%m-%d'),'horizon_step':h,'actual':float(actual),'prediction':float(value),'residual':float(actual-value)})
                except Exception as exc:
                    metrics.append({'sku':sku,'model':name,'scope':scope,'eligible':False,'reason':str(exc)})
                    if scope=='selection':val_flags[name]=False
        status(sku,'fitting_final')
        final_flags={}
        for name in NAMES:
            try:
                model=fit(name,hist,params[name]);diag=diagnostics(model);final_flags[name]=diag
                if model is not None:
                    with (MODELS/f'{sku}_{NAMES.index(name)}.pkl').open('wb') as stream:pickle.dump(model,stream)
            except Exception as exc:final_flags[name]={'eligible':False,'reason':str(exc)}
        held_flags={r['model']:r['eligible'] for r in metrics if r['scope']=='holdout'}
        eligible=[r for r in metrics if r['scope']=='selection' and val_flags.get(r['model'],False) and final_flags[r['model']]['eligible'] and held_flags.get(r['model'],False) and np.isfinite(r.get('WAPE',np.nan))]
        winner=sorted(eligible,key=lambda r:(r['WAPE'],r['MAE']))[0]
        hold=next(r for r in metrics if r['model']==winner['model'] and r['scope']=='holdout')
        if not hold['eligible']:raise ValueError('El ganador no tiene comprobación final numéricamente válida.')
        result={'version':VERSION,'source_sha256':checksum,'sku':sku,'winner':winner['model'],'params':params,'history_start':hist.week.iloc[0].strftime('%Y-%m-%d'),'history_end':hist.week.iloc[-1].strftime('%Y-%m-%d'),'n_weeks':len(hist),'selection_WAPE':float(winner['WAPE']),'holdout_WAPE':float(hold['WAPE']),'holdout_MAE':float(hold['MAE']),'final_diagnostics':final_flags,'calibration_end':dev.week.iloc[-1].strftime('%Y-%m-%d'),'selection_end':hist.iloc[-13].week.strftime('%Y-%m-%d'),'holdout_end':hist.week.iloc[-1].strftime('%Y-%m-%d')}
        pd.DataFrame(calibration).to_csv(CHECK/f'{sku}_calibration.csv',index=False)
        pd.DataFrame(metrics).to_csv(CHECK/f'{sku}_metrics.csv',index=False)
        pd.DataFrame(predictions).to_csv(CHECK/f'{sku}_predictions.csv',index=False)
        receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        status(sku,'complete',winner=winner['model'],holdout_WAPE=result['holdout_WAPE'])
        return sku
    except Exception:
        status(sku,'failed',error=traceback.format_exc());raise

def main():
    skus=sorted(fc.load_data(DATA).sku.unique());assert len(skus)==30
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(train,skus))
    records=[json.loads((CHECK/f'{s}_result.json').read_text()) for s in skus]
    pd.DataFrame([{k:r[k] for k in ['sku','winner','n_weeks','history_start','history_end','selection_WAPE','holdout_WAPE','holdout_MAE']} for r in records]).to_csv(ART/'model_registry.csv',index=False)
    for table in ['calibration','metrics','predictions']:
        pd.concat([pd.read_csv(CHECK/f'{s}_{table}.csv') for s in skus],ignore_index=True).to_csv(ART/f'{table}.csv',index=False)
    manifest={'version':VERSION,'source_sha256':records[0]['source_sha256'],'sku_count':30,'as_of':max(r['history_end'] for r in records),'calibration_horizon':4,'selection_horizon':12,'holdout_horizon':12,'exogenous_policy':'Precio/promoción últimos cuatro datos disponibles al origen; calendario de prueba tomado como programado y conocido.','winner_policy':'WAPE/MAE de selección entre candidatos estables en selección, comprobación y ajuste final; el error holdout no determina ganador. La comprobación de estabilidad en holdout restringe admisibilidad.'}
    (ART/'training_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('TRAINING COMPLETE: treinta SKU, selección y comprobación H12.',flush=True)

if __name__=='__main__':main()
