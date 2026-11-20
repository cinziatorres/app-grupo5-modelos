"""Modelos semanales compartidos, sin dependencias de la interfaz."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import forecast_core as fc

# Fuente y núcleo de forecast dentro de la carpeta de la app (antes estaban en
# ../anexos y ../forecast-30-skus, lo que impedía publicarla como web).
DATA=ROOT/'data'/'weekly_df_final_for_modeling.xlsx'
COMBO='Combinación (promedio)'
NAMES=list(fc.MODEL_NAMES)

def features(frame,start,params):
    x=fc.add_fourier_features(frame,start,k=int(params.get('fourier_k',2)))
    if not params.get('quadratic',True):x=x.drop(columns='t2')
    return x

def fit(name,hist,params):
    if name==fc.MODEL_NAIVE:return None
    if name==fc.MODEL_RIDGE:
        return make_pipeline(StandardScaler(),Ridge(alpha=float(params['ridge_alpha']))).fit(features(hist,hist.week.iloc[0],params),hist.units_sold)
    if name==fc.MODEL_DHR:return fc.fit_dhr_arma(hist,k=int(params['fourier_k']),order=params['dhr_order'])
    return fc.fit_seasonal_sarimax(hist,order=params['sarimax_order'],seasonal_order=params['sarimax_seasonal_order'])

def predict(name,model,hist,future,params):
    if name==fc.MODEL_NAIVE:return np.maximum(fc.seasonal_naive_forecast(hist,len(future)),0)
    if name==fc.MODEL_RIDGE:return np.maximum(np.asarray(model.predict(features(future,hist.week.iloc[0],params)),float),0)
    x=fc.add_fourier_features(future,hist.week.iloc[0],k=int(params['fourier_k'])) if name==fc.MODEL_DHR else fc.add_sarimax_exog(future)
    return fc._forecast_sarimax_result(model,len(future),x)

def sarimax_spec(name,hist,params):
    """Misma especificación usada al entrenar (forecast_core), sin estimar."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y=hist.units_sold.astype(float)
    if name==fc.MODEL_DHR:
        return SARIMAX(y,exog=fc.add_fourier_features(hist,hist.week.iloc[0],k=int(params['fourier_k'])),order=tuple(params['dhr_order']),seasonal_order=(0,0,0,0),trend='c',enforce_stationarity=False,enforce_invertibility=False)
    return SARIMAX(y,exog=fc.add_sarimax_exog(hist),order=tuple(params['sarimax_order']),seasonal_order=tuple(params['sarimax_seasonal_order']),trend='ct',enforce_stationarity=False,enforce_invertibility=False)

def ridge_from_json(spec):
    """Reconstruye el pipeline StandardScaler+Ridge guardado sin volver a ajustarlo."""
    scaler=StandardScaler();scaler.mean_=np.asarray(spec['mean'],float);scaler.scale_=np.asarray(spec['scale'],float)
    scaler.var_=scaler.scale_**2;scaler.n_features_in_=len(scaler.mean_);scaler.n_samples_seen_=int(spec.get('n',1))
    scaler.feature_names_in_=np.asarray(spec['features'],dtype=object)
    ridge=Ridge(alpha=float(spec['alpha']));ridge.coef_=np.asarray(spec['coef'],float);ridge.intercept_=float(spec['intercept'])
    ridge.n_features_in_=len(ridge.coef_)
    return make_pipeline(scaler,ridge)

def diagnostics(model):
    if model is None or not hasattr(model,'arroots'):return {'eligible':True,'reason':'ok'}
    ar=np.abs(np.asarray(model.arroots));ma=np.abs(np.asarray(model.maroots))
    flags={'converged':fc._is_converged(model),'finite':bool(np.isfinite(model.params).all()),'stationary':bool((ar>1+1e-6).all()),'invertible':bool((ma>1+1e-6).all())}
    return {**flags,'eligible':all(flags.values()),'reason':';'.join(k for k,v in flags.items() if not v) or 'ok'}

def origin_scenario(hist,target,source):
    """Precio/promoción conocidos desde origen; calendario conocido de la fuente."""
    future=target[['week','is_summer','is_winter','is_holiday_week']].copy().reset_index(drop=True)
    future['price_unit']=float(hist.price_unit.tail(4).mean())
    future['promotion_flag']=float(hist.promotion_flag.tail(4).mean())
    return future

