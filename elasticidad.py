"""OLS log-log del panel con controles, sin duplicar forecast o estacionalidad."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

CONTROLS=['promotion_flag','stock_available','delivery_days','is_holiday_peak','is_holiday_week','is_summer','is_winter']

def estimate(panel):
    x=panel.copy()
    cols=['units_sold','price_unit',*CONTROLS]
    for c in cols:x[c]=pd.to_numeric(x[c],errors='coerce').astype(float)
    x=x[(x.units_sold>0)&(x.price_unit>0)].dropna(subset=cols+['week','channel','region'])
    X=x[CONTROLS].copy();X.insert(0,'ln_price',np.log(x.price_unit))
    X=pd.concat([X,pd.get_dummies(x[['channel','region']].astype(str),drop_first=True,dtype=float)],axis=1)
    X=sm.add_constant(X,has_constant='add').astype(float)
    if len(x)<30 or x.price_unit.nunique()<2:raise ValueError('Diseño OLS sin identificación numérica suficiente.')
    # Mantener precio/constante y retirar solo columnas exactamente redundantes.
    ordered=['const','ln_price',*[c for c in X.columns if c not in ['const','ln_price']]]
    keep=[];dropped=[]
    for column in ordered:
        candidate=X[keep+[column]]
        scaled=candidate.copy()
        for c in scaled.columns:
            if c!='const':scaled[c]=(scaled[c]-scaled[c].mean())/(scaled[c].std(ddof=0) or 1.)
        if np.linalg.matrix_rank(scaled.to_numpy())>len(keep):keep.append(column)
        else:dropped.append(column)
    if 'ln_price' not in keep:raise ValueError('Precio sin variación identificable.')
    X=X[keep]
    result=sm.OLS(np.log(x.units_sold),X).fit()
    groups=x.week.factorize()[0]
    robust=result.get_robustcov_results(cov_type='cluster',groups=groups,use_correction=True,df_correction=True,use_t=True)
    i=list(X.columns).index('ln_price');lo,hi=robust.conf_int()[i]
    row={'sku':str(x.sku.iloc[0]),'elasticity':float(result.params['ln_price']),'se_cluster':float(robust.bse[i]),'ci_low':float(lo),'ci_high':float(hi),'p_cluster':float(robust.pvalues[i]),'p_ols':float(result.pvalues['ln_price']),'n':int(result.nobs),'weeks':int(x.week.nunique()),'r2':float(result.rsquared),'price_min':float(x.price_unit.min()),'price_max':float(x.price_unit.max()),'dropped_redundant':','.join(dropped),'covariance':'Agrupación por semana; no corrige toda dependencia entre semanas ni identifica causalidad.'}
    coefficients=pd.DataFrame({'sku':row['sku'],'parameter':X.columns,'coefficient':result.params.to_numpy(),'se_cluster':robust.bse,'p_cluster':robust.pvalues})
    return row,coefficients

