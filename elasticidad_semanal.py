"""Media de ventas SEMANALES, conservando precios/promociones diarios.

mu_semana = exp(X beta) * sum_dia_disponible exp(Z_dia theta).
El primer parámetro de theta es epsilon sobre ln(precio/5.25).
La restricción epsilon <= 0 es económica; no identifica causalidad.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import xlogy

KEY=['sku','week','channel','region']
VERSION='precio-semanal-exposicion-v3'

@dataclass
class Design:
    weekly:pd.DataFrame
    X:np.ndarray
    Z:np.ndarray
    cell:np.ndarray
    columns:list
    daily_columns:list

def build(weekly,daily):
    w=weekly.sort_values(KEY).reset_index(drop=True).copy()
    d=daily[daily.stock_available.gt(0)].merge(w[KEY].assign(cell=np.arange(len(w))),on=KEY,how='inner',validate='many_to_one')
    counts=np.bincount(d.cell.to_numpy(int),minlength=len(w))
    if (counts==0).any():
        if (w.loc[counts==0,'units_sold']!=0).any():raise ValueError('Ventas positivas sin exposición disponible.')
        return build(w.loc[counts>0],daily)
    tau=(w.week-pd.Timestamp('2022-01-01')).dt.days/365.25
    x=pd.DataFrame({'const':1.,'trend':tau})
    for k in [1,2]:x[f'sin{k}']=np.sin(2*np.pi*k*tau);x[f'cos{k}']=np.cos(2*np.pi*k*tau)
    cells=pd.Categorical(w.channel+'|'+w.region,categories=sorted((weekly.channel+'|'+weekly.region).unique()))
    x=pd.concat([x,pd.get_dummies(cells,prefix='cell',drop_first=True,dtype=float)],axis=1)
    z=pd.DataFrame({'ln_price':np.log(d.price_unit/5.25),'promotion':d.promotion_flag.astype(float)})
    for name,values,categories in [('pack',d.pack_type,sorted(daily.pack_type.unique())),('weekday',d.date.dt.dayofweek,list(range(7)))]:
        z=pd.concat([z,pd.get_dummies(pd.Categorical(values,categories=categories),prefix=name,drop_first=True,dtype=float)],axis=1)
    return Design(w,x.to_numpy(float),z.to_numpy(float),d.cell.to_numpy(int),x.columns.to_list(),z.columns.to_list())

def mean_jacobian(parameters,design):
    a=design.X.shape[1];eta=design.Z@parameters[a:]
    # Un único máximo conserva las sumas por semana sin grandes exponentes.
    shift=float(eta.max());v=np.exp(eta-shift)
    total=np.bincount(design.cell,weights=v,minlength=len(design.weekly))
    logmu=design.X@parameters[:a]+np.log(total)+shift
    if (abs(logmu)>600).any():raise FloatingPointError('Media fuera del dominio numérico.')
    mu=np.exp(logmu)
    weights=v/total[design.cell]
    zmean=np.column_stack([np.bincount(design.cell,weights=weights*design.Z[:,j],minlength=len(mu)) for j in range(design.Z.shape[1])])
    return mu,np.column_stack([design.X,zmean]),weights

def fit(design,mode='free'):
    if mode not in ['free','nonpositive','no_price']:raise ValueError('Modo desconocido.')
    a=design.X.shape[1];k=a+design.Z.shape[1]
    y=design.weekly.units_sold.to_numpy(float);p=np.zeros(k)
    p[0]=np.log(max(y.sum(),1)/len(design.cell));p[a+1]=.5
    active=np.array([j for j in range(k) if mode!='no_price' or j!=a])
    def objective(q):
        full=p.copy();full[active]=q
        mu,J,_=mean_jacobian(full,design)
        scale=max(y.sum(),1.)
        return float(np.sum(mu-xlogy(y,mu))/scale),(J.T@(mu-y)/scale)[active]
    bounds=[(None,0.) if mode=='nonpositive' and j==a else (None,None) for j in active]
    r=minimize(objective,p[active],jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':800,'ftol':1e-14,'gtol':1e-8,'maxls':40})
    p[active]=r.x;mu,J,weights=mean_jacobian(p,design)
    grad=objective(r.x)[1]
    projected=grad.copy()
    if mode=='nonpositive' and p[a]>=-1e-8:
        projected[np.where(active==a)[0][0]]=max(0.,grad[np.where(active==a)[0][0]])
    if not r.success or np.max(abs(projected))>1e-6:raise ValueError(f'Ajuste no convergente: {r.message}; grad={np.max(abs(projected))}')
    # Covarianza agrupada del ajuste libre, con Hessiano observado exacto.
    covariance=None
    if mode=='free':
        H=J.T@(mu[:,None]*J)
        dz=design.Z-J[design.cell,a:]
        H[a:,a:]+=dz.T@((weights*(mu-y)[design.cell])[:,None]*dz)
        score=J*(y-mu)[:,None];groups=design.weekly.week.factorize()[0];G=int(groups.max()+1)
        aggregate=np.zeros((G,k));np.add.at(aggregate,groups,score)
        inv=np.linalg.inv(H)
        covariance=inv@(aggregate.T@aggregate)@inv*(G/(G-1))*((len(y)-1)/(len(y)-k))
    return {'parameters':p,'epsilon':float(p[a]),'covariance':covariance,'mu':mu,'n':len(y),'weeks':int(design.weekly.week.nunique()),'converged':bool(r.success),'gradient_max':float(np.max(abs(projected))),'mode':mode,'columns':design.columns+design.daily_columns}

def predict(result,design):return mean_jacobian(result['parameters'],design)[0]

def poisson_deviance(y,mu):
    y=np.asarray(y,float);mu=np.asarray(mu,float)
    return float(2*np.sum(xlogy(y,y/mu)-(y-mu)))
