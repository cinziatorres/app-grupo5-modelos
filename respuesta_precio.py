"""Sensibilidad de escenarios, separada de la evidencia confirmatoria de precio.

Modelo resumido aproximado: b_hat | epsilon ~ N(epsilon, s_cluster**2).
Una densidad previa plana en epsilon <= 0 da una normal truncada en cero.
Su media es la elasticidad central del escenario; NO prueba un efecto causal.
No se cambia el signo del estimador ni se asigna una elasticidad fija común.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

VERSION='precio-sensibilidad-condicionada-v4'
ROOT=Path(__file__).resolve().parent

def distribution(estimate,standard_error):
    estimate=float(estimate);standard_error=float(standard_error)
    if not np.isfinite([estimate,standard_error]).all() or standard_error<=0:
        raise ValueError('La sensibilidad requiere estimación libre y error estándar positivo verificados.')
    return truncnorm(-np.inf,-estimate/standard_error,loc=estimate,scale=standard_error)

def prepare():
    source=ROOT/'artifacts/price_response.csv'
    receipt=ROOT/'artifacts/price_response_manifest.json'
    manifest=json.loads(receipt.read_text(encoding='utf8'))
    response=pd.read_csv(source)
    # Se preservan el ajuste libre, la optimización restringida y la selección
    # predictiva v3. Su evidencia se comunica, pero no anula la sensibilidad.
    response['predictive_model_selected']=response.selected_model
    response['predictive_elasticity_selected']=response.elasticity_used
    for i,row in response.iterrows():
        d=distribution(row.epsilon_corrected_free,row.corrected_se)
        values={'elasticity_used':float(d.mean()),'scenario_elasticity_sd':float(d.std()),
                'scenario_elasticity_low':float(d.ppf(.025)),
                'scenario_elasticity_high':float(d.ppf(.975)),
                'scenario_model':'normal resumida condicionada a epsilon <= 0',
                'status':'sensibilidad calibrada bajo supuesto de demanda decreciente'}
        for key,value in values.items():response.loc[i,key]=value
    response.to_csv(ROOT/'artifacts/price_scenario.csv',index=False)
    result={'version':VERSION,'source_sha256':manifest['source_sha256'],
            'scenario_csv_sha256':hashlib.sha256((ROOT/'artifacts/price_scenario.csv').read_bytes()).hexdigest(),
            'response_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'response_manifest_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),
            'sku_count':len(response),'estimation_model_version':manifest['version'],
            'model':'Aproximación normal del estimador semanal libre y su error agrupado; condicionada a epsilon <= 0. Media de la distribución para la curva central y sorteos de la misma para sensibilidad.',
            'prior':'Densidad plana impropia en (-infinito, 0]; posterior resumida propia. Supuesto económico de signo; ninguna magnitud fija impuesta.',
            'evidence':'price_supported conserva el resultado confirmatorio v3. No decide si se simula. Los intervalos libres y la validación se conservan.',
            'scope':'Simulación académica condicional. No es ajuste bayesiano completo de la cuasiverosimilitud ni elasticidad causal identificada. El signo y la media negativa cuando hay señal débil dependen del supuesto. No contiene una masa puntual en cero ni identifica probabilidad de inelasticidad exacta.',
            'limitations':'Normalidad aproximada del estimador; error agrupado no resuelve toda dependencia temporal/endogeneidad. Sin información de magnitud, mayor error puede aumentar la sensibilidad central al condicionar por signo; no indica mayor evidencia.'}
    (ROOT/'artifacts/price_scenario_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return response

if __name__=='__main__':
    print(prepare()[['sku','elasticity_used','scenario_elasticity_low','scenario_elasticity_high','price_supported']].to_string(index=False))
