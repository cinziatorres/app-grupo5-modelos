from pathlib import Path
import hashlib,json
import pandas as pd
from modelos import ROOT,DATA,fc
from elasticidad import estimate

def main():
    data=fc.load_data(DATA);records=[];coefficients=[]
    for sku,g in data.groupby('sku'):
        # Mismo ámbito temporal completo usado para entrenar forecast de este SKU.
        hist=fc.aggregate_sku(data,sku)
        panel=g[g.week.between(hist.week.iloc[0],hist.week.iloc[-1])]
        row,coef=estimate(panel);row['p0_recent']=float(hist.price_unit.tail(4).mean());row['history_end']=hist.week.iloc[-1].strftime('%Y-%m-%d');records.append(row);coefficients.append(coef)
    assert len(records)==30
    art=ROOT/'artifacts';art.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(art/'elasticity.csv',index=False)
    pd.concat(coefficients,ignore_index=True).to_csv(art/'elasticity_coefficients.csv',index=False)
    (art/'elasticity_manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'sku_count':30,'as_of':max(r['history_end'] for r in records),'scope':'Panel por SKU/canal/región/semana dentro del intervalo común de cada SKU; sensibilidad asociativa.'},indent=2),encoding='utf8')
    print('Elasticidad de treinta SKU preparada.')

if __name__=='__main__':main()

