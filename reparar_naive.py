"""Repara la evaluación del benchmark Seasonal Naive (bug del entrenamiento original).

En entrenar_integrado.py el Naive llamaba a fc.seasonal_naive_forecast, que no existía
en forecast_core; todas sus evaluaciones fallaron con AttributeError y el benchmark
quedó "no elegible" con el motivo 'Seasonal Naive'. Este script lo evalúa con
exactamente los mismos cortes (calibración 4, selección H12, comprobación H12) y vuelve
a aplicar la regla de ganador: menor WAPE/MAE de SELECCIÓN entre candidatos admisibles.
Los modelos ya entrenados (Ridge, SARIMAX, DHR) no se modifican: el Naive no se ajusta.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modelos import ROOT, DATA, fc, NAMES

CHECK = ROOT / 'training'
ART = ROOT / 'artifacts'
NAIVE = fc.MODEL_NAIVE


def naive_rows(sku, hist):
    rows, preds = [], []
    for scope, train, target in [('selection', hist.iloc[:-24], hist.iloc[-24:-12]),
                                 ('holdout', hist.iloc[:-12], hist.iloc[-12:])]:
        y = np.maximum(fc.seasonal_naive_forecast(train, len(target)), 0)
        a = target.units_sold.to_numpy(float)
        rows.append({'sku': sku, 'model': NAIVE, 'scope': scope, 'eligible': True, 'reason': 'ok',
                     'WAPE': fc.wape(a, y), 'MAE': fc.mae(a, y), 'bias': float(np.mean(a - y))})
        for h, (week, actual, value) in enumerate(zip(target.week, a, y), 1):
            preds.append({'sku': sku, 'model': NAIVE, 'scope': scope,
                          'origin_week': train.week.iloc[-1].strftime('%Y-%m-%d'),
                          'week': week.strftime('%Y-%m-%d'), 'horizon_step': h, 'actual': float(actual),
                          'prediction': float(value), 'residual': float(actual - value)})
    dev = hist.iloc[:-24]
    yc = np.maximum(fc.seasonal_naive_forecast(dev.iloc[:-4], 4), 0)
    cal = {'sku': sku, 'model': NAIVE, 'config': '{}', 'WAPE': fc.wape(dev.iloc[-4:].units_sold, yc),
           'MAE': fc.mae(dev.iloc[-4:].units_sold, yc), 'eligible': True, 'reason': 'ok'}
    return rows, preds, cal


def main():
    data = fc.load_data(DATA)
    changes = []
    skus = sorted(p.name.replace('_result.json', '') for p in CHECK.glob('*_result.json'))
    for sku in skus:
        meta = json.loads((CHECK / f'{sku}_result.json').read_text())
        hist = fc.aggregate_sku(data, sku)
        rows, preds, cal = naive_rows(sku, hist)
        m = pd.read_csv(CHECK / f'{sku}_metrics.csv')
        m = pd.concat([m[m.model.ne(NAIVE)], pd.DataFrame(rows)], ignore_index=True)
        p = pd.read_csv(CHECK / f'{sku}_predictions.csv')
        p = pd.concat([p[p.model.ne(NAIVE)], pd.DataFrame(preds)], ignore_index=True)
        c = pd.read_csv(CHECK / f'{sku}_calibration.csv')
        c = pd.concat([c[c.model.ne(NAIVE)], pd.DataFrame([cal])], ignore_index=True)

        meta['params'][NAIVE] = {}
        meta['final_diagnostics'][NAIVE] = {'eligible': True, 'reason': 'ok'}
        sel = m[m.scope.eq('selection')].set_index('model')
        hold = m[m.scope.eq('holdout')].set_index('model')
        ok = [n for n in NAMES if n in sel.index and bool(sel.loc[n, 'eligible']) and n in hold.index
              and bool(hold.loc[n, 'eligible']) and meta['final_diagnostics'].get(n, {}).get('eligible', False)
              and np.isfinite(sel.loc[n, 'WAPE'])]
        winner = sorted(ok, key=lambda n: (sel.loc[n, 'WAPE'], sel.loc[n, 'MAE']))[0]
        old = meta['winner']
        meta.update(winner=winner, selection_WAPE=float(sel.loc[winner, 'WAPE']),
                    holdout_WAPE=float(hold.loc[winner, 'WAPE']), holdout_MAE=float(hold.loc[winner, 'MAE']),
                    naive_repair='Seasonal Naive evaluado con los mismos cortes; regla de ganador sin cambios.')
        m.to_csv(CHECK / f'{sku}_metrics.csv', index=False)
        p.to_csv(CHECK / f'{sku}_predictions.csv', index=False)
        c.to_csv(CHECK / f'{sku}_calibration.csv', index=False)
        (CHECK / f'{sku}_result.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
        changes.append({'sku': sku, 'ganador_anterior': old, 'ganador_nuevo': winner,
                        'naive_WAPE_seleccion': float(sel.loc[NAIVE, 'WAPE']),
                        'ganador_WAPE_seleccion': float(sel.loc[winner, 'WAPE'])})

    records = [json.loads((CHECK / f'{s}_result.json').read_text()) for s in skus]
    cols = ['sku', 'winner', 'n_weeks', 'history_start', 'history_end', 'selection_WAPE', 'holdout_WAPE', 'holdout_MAE']
    pd.DataFrame([{k: r[k] for k in cols} for r in records]).to_csv(ART / 'model_registry.csv', index=False)
    for table in ['calibration', 'metrics', 'predictions']:
        pd.concat([pd.read_csv(CHECK / f'{s}_{table}.csv') for s in skus], ignore_index=True).to_csv(ART / f'{table}.csv', index=False)
    out = pd.DataFrame(changes)
    out.to_csv(ART / 'naive_repair.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
