"""Agrega la combinación de pronósticos como quinto candidato.

Combinación = promedio simple de los modelos ADMISIBLES de cada SKU (los que pasaron
calibración, selección, comprobación y diagnósticos finales). Se evalúa con los mismos
cortes (selección H12 y comprobación H12) y se aplica la misma regla de ganador: menor
WAPE de SELECCIÓN (MAE como desempate). La comprobación no interviene en la elección.

Respaldo: en las competencias M4/M5 las combinaciones simples de pronósticos estuvieron
entre los métodos más precisos y robustos (Makridakis et al., 2022).
"""
import json

import numpy as np
import pandas as pd

from modelos import ROOT, fc, NAMES, COMBO

CHECK = ROOT / 'training'
ART = ROOT / 'artifacts'


def eligible_members(meta, metrics):
    sel = metrics[metrics.scope.eq('selection')].set_index('model')
    hold = metrics[metrics.scope.eq('holdout')].set_index('model')
    return [n for n in NAMES if n in sel.index and n in hold.index and bool(sel.loc[n, 'eligible'])
            and bool(hold.loc[n, 'eligible']) and meta['final_diagnostics'].get(n, {}).get('eligible', False)
            and np.isfinite(sel.loc[n, 'WAPE'])]


def main():
    changes = []
    skus = sorted(p.name.replace('_result.json', '') for p in CHECK.glob('*_result.json'))
    for sku in skus:
        meta = json.loads((CHECK / f'{sku}_result.json').read_text())
        m = pd.read_csv(CHECK / f'{sku}_metrics.csv')
        p = pd.read_csv(CHECK / f'{sku}_predictions.csv')
        m, p = m[m.model.ne(COMBO)], p[p.model.ne(COMBO)]
        members = eligible_members(meta, m)
        rows, preds = [], []
        for scope in ['selection', 'holdout']:
            q = p[p.scope.eq(scope) & p.model.isin(members)]
            g = q.groupby(['week', 'horizon_step', 'origin_week'], as_index=False).agg(
                actual=('actual', 'first'), prediction=('prediction', 'mean'))
            g['residual'] = g.actual - g.prediction
            a, y = g.actual.to_numpy(float), g.prediction.to_numpy(float)
            rows.append({'sku': sku, 'model': COMBO, 'scope': scope, 'eligible': len(members) >= 2,
                         'reason': 'ok' if len(members) >= 2 else 'menos de dos modelos admisibles',
                         'WAPE': fc.wape(a, y), 'MAE': fc.mae(a, y), 'bias': float(np.mean(a - y))})
            preds.append(g.assign(sku=sku, model=COMBO, scope=scope))
        m = pd.concat([m, pd.DataFrame(rows)], ignore_index=True)
        p = pd.concat([p, *[x[p.columns] for x in preds]], ignore_index=True)

        sel = m[m.scope.eq('selection')].set_index('model')
        hold = m[m.scope.eq('holdout')].set_index('model')
        cands = members + ([COMBO] if len(members) >= 2 else [])
        winner = sorted(cands, key=lambda n: (sel.loc[n, 'WAPE'], sel.loc[n, 'MAE']))[0]
        old = meta['winner']
        meta['params'][COMBO] = {'members': members}
        meta['final_diagnostics'][COMBO] = {'eligible': len(members) >= 2, 'reason': 'promedio de modelos admisibles'}
        meta.update(winner=winner, selection_WAPE=float(sel.loc[winner, 'WAPE']),
                    holdout_WAPE=float(hold.loc[winner, 'WAPE']), holdout_MAE=float(hold.loc[winner, 'MAE']))
        m.to_csv(CHECK / f'{sku}_metrics.csv', index=False)
        p.to_csv(CHECK / f'{sku}_predictions.csv', index=False)
        (CHECK / f'{sku}_result.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
        changes.append({'sku': sku, 'ganador_anterior': old, 'ganador_nuevo': winner, 'miembros': ' + '.join(members),
                        'WAPE_comprobacion_anterior': float(hold.loc[old, 'WAPE']),
                        'WAPE_comprobacion_nuevo': float(hold.loc[winner, 'WAPE'])})

    records = [json.loads((CHECK / f'{s}_result.json').read_text()) for s in skus]
    cols = ['sku', 'winner', 'n_weeks', 'history_start', 'history_end', 'selection_WAPE', 'holdout_WAPE', 'holdout_MAE']
    pd.DataFrame([{k: r[k] for k in cols} for r in records]).to_csv(ART / 'model_registry.csv', index=False)
    for table in ['metrics', 'predictions']:
        pd.concat([pd.read_csv(CHECK / f'{s}_{table}.csv') for s in skus], ignore_index=True).to_csv(ART / f'{table}.csv', index=False)
    out = pd.DataFrame(changes)
    out.to_csv(ART / 'combination_update.csv', index=False)
    print(out.to_string(index=False))
    print('WAPE comprobación promedio:', round(out.WAPE_comprobacion_anterior.mean(), 2), '->', round(out.WAPE_comprobacion_nuevo.mean(), 2))
    print('WAPE comprobación mediana :', round(out.WAPE_comprobacion_anterior.median(), 2), '->', round(out.WAPE_comprobacion_nuevo.median(), 2))


if __name__ == '__main__':
    main()
