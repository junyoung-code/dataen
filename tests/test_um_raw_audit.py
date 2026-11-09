"""Raw-only auditing of the submission period must not fit historical PCA."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_um_features import build, sha


class RawAuditTests(unittest.TestCase):
    def test_future_day_raw_values_and_missing_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)
            (r/'responses').mkdir(); (r/'days').mkdir()
            points = [{'label': f'p{i}', 'X': i, 'Y': i + 10} for i in range(1, 10)]
            (r/'points.json').write_text(json.dumps({'points': points}))
            signature = sha(r/'points.json')
            (r/'reference.json').write_text(json.dumps({'status':'verified_core_only','comparisons':162}))
            (r/'status.json').write_text(json.dumps({'status':'collected','completed_days':1,'expected_days':1}))
            (r/'config.json').write_text(json.dumps({'points_sha256':signature,'leads':[24,32,1],
                                                     'start':'2025-06-01','end':'2025-06-01'}))
            hashes = {}
            for p in points:
                f = r/'responses'/f"2025-06-01_{p['label']}.txt"
                rows = [f'2025053100 20250601{h:02d} {v} 16777215 {val}'
                        for h in range(9) for v, val in [(4200, 123.5),(6213, .25)]]
                f.write_text('#START7777\n'+'\n'.join(rows)+'\n#7777END\n')
                hashes[p['label']] = sha(f)
                f.with_suffix('.json').write_text(json.dumps({'status':'verified','sha256':sha(f),'rows':18,
                    'params':{'group':'UMKR','nwp':'N512','data':'U','varn':'4200,6213','tmfc':'2025053100',
                              'ef':'24,32,1','X':str(p['X']),'Y':str(p['Y'])}}))
            (r/'days/2025-06-01.json').write_text(json.dumps({'target_day':'2025-06-01',
                  'points_sha256':signature,'response_sha256':hashes}))
            build(r,r/'points.json',r/'reference.json',r/'out',raw_only=True)
            d = pd.read_parquet(r/'out/um_core_hourly.parquet')
            self.assertEqual(d.shape,(24,19))
            core=d.ts.dt.hour.between(9,17)
            self.assertTrue(d.loc[~core,d.columns!='ts'].isna().all().all())
            self.assertTrue((d.loc[core,'um_tdsw_p1']==123.5).all())
            self.assertEqual(list((r/'out').glob('*pca*')),[])
            meta=json.loads((r/'out/um_features.json').read_text())
            self.assertEqual(meta['finite_core_cells'],162)
            self.assertEqual(meta['complete_days'],1)
            f.write_text(f.read_text().replace('123.5','124.5'))
            with self.assertRaises(AssertionError):
                build(r,r/'points.json',r/'reference.json',r/'tampered',raw_only=True)


if __name__=='__main__':
    unittest.main()
