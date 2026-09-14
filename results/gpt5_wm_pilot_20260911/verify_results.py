"""Recompute the published paired pilot tables without NFS, GPT or a simulator."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read_csv(name):
    with (ROOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def aggregate(rows):
    n = len(rows)
    left = sum(r['gpt_completed'] == 'true' for r in rows)
    right = sum(r['wm_completed'] == 'true' for r in rows)
    wins = sum(r['gpt_completed'] == 'false' and r['wm_completed'] == 'true' for r in rows)
    losses = sum(r['gpt_completed'] == 'true' and r['wm_completed'] == 'false' for r in rows)
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    jointly_successful = [r for r in rows if r['gpt_completed'] == r['wm_completed'] == 'true']
    return {
        'paired_n': n,
        'gpt_success_rate': left/n, 'wm_gpt_success_rate': right/n,
        'delta_success_pp': 100*(right-left)/n,
        'wm_wins': wins, 'wm_losses': losses, 'mcnemar_exact_two_sided_p': p,
        'token_paired_n': len([r for r in rows if r['wm_total_tokens'] and r['gpt_total_tokens']]),
        'mean_extra_tokens': statistics.mean(int(r['wm_total_tokens'])-int(r['gpt_total_tokens']) for r in rows),
        'mean_extra_seconds': statistics.mean(float(r['wm_duration_sec'])-float(r['gpt_duration_sec']) for r in rows),
        'steps_saved_on_joint_success': statistics.mean(int(r['gpt_actual_steps'])-int(r['wm_actual_steps']) for r in jointly_successful) if jointly_successful else None,
    }


def main():
    checksums = json.loads((ROOT / 'SHA256.json').read_text())
    for name, expected in checksums.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    rows, cohort = read_csv('paired_tasks.csv'), read_csv('selected_cohort.csv')
    key = lambda r: (r['env'], r['task_id'])
    assert len(rows) == len({key(r) for r in rows}) == 17
    assert len(cohort) == len({key(r) for r in cohort}) == 20
    assert {key(r) for r in rows} == {key(r) for r in cohort if r['included_in_paired_comparison'] == 'true'}
    selected = {key(r): r for r in cohort}
    for row in rows:
        for arm in ['gpt', 'wm']:
            assert row[arm+'_completed'] in ('true', 'false')
            assert row[arm+'_completed'] == selected[key(row)][arm+'_completed']
            assert (row[arm+'_status'] == 'success') == (row[arm+'_completed'] == 'true')
    for row in cohort:
        if row['included_in_paired_comparison'] == 'false':
            assert row['wm_completed'] == '' and row['exclusion_reason'] == 'wm_interrupted_api_quota'
    summary = json.loads((ROOT / 'summary.json').read_text())
    groups = {'all': rows, **{env: [r for r in rows if r['env'] == env] for env in ['ai2thor', 'procthor']}}
    for group, subset in groups.items():
        actual = aggregate(subset)
        expected = summary['all'] if group == 'all' else summary['by_env'][group]
        for name, value in actual.items():
            assert (value is None and expected[name] is None) or (
                value is not None and expected[name] is not None and math.isclose(value, expected[name], abs_tol=1e-9)), (group, name)
        print(f"{group}: n={len(subset)}, GPT={actual['gpt_success_rate']:.1%}, "
              f"WM+GPT={actual['wm_gpt_success_rate']:.1%}, delta={actual['delta_success_pp']:+.1f} pp")
    assert sum(r['gpt_completed'] == 'true' for r in cohort) == summary['baseline_cohort']['successes'] == 2
    print('Verified file hashes, 20 selected tasks, 17 decided pairs, success rates, exact McNemar p, and paired token/time/step summaries.')


if __name__ == '__main__':
    main()
