"""
reprocess_possessions.py
------------------------
Re-runs possession tracking + four-factors on existing plays_*.csv files
and overwrites possessions_*.csv, possessions_enriched_*.csv, and
four_factors_*.csv in cbbd_data/.

Run after fixing track_possessions_v2 (and-1 + tech-FT detection) to
correct CSVs without re-fetching data from the API.

Usage:
    python reprocess_possessions.py [--data-dir cbbd_data]
"""

import argparse
import glob
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Core functions (kept in sync with cbbd_batch_pipeline.ipynb)
# ---------------------------------------------------------------------------

FG_TYPES = {'JumpShot', 'LayUpShot', 'DunkShot', 'TipShot'}


def _is_made(txt):
    return 'makes' in txt or (' made ' in txt and 'missed' not in txt)


def _is_missed(txt):
    return 'misses' in txt or 'missed' in txt


def _safe_txt(val):
    return str(val).lower() if pd.notna(val) else ''


def _safe_str(val):
    return str(val) if pd.notna(val) else ''


def _precompute_last_ft_flags(game_df):
    sorted_df = game_df.sort_values(
        ['period', 'secondsRemaining'], ascending=[True, False]
    ).reset_index()
    is_last_ft = {}
    ft_indices = sorted_df.index[sorted_df['playType'] == 'MadeFreeThrow'].tolist()

    for pos in ft_indices:
        row = sorted_df.iloc[pos]
        txt_lower = _safe_txt(row['playText'])
        orig_idx = row['index']

        if any(f'{n} of {n}' in txt_lower for n in ('1', '2', '3')):
            is_last_ft[orig_idx] = True
            continue

        if pos + 1 < len(sorted_df):
            nxt = sorted_df.iloc[pos + 1]
            same_clock = (nxt['period'] == row['period']
                          and nxt['secondsRemaining'] == row['secondsRemaining'])
            is_last_ft[orig_idx] = not (same_clock and nxt['playType'] == 'MadeFreeThrow')
        else:
            is_last_ft[orig_idx] = True

    return is_last_ft


def track_possessions_v2(game_df):
    """State-machine possession tracker (v4: and-1 + tech-FT detection)."""
    game_df = game_df.sort_values(
        ['period', 'secondsRemaining'], ascending=[True, False]
    ).copy()
    teams = [t for t in game_df['team'].unique() if pd.notna(t)]

    def other_team(t):
        others = [x for x in teams if x != t]
        return others[0] if others else None

    last_ft_flags = _precompute_last_ft_flags(game_df)
    poss_id, poss_team, records = 0, None, []
    last_end_reason = last_end_team = None
    pending_tech_ft = False

    for idx, row in game_df.iterrows():
        pt, txt, team = _safe_str(row.get('playType')), _safe_txt(row.get('playText')), row.get('team')
        outcome, end_poss, next_team = None, False, None

        if pt == 'Jumpball':
            if 'won' in txt and team and poss_team is None:
                poss_team = team
        elif pt in FG_TYPES:
            if poss_team is None:
                poss_team = team
            if _is_made(txt):
                outcome, end_poss, next_team = 'made_fg', True, other_team(poss_team)
        elif pt == 'MadeFreeThrow':
            if pending_tech_ft or 'technical' in txt:
                outcome = 'tech_ft'
                if last_ft_flags.get(idx, False):
                    pending_tech_ft = False
            else:
                if (team and poss_team is not None and team != poss_team
                        and last_end_reason == 'made_fg' and last_end_team == team):
                    poss_id -= 1
                    poss_team = team
                    for rec in reversed(records):
                        if rec['possession_id'] == poss_id + 1:
                            rec['possession_id'] = poss_id
                            rec['possession_team'] = poss_team
                        else:
                            break
                    last_end_reason = 'and1_ft'
                elif poss_team is None and team:
                    poss_team = team

                if last_ft_flags.get(idx, False):
                    if _is_made(txt):
                        outcome, end_poss, next_team = 'made_ft', True, other_team(poss_team)
                    else:
                        outcome = 'missed_last_ft'
        elif 'Technical' in pt:
            pending_tech_ft = True
            outcome = 'technical_foul'
        elif 'Turnover' in pt:
            if poss_team is None and team:
                poss_team = team
            outcome, end_poss, next_team = 'turnover', True, other_team(poss_team)
        elif pt == 'Steal':
            outcome = 'steal'
        elif pt == 'Defensive Rebound':
            outcome, end_poss, next_team = 'def_rebound', True, team
        elif pt == 'Offensive Rebound':
            outcome = 'off_rebound'
        elif pt == 'Dead Ball Rebound':
            outcome, end_poss, next_team = 'dead_ball_rebound', True, team
        elif pt in ('End Period', 'End Game'):
            outcome, end_poss, next_team = 'end_period', True, None

        records.append({
            'play_id': row.get('id'), 'gameId': row.get('gameId'),
            'possession_id': poss_id, 'possession_team': poss_team,
            'play_type': pt, 'play_text': row.get('playText', ''),
            'team': team, 'outcome': outcome,
        })

        if end_poss:
            last_end_reason, last_end_team = outcome, poss_team
            poss_id += 1
            poss_team = next_team
        elif pt not in ('PersonalFoul', 'MadeFreeThrow', 'Substitution', 'Official TV Timeout', ''):
            last_end_reason = last_end_team = None

    return pd.DataFrame(records)


def classify_possessions(possessions_df, game_df):
    game_sorted = game_df.sort_values(
        ['period', 'secondsRemaining'], ascending=[True, False]
    ).reset_index(drop=True)
    game_sorted['play_order'] = range(len(game_sorted))
    time_info = game_sorted[['id', 'secondsRemaining', 'period', 'play_order']].rename(columns={'id': 'play_id'})
    poss = possessions_df.merge(time_info, on='play_id', how='left').sort_values('play_order').reset_index(drop=True)

    rows = []
    for pid in sorted(poss['possession_id'].unique()):
        grp = poss[poss['possession_id'] == pid].sort_values('play_order')
        plays = grp.to_dict('records')
        poss_team = grp['possession_team'].iloc[0]
        period = grp['period'].iloc[0]
        start_sec, end_sec = grp['secondsRemaining'].iloc[0], grp['secondsRemaining'].iloc[-1]

        all_outcomes = [p['outcome'] for p in plays if p['outcome'] is not None]
        final = all_outcomes[-1] if all_outcomes else None
        outcome_set = set(all_outcomes)
        has_steal, has_oreb = 'steal' in outcome_set, 'off_rebound' in outcome_set

        if final == 'turnover':
            refined = 'live_ball_turnover' if has_steal else 'dead_ball_turnover'
        elif final == 'def_rebound':
            miss_type, found = 'fga', False
            for p in reversed(plays):
                if not found:
                    found = p['outcome'] == 'def_rebound'
                    continue
                if p['play_type'] in FG_TYPES:
                    break
                if p['play_type'] == 'MadeFreeThrow' and _is_missed(_safe_txt(p.get('play_text'))):
                    miss_type = 'fta'; break
            refined = f'{miss_type}_def_rebound'
        elif final in ('made_fg', 'made_ft', 'end_period', 'dead_ball_rebound'):
            refined = final
        else:
            refined = final

        for i_p, p in enumerate(plays):
            pt_lower = _safe_txt(p.get('play_text'))
            if p['play_type'] in FG_TYPES and 'block' in pt_lower and _is_missed(pt_lower):
                remaining = plays[i_p + 1:]
                if not any(r['outcome'] in ('def_rebound', 'off_rebound', 'dead_ball_rebound') for r in remaining):
                    if len(poss[poss['possession_id'] == pid + 1]) > 0:
                        refined = 'block_oob'
                break

        fga_plays = [p for p in plays if p['play_type'] in FG_TYPES]
        first_fga_sec = fga_plays[0]['secondsRemaining'] if fga_plays else None
        time_to_first_fga = (start_sec - first_fga_sec) if first_fga_sec is not None else None

        oreb_list = [p for p in plays if p['outcome'] == 'off_rebound']
        time_oreb_to_fga = None
        if oreb_list:
            oreb_sec = oreb_list[0]['secondsRemaining']
            post = [p for p in fga_plays if p['secondsRemaining'] < oreb_sec]
            if post:
                time_oreb_to_fga = oreb_sec - post[0]['secondsRemaining']

        foul_plays = [p for p in plays if 'Foul' in (p.get('play_type') or '')
                      and 'shooting' not in _safe_txt(p.get('play_text'))]
        foul_within_10s = bool(foul_plays and (start_sec - foul_plays[0]['secondsRemaining']) <= 10)

        if has_oreb:
            poss_type = 'scramble_putback' if (time_oreb_to_fga is not None and time_oreb_to_fga <= 3) else 'second_chance'
        elif foul_plays and foul_within_10s and not fga_plays and start_sec <= 120:
            poss_type = 'intentional_foul'
        elif time_to_first_fga is not None:
            poss_type = 'transition' if time_to_first_fga <= 7 else 'half_court'
        else:
            poss_type = 'half_court'

        rows.append({
            'gameId': grp['gameId'].iloc[0] if 'gameId' in grp.columns else None,
            'possession_id': pid, 'possession_team': poss_team, 'period': period,
            'start_seconds': start_sec, 'end_seconds': end_sec,
            'duration_sec': start_sec - end_sec, 'raw_outcome': final,
            'refined_outcome': refined, 'possession_type': poss_type,
            'has_oreb': has_oreb, 'time_to_first_fga': time_to_first_fga,
            'time_oreb_to_fga': time_oreb_to_fga,
        })

    result = pd.DataFrame(rows)
    prev_enders = ['start_of_period']
    for i in range(1, len(result)):
        if result.iloc[i]['period'] != result.iloc[i - 1]['period']:
            prev_enders.append('start_of_period')
        else:
            prev_enders.append(result.iloc[i - 1]['refined_outcome'])
    result['prev_poss_ender'] = prev_enders
    return result


def compute_four_factors(game_df):
    teams = game_df[game_df['team'].notna()]['team'].unique()
    results = {}
    for team in teams:
        tp = game_df[game_df['team'] == team]
        opp = [t for t in teams if t != team]
        opp_plays = game_df[game_df['team'] == opp[0]] if opp else pd.DataFrame()

        fg = tp[tp['playType'].isin(FG_TYPES)]
        fga = len(fg)
        fg_txt = fg['playText'].fillna('').str.lower()
        fgm = (fg_txt.str.contains('makes') | (fg_txt.str.contains(' made ') & ~fg_txt.str.contains('missed'))).sum()
        tpa = fg['playText'].fillna('').str.contains('three point', case=False, na=False).sum()
        tpm = ((fg_txt.str.contains('three point')) & (fg_txt.str.contains('makes') | (fg_txt.str.contains(' made ') & ~fg_txt.str.contains('missed')))).sum()
        ft = tp[tp['playType'] == 'MadeFreeThrow']
        fta = len(ft)
        ftm = (ft['playText'].fillna('').str.lower().str.contains('makes') | (ft['playText'].fillna('').str.lower().str.contains(' made ') & ~ft['playText'].fillna('').str.lower().str.contains('missed'))).sum()
        tov = len(tp[tp['playType'].str.contains('Turnover', na=False)])
        orb = len(tp[tp['playType'] == 'Offensive Rebound'])
        drb = len(tp[tp['playType'] == 'Defensive Rebound'])
        opp_drb = len(opp_plays[opp_plays['playType'] == 'Defensive Rebound']) if len(opp_plays) else 0
        possessions = fga - orb + tov + 0.475 * fta
        n_per = game_df['period'].max()
        mins = 40 if n_per <= 2 else 40 + (n_per - 2) * 5

        results[team] = {
            'FGA': fga, 'FGM': int(fgm), '3PA': tpa, '3PM': int(tpm),
            '2PA': fga - tpa, '2PM': int(fgm - tpm), 'FTA': fta, 'FTM': int(ftm),
            'TOV': tov, 'ORB': orb, 'DRB': drb, 'Opp_DRB': opp_drb,
            'Possessions': round(possessions, 1),
            'eFG%': round((fgm + 0.5 * tpm) / fga * 100 if fga else 0, 1),
            'TO%': round(tov / possessions * 100 if possessions else 0, 1),
            'ORB%': round(orb / (orb + opp_drb) * 100 if (orb + opp_drb) else 0, 1),
            'FT_Rate': round(fta / fga * 100 if fga else 0, 1),
            '3PA_Rate': round(tpa / fga * 100 if fga else 0, 1),
            'Tempo': round(possessions / (mins / 40), 1),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='cbbd_data')
    args = parser.parse_args()
    data_dir = args.data_dir

    plays_files = sorted(glob.glob(os.path.join(data_dir, 'plays_*.csv')))
    if not plays_files:
        print(f"No plays_*.csv files found in {data_dir}/")
        return

    print(f"Found {len(plays_files)} plays file(s) to reprocess:\n")

    for plays_path in plays_files:
        basename = os.path.basename(plays_path)
        suffix = basename[len('plays_'):-len('.csv')]
        print(f"  {basename}")

        plays_df = pd.read_csv(plays_path)
        game_ids = plays_df['gameId'].unique()
        print(f"    {len(plays_df):,} plays, {len(game_ids)} games")

        all_poss, all_enriched, ff_rows, failed = [], [], [], []

        for gid in game_ids:
            game_df = plays_df[plays_df['gameId'] == gid].copy()
            teams = game_df[game_df['team'].notna()]['team'].unique()
            try:
                poss_df  = track_possessions_v2(game_df)
                enriched = classify_possessions(poss_df, game_df)
                ff       = compute_four_factors(game_df)
                all_poss.append(poss_df)
                all_enriched.append(enriched)
                for team, stats in ff.items():
                    opp = [t for t in teams if t != team]
                    stats.update({'game_id': gid, 'team': team, 'opponent': opp[0] if opp else None})
                    ff_rows.append(stats)
            except Exception as e:
                failed.append(gid)
                print(f"    FAILED game {gid}: {e}")

        if not all_poss:
            print("    Nothing processed successfully, skipping.\n")
            continue

        for name, df in [
            ('possessions',          pd.concat(all_poss,     ignore_index=True)),
            ('possessions_enriched', pd.concat(all_enriched, ignore_index=True)),
            ('four_factors',         pd.DataFrame(ff_rows)),
        ]:
            out = os.path.join(data_dir, f'{name}_{suffix}.csv')
            df.to_csv(out, index=False)
            print(f"    -> {out}  ({len(df):,} rows)")

        if failed:
            print(f"    {len(failed)} failed: {failed}")
        print()

    print("Done.")


if __name__ == '__main__':
    main()
