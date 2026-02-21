"""
Re-run track_possessions_v2 + classify_possessions on all raw plays CSVs
to regenerate corrected possessions and possessions_enriched CSVs.

No API calls — purely offline from saved plays files.

Handles both API text formats:
  NEW: "Player makes 18-foot jumper" / "makes free throw 2 of 2"
  OLD: "Player made Layup." / "made Free Throw."

v5 fixes:
  1. M-of-N regex for last-FT detection (not just N-of-N)
  2. id tiebreaker in sort for stable ordering
  3. Context-aware dead ball rebounds
  4. end_period filtered from enriched output
  5. Technical foul FTs don't end possession
"""

import pandas as pd
import numpy as np
import re
import os
import time

# ---------------------------------------------------------------------------
# Analysis functions — fixed for dual text format (v5)
# ---------------------------------------------------------------------------
FG_TYPES = {'JumpShot', 'LayUpShot', 'DunkShot', 'TipShot'}


def _is_made(txt):
    """Check if play text indicates a made shot (handles both formats)."""
    return 'makes' in txt or (' made ' in txt and 'missed' not in txt)


def _is_missed(txt):
    """Check if play text indicates a missed shot (handles both formats)."""
    return 'misses' in txt or 'missed' in txt


def _safe_txt(val):
    """Safely convert a value to lowercase string (handles NaN)."""
    if pd.notna(val):
        return str(val).lower()
    return ''


def _safe_str(val):
    """Safely convert a value to string (handles NaN)."""
    if pd.notna(val):
        return str(val)
    return ''


# ---------------------------------------------------------------------------
# Fix 1 + Fix 2: Last-FT detection with M-of-N regex + id tiebreaker sort
# ---------------------------------------------------------------------------
def _precompute_last_ft_flags(game_df):
    """Pre-compute which MadeFreeThrow rows are the last FT in a sequence.

    NEW format: has 'M of N' pattern — last when M == N (e.g. '2 of 2').
    OLD format: no 'M of N'; detect last FT by checking whether the next
                play is also a MadeFreeThrow at the same clock time.

    Uses 'id' as sort tiebreaker for stable ordering at the same clock.
    """
    sorted_df = game_df.sort_values(
        ['period', 'secondsRemaining', 'id'], ascending=[True, False, True]
    ).reset_index()
    is_last_ft = {}  # original_index -> bool

    ft_mask = sorted_df['playType'] == 'MadeFreeThrow'
    ft_indices = sorted_df.index[ft_mask].tolist()

    for pos in ft_indices:
        row = sorted_df.iloc[pos]
        txt_lower = _safe_txt(row['playText'])
        orig_idx = row['index']  # original DataFrame index

        # New format: use regex to extract M and N from "M of N"
        m_of_n = re.search(r'(\d+)\s+of\s+(\d+)', txt_lower)
        if m_of_n:
            m_val, n_val = int(m_of_n.group(1)), int(m_of_n.group(2))
            is_last_ft[orig_idx] = (m_val == n_val)
            continue

        # Old format: check if the NEXT row is also a FT at the same clock
        if pos + 1 < len(sorted_df):
            next_row = sorted_df.iloc[pos + 1]
            same_clock = (next_row['period'] == row['period']
                          and next_row['secondsRemaining'] == row['secondsRemaining'])
            next_is_ft = next_row['playType'] == 'MadeFreeThrow'
            if same_clock and next_is_ft:
                is_last_ft[orig_idx] = False
            else:
                is_last_ft[orig_idx] = True
        else:
            is_last_ft[orig_idx] = True  # last play in game

    return is_last_ft


# ---------------------------------------------------------------------------
# Fix 5: Technical Foul FT detection
# ---------------------------------------------------------------------------
def _precompute_tech_ft_flags(game_df):
    """Pre-compute which MadeFreeThrow rows follow a Technical Foul.

    Technical FTs don't change possession — the fouled team shoots then
    retains the ball. We detect them by looking for a TechnicalFoul play
    immediately preceding the FT at the same clock time.

    Returns dict: original_index -> bool (True = this is a tech FT).
    """
    sorted_df = game_df.sort_values(
        ['period', 'secondsRemaining', 'id'], ascending=[True, False, True]
    ).reset_index()
    is_tech_ft = {}

    ft_mask = sorted_df['playType'] == 'MadeFreeThrow'
    ft_indices = sorted_df.index[ft_mask].tolist()

    for pos in ft_indices:
        row = sorted_df.iloc[pos]
        orig_idx = row['index']

        # Look backwards for a TechnicalFoul at the same clock
        found_tech = False
        for back in range(pos - 1, max(pos - 6, -1), -1):
            prev = sorted_df.iloc[back]
            if prev['period'] != row['period']:
                break
            if prev['secondsRemaining'] != row['secondsRemaining']:
                break
            if 'Technical' in _safe_str(prev.get('playType', '')):
                found_tech = True
                break
            # Skip over other FTs and fouls at same clock
            if prev['playType'] in ('MadeFreeThrow', 'PersonalFoul'):
                continue
            break

        is_tech_ft[orig_idx] = found_tech

    return is_tech_ft


# ---------------------------------------------------------------------------
# Fix 3: Dead Ball Rebound classification
# ---------------------------------------------------------------------------
def _classify_dead_ball_rebounds(game_df, last_ft_flags):
    """Classify each Dead Ball Rebound so the tracker knows whether to end possession.

    Categories:
      'mid_ft_sequence'    — between FTs of a multi-FT trip -> don't end possession
      'after_made_last_ft' — after a made last FT -> don't end (already ended by made_ft)
      'same_team_fg_miss'  — same team missed FG (offensive DB reb) -> don't end
      'end_possession'     — opponent missed FG or standard -> end possession (current behavior)

    Returns dict: original_index -> category string.
    """
    sorted_df = game_df.sort_values(
        ['period', 'secondsRemaining', 'id'], ascending=[True, False, True]
    ).reset_index()
    db_reb_class = {}

    db_mask = sorted_df['playType'] == 'Dead Ball Rebound'
    db_indices = sorted_df.index[db_mask].tolist()

    for pos in db_indices:
        row = sorted_df.iloc[pos]
        orig_idx = row['index']
        db_team = row['team']

        # Look backwards for context
        category = 'end_possession'  # default
        for back in range(pos - 1, max(pos - 10, -1), -1):
            prev = sorted_df.iloc[back]
            if prev['period'] != row['period']:
                break
            prev_pt = _safe_str(prev.get('playType', ''))

            # Skip non-play events
            if prev_pt in ('Substitution', 'Official TV Timeout', ''):
                continue

            if prev_pt == 'MadeFreeThrow':
                prev_orig = prev['index']
                prev_is_last = last_ft_flags.get(prev_orig, False)
                if not prev_is_last:
                    # This DB reb is between FTs of a multi-FT trip
                    category = 'mid_ft_sequence'
                else:
                    # After last FT
                    prev_txt = _safe_txt(prev.get('playText', ''))
                    if _is_made(prev_txt):
                        category = 'after_made_last_ft'
                    else:
                        # Missed last FT -> this is a live rebound situation
                        category = 'end_possession'
                break

            if prev_pt in FG_TYPES:
                prev_txt = _safe_txt(prev.get('playText', ''))
                if _is_missed(prev_txt):
                    # Check if same team as the miss
                    if pd.notna(prev['team']) and pd.notna(db_team) and prev['team'] == db_team:
                        category = 'same_team_fg_miss'
                    else:
                        category = 'end_possession'
                break

            if 'Turnover' in prev_pt or prev_pt in ('PersonalFoul',):
                break

        db_reb_class[orig_idx] = category

    return db_reb_class


# ---------------------------------------------------------------------------
# Possession tracker (state machine) — v5
# ---------------------------------------------------------------------------
def track_possessions_v2(game_df):
    """State-machine possession tracker."""
    game_df = game_df.sort_values(
        ['period', 'secondsRemaining', 'id'], ascending=[True, False, True]
    ).copy()
    teams = [t for t in game_df['team'].unique() if pd.notna(t)]

    def other_team(t):
        others = [x for x in teams if x != t]
        return others[0] if others else None

    # Pre-compute flags
    last_ft_flags = _precompute_last_ft_flags(game_df)
    tech_ft_flags = _precompute_tech_ft_flags(game_df)
    db_reb_classes = _classify_dead_ball_rebounds(game_df, last_ft_flags)

    poss_id = 0
    poss_team = None
    records = []
    last_end_reason = None
    last_end_team = None

    for idx, row in game_df.iterrows():
        pt   = _safe_str(row.get('playType'))
        txt  = _safe_txt(row.get('playText'))
        team = row.get('team')
        outcome = None
        end_poss = False
        next_team = None

        if pt == 'Jumpball':
            if 'won' in txt and team and poss_team is None:
                poss_team = team
        elif pt in FG_TYPES:
            if poss_team is None:
                poss_team = team
            if _is_made(txt):
                outcome = 'made_fg'
                end_poss = True
                next_team = other_team(poss_team)
        elif pt == 'MadeFreeThrow':
            # --- Fix 5: Technical FT — don't end possession ---
            is_tech = tech_ft_flags.get(idx, False)
            if is_tech:
                outcome = 'tech_ft'
                # Don't set end_poss, don't flip teams
            else:
                # --- And-1 detection ---
                if (team and poss_team is not None
                        and team != poss_team
                        and last_end_reason == 'made_fg'
                        and last_end_team == team):
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

                is_last = last_ft_flags.get(idx, False)
                if is_last:
                    if _is_made(txt):
                        outcome = 'made_ft'
                        end_poss = True
                        next_team = other_team(poss_team)
                    else:
                        outcome = 'missed_last_ft'
        elif 'Turnover' in pt:
            if poss_team is None and team:
                poss_team = team
            outcome = 'turnover'
            end_poss = True
            next_team = other_team(poss_team)
        elif pt == 'Steal':
            outcome = 'steal'
        elif pt == 'Defensive Rebound':
            outcome = 'def_rebound'
            end_poss = True
            next_team = team
        elif pt == 'Offensive Rebound':
            outcome = 'off_rebound'
        elif pt == 'Dead Ball Rebound':
            # --- Fix 3: context-aware dead ball rebounds ---
            db_class = db_reb_classes.get(idx, 'end_possession')
            outcome = 'dead_ball_rebound'
            if db_class == 'end_possession':
                end_poss = True
                next_team = team
            # mid_ft_sequence, after_made_last_ft, same_team_fg_miss -> don't end
        elif pt in ('End Period', 'End Game'):
            outcome = 'end_period'
            end_poss = True
            next_team = None

        records.append({
            'play_id': row.get('id'),
            'gameId':  row.get('gameId'),
            'possession_id': poss_id,
            'possession_team': poss_team,
            'play_type': pt,
            'play_text': row.get('playText', ''),
            'team': team,
            'outcome': outcome,
        })

        if end_poss:
            last_end_reason = outcome
            last_end_team = poss_team
            poss_id += 1
            poss_team = next_team
        elif pt not in ('PersonalFoul', 'MadeFreeThrow', 'Substitution',
                        'Official TV Timeout', ''):
            last_end_reason = None
            last_end_team = None

    return pd.DataFrame(records)


def classify_possessions(possessions_df, game_df):
    """Build possession-level features: refined_outcome, prev_poss_ender, possession_type.

    Fix 4: Filters out end_period possessions from the final output.
    """
    game_sorted = game_df.sort_values(
        ['period', 'secondsRemaining', 'id'], ascending=[True, False, True]
    ).reset_index(drop=True)
    game_sorted['play_order'] = range(len(game_sorted))
    time_info = game_sorted[['id', 'secondsRemaining', 'period', 'play_order']].rename(
        columns={'id': 'play_id'}
    )
    poss = possessions_df.merge(time_info, on='play_id', how='left')
    poss = poss.sort_values('play_order').reset_index(drop=True)

    possession_rows = []
    for pid in sorted(poss['possession_id'].unique()):
        grp = poss[poss['possession_id'] == pid].sort_values('play_order')
        plays = grp.to_dict('records')
        game_id   = grp['gameId'].iloc[0] if 'gameId' in grp.columns else None
        poss_team = grp['possession_team'].iloc[0]
        period    = grp['period'].iloc[0]
        start_sec = grp['secondsRemaining'].iloc[0]
        end_sec   = grp['secondsRemaining'].iloc[-1]
        duration  = start_sec - end_sec

        all_outcomes = [p['outcome'] for p in plays if p['outcome'] is not None]
        final_outcome = all_outcomes[-1] if all_outcomes else None
        outcome_set = set(all_outcomes)
        has_steal = 'steal' in outcome_set
        has_oreb  = 'off_rebound' in outcome_set

        if final_outcome == 'turnover':
            refined = 'live_ball_turnover' if has_steal else 'dead_ball_turnover'
        elif final_outcome == 'def_rebound':
            miss_type = 'fga'
            found_dreb = False
            for p in reversed(plays):
                if not found_dreb:
                    if p['outcome'] == 'def_rebound':
                        found_dreb = True
                    continue
                if p['play_type'] in FG_TYPES:
                    miss_type = 'fga'; break
                if p['play_type'] == 'MadeFreeThrow' and _is_missed(
                        _safe_txt(p.get('play_text'))):
                    miss_type = 'fta'; break
            refined = f'{miss_type}_def_rebound'
        elif final_outcome in ('made_fg', 'made_ft', 'end_period', 'dead_ball_rebound'):
            refined = final_outcome
        else:
            refined = final_outcome

        for i_p, p in enumerate(plays):
            pt_lower = _safe_txt(p.get('play_text'))
            if (p['play_type'] in FG_TYPES
                    and 'block' in pt_lower
                    and _is_missed(pt_lower)):
                remaining = plays[i_p + 1:]
                has_reb = any(r['outcome'] in ('def_rebound', 'off_rebound', 'dead_ball_rebound') for r in remaining)
                if not has_reb:
                    next_p = poss[poss['possession_id'] == pid + 1]
                    if len(next_p) > 0:
                        refined = 'block_oob'
                    break

        fga_plays = [p for p in plays if p['play_type'] in FG_TYPES]
        first_fga_sec = fga_plays[0]['secondsRemaining'] if fga_plays else None
        time_to_first_fga = (start_sec - first_fga_sec) if first_fga_sec is not None else None

        oreb_list = [p for p in plays if p['outcome'] == 'off_rebound']
        time_oreb_to_fga = None
        if oreb_list:
            oreb_sec = oreb_list[0]['secondsRemaining']
            post_oreb_fga = [p for p in fga_plays if p['secondsRemaining'] < oreb_sec]
            if post_oreb_fga:
                time_oreb_to_fga = oreb_sec - post_oreb_fga[0]['secondsRemaining']

        foul_plays = [p for p in plays
                      if 'Foul' in (p.get('play_type') or '')
                      and 'shooting' not in _safe_txt(p.get('play_text'))]
        foul_within_10s = False
        if foul_plays:
            if (start_sec - foul_plays[0]['secondsRemaining']) <= 10:
                foul_within_10s = True

        if has_oreb:
            poss_type = 'scramble_putback' if (time_oreb_to_fga is not None and time_oreb_to_fga <= 3) else 'second_chance'
        elif foul_plays and foul_within_10s and not fga_plays and start_sec <= 120:
            poss_type = 'intentional_foul'
        elif time_to_first_fga is not None:
            poss_type = 'transition' if time_to_first_fga <= 7 else 'half_court'
        else:
            poss_type = 'half_court'

        possession_rows.append({
            'gameId': game_id, 'possession_id': pid, 'possession_team': poss_team,
            'period': period, 'start_seconds': start_sec, 'end_seconds': end_sec,
            'duration_sec': duration, 'raw_outcome': final_outcome,
            'refined_outcome': refined, 'possession_type': poss_type,
            'has_oreb': has_oreb, 'time_to_first_fga': time_to_first_fga,
            'time_oreb_to_fga': time_oreb_to_fga,
        })

    result = pd.DataFrame(possession_rows)

    prev_enders = ['start_of_period']
    for i_r in range(1, len(result)):
        if result.iloc[i_r]['period'] != result.iloc[i_r - 1]['period']:
            prev_enders.append('start_of_period')
        else:
            prev_enders.append(result.iloc[i_r - 1]['refined_outcome'])
    result['prev_poss_ender'] = prev_enders

    # --- Fix 4: Filter out end_period possessions from enriched output ---
    result = result[result['raw_outcome'] != 'end_period'].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Main: iterate over all plays files
# ---------------------------------------------------------------------------
data_dir = '.'

plays_files = sorted([f for f in os.listdir(data_dir) if f.startswith('plays_') and f.endswith('.csv')])

for plays_file in plays_files:
    suffix = plays_file.replace('plays_', '').replace('.csv', '')

    print(f'\n{"="*65}')
    print(f'Processing {plays_file} ...')
    print(f'{"="*65}')

    plays_df = pd.read_csv(os.path.join(data_dir, plays_file))
    game_ids = plays_df['gameId'].unique()
    n_games = len(game_ids)
    print(f'  {n_games} games, {len(plays_df):,} plays')

    all_poss = []
    all_enriched = []
    failed = []

    t0 = time.time()
    for i, gid in enumerate(game_ids):
        game_df = plays_df[plays_df['gameId'] == gid].copy()
        try:
            poss_df = track_possessions_v2(game_df)
            enriched_df = classify_possessions(poss_df, game_df)
            all_poss.append(poss_df)
            all_enriched.append(enriched_df)
        except Exception as e:
            failed.append((gid, str(e)))

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f'    {i+1}/{n_games} games ({elapsed:.0f}s)')

    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.0f}s. Failed: {len(failed)}')
    if failed:
        for gid, err in failed[:5]:
            print(f'    Game {gid}: {err}')

    if all_poss:
        combined_poss = pd.concat(all_poss, ignore_index=True)
        fname = f'possessions_{suffix}.csv'
        combined_poss.to_csv(os.path.join(data_dir, fname), index=False)
        print(f'  Saved {fname}: {len(combined_poss):,} rows')
        made_fg = (combined_poss['outcome'] == 'made_fg').sum()
        made_ft = (combined_poss['outcome'] == 'made_ft').sum()
        print(f'    made_fg={made_fg:,}, made_ft={made_ft:,}')

    if all_enriched:
        combined_enriched = pd.concat(all_enriched, ignore_index=True)
        fname = f'possessions_enriched_{suffix}.csv'
        combined_enriched.to_csv(os.path.join(data_dir, fname), index=False)
        print(f'  Saved {fname}: {len(combined_enriched):,} rows')
        print(f'    Outcome distribution:')
        print(combined_enriched['refined_outcome'].value_counts().head(10).to_string())

print('\n\nALL DONE.')
