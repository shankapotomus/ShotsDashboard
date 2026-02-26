# -*- coding: utf-8 -*-
"""
What types of missed shots triggered the Dead Ball Rebound in the prev possession?
Joins trigger plays back to shots data for shot_range, is_three, distance, blocked.
"""
import glob, os, pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def _concat(subdir, **kw):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, subdir, '*.csv')))
    return pd.concat([pd.read_csv(p, **kw) for p in paths], ignore_index=True) if paths else pd.DataFrame()

_SKIP = {'Substitution','Official TV Timeout','OfficialTVTimeOut','ShortTimeOut','RegularTimeOut',''}
_SHOT_TYPES = {'JumpShot','LayUpShot','DunkShot','TipShot'}

print("Loading...")
pe  = _concat('possessions_enriched').drop_duplicates(['gameId','possession_id','possession_team'])
pr  = _concat('possessions').drop_duplicates(['play_id','gameId','possession_team'])
pl  = _concat('plays', usecols=['id','playType','playText','period','secondsRemaining','gameId']).drop_duplicates('id')
sh  = _concat('shots', usecols=['play_id','shot_range','is_three','distance','made','x','y','gameId']).drop_duplicates('play_id')
print(f"  pe={len(pe):,}  pr={len(pr):,}  pl={len(pl):,}  shots={len(sh):,}")

pr_full = pr.merge(pl.rename(columns={'id':'play_id'}), on=['play_id','gameId'], how='left')

dbr = pe[pe['prev_poss_ender']=='dead_ball_rebound'].copy()
dbr['prev_poss_id'] = dbr['possession_id'] - 1

pe_prev = pe[['gameId','possession_id','possession_team']].rename(
    columns={'possession_id':'prev_poss_id','possession_team':'prev_team'})
dbr = dbr.merge(pe_prev, on=['gameId','prev_poss_id'], how='left')

# Get all plays for each previous possession, sorted tracker-order
prev_all = pr_full.merge(
    dbr[['gameId','prev_poss_id','prev_team']].drop_duplicates(),
    left_on=['gameId','possession_id','possession_team'],
    right_on=['gameId','prev_poss_id','prev_team'],
    how='inner'
)

# For each prev possession, find:
#   1. The trigger play (play right before the DBR)
#   2. Whether there's a Block Shot between the trigger shot and the DBR
def analyze_prev(group):
    g = group[group['playType'].notna() & ~group['playType'].isin(_SKIP)]
    g = g.sort_values(['secondsRemaining','play_id'], ascending=[True, False])
    # g.iloc[0] = DBR (last play), g.iloc[1] = play before DBR
    if len(g) < 2:
        return pd.Series({'trigger_play_id': None, 'trigger_type': 'only_one_play', 'blocked': False})
    trigger_row = g.iloc[1]
    trigger_type = trigger_row['playType']
    trigger_play_id = trigger_row['play_id']
    # Was it blocked? Check if a Block Shot appears between the trigger and the DBR
    # i.e. any play between iloc[1] and iloc[0] with playType == 'Block Shot'
    mid = g.iloc[2:]  # plays between trigger and DBR (if any)
    blocked = ('Block Shot' in mid['playType'].values) if len(mid) > 0 else False
    return pd.Series({'trigger_play_id': trigger_play_id,
                      'trigger_type': trigger_type,
                      'blocked': blocked})

print("Analyzing prev possessions...")
result = (
    prev_all
    .groupby(['gameId','prev_poss_id','prev_team'], group_keys=False)
    .apply(analyze_prev, include_groups=False)
    .reset_index()
)

dbr2 = dbr.merge(result, on=['gameId','prev_poss_id','prev_team'], how='left')

# Join shots data onto trigger plays
dbr2 = dbr2.merge(
    sh.rename(columns={'play_id':'trigger_play_id'}),
    on=['trigger_play_id','gameId'],
    how='left'
)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nTotal DBR possessions: {len(dbr2):,}")

# Shots only
shots_only = dbr2[dbr2['trigger_type'].isin(_SHOT_TYPES)].copy()
print(f"Shot-triggered DBRs: {len(shots_only):,}\n")

print("=== shot_range breakdown ===")
print(shots_only['shot_range'].value_counts())

print("\n=== blocked vs unblocked ===")
print(dbr2[dbr2['trigger_type'].isin(_SHOT_TYPES | {'Block Shot'})]['blocked'].value_counts())

print("\n=== is_three breakdown (shot-triggered only) ===")
print(shots_only['is_three'].value_counts())

# Combine into a clean table
shots_only['shot_category'] = shots_only.apply(lambda r: (
    'rim'          if r['shot_range'] == 'rim'
    else 'mid_range'   if r['shot_range'] in ('mid_range', 'long_two', 'short_mid_range', 'long_mid_range')
    else 'three_pointer' if r['shot_range'] == 'three_pointer' or r['is_three'] == True
    else 'other'
), axis=1)

print("\n=== Shot category (all DBR-triggering shots) ===")
cats = shots_only['shot_category'].value_counts()
for cat, n in cats.items():
    print(f"  {cat:20s}  {n:5,}  ({100*n/len(shots_only):.1f}%)")

# Among the three_pointers — corner vs above the break
# Corner 3: y < 90 or y > 560 (roughly, on a 0–750 y scale)  OR use shot_range if it splits
threes = shots_only[shots_only['shot_category']=='three_pointer'].copy()
print(f"\n=== 3-pointers: corner vs above the break (n={len(threes):,}) ===")
# Corner 3 roughly: |y - 375| > 260  (far from center of court)
# Use x coordinate: shots near the baseline corners have low y values
threes['corner'] = threes['y'].apply(lambda y: y < 90 or y > 560 if pd.notna(y) else None)
print(threes['corner'].value_counts())

print("\n=== Distance distribution for shot-triggered DBRs ===")
print(shots_only['distance'].describe().round(1))
