# -*- coding: utf-8 -*-
"""
diag_dead_ball_rebound.py

Investigate the remaining dead_ball_rebound prev_poss_ender possessions after v7.
Focus: the 302 possessions that score exactly 1 pt — are they still phantom FT splits
or legitimate (e.g. foul-on-inbound and-1)?

Outputs:
  1. Summary counts
  2. Sample 20 of the 1-pt cases showing the preceding possession's plays + the poss itself
  3. Breakdown of the 1,162 FT-only possessions by their raw_outcome
"""

import glob, os, sys
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def _concat(subdir, **kwargs):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, subdir, '*.csv')))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, **kwargs) for p in paths], ignore_index=True)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading possessions_enriched...")
pe = _concat('possessions_enriched')
pe = pe.drop_duplicates(subset=['gameId', 'possession_id', 'possession_team'])
print(f"  {len(pe):,} rows")

print("Loading possessions (raw play rows)...")
pr = _concat('possessions')
pr = pr.drop_duplicates(subset=['play_id', 'gameId', 'possession_team'])
print(f"  {len(pr):,} rows")

print("Loading plays...")
pl = _concat('plays', usecols=['id', 'scoreValue', 'scoringPlay', 'playType', 'playText', 'team', 'period', 'secondsRemaining', 'gameId'])
pl = pl.drop_duplicates(subset=['id'])
print(f"  {len(pl):,} rows")


# ── Compute points per possession ─────────────────────────────────────────────
pr_scored = pr.merge(
    pl[['id', 'scoreValue', 'scoringPlay', 'playType', 'playText', 'team', 'period', 'secondsRemaining']],
    left_on='play_id', right_on='id',
    how='left'
)
pr_scored['points'] = pr_scored['scoreValue'].where(pr_scored['scoringPlay'] == True, 0).fillna(0)

poss_pts = (
    pr_scored
    .groupby(['gameId', 'possession_id', 'possession_team'], as_index=False)
    ['points'].sum()
)

df = pe.merge(poss_pts, on=['gameId', 'possession_id', 'possession_team'], how='left')
df['points'] = df['points'].fillna(0)
df = df[df['possession_team'].notna()].copy()

# ── DBR subset ────────────────────────────────────────────────────────────────
dbr = df[df['prev_poss_ender'] == 'dead_ball_rebound'].copy()
print(f"\ndead_ball_rebound possessions: {len(dbr):,}")
print(f"Points distribution:")
print(dbr['points'].value_counts().sort_index())

# ── Section 1: FT-only breakdown ──────────────────────────────────────────────
# Tag which possessions have only FTs (no FG attempts)
has_fg = (
    pr_scored[pr_scored['playType'].isin({'JumpShot', 'LayUpShot', 'DunkShot', 'TipShot'})]
    .groupby(['gameId', 'possession_id', 'possession_team'])
    .size()
    .reset_index(name='n_fg')
)
dbr = dbr.merge(has_fg, on=['gameId', 'possession_id', 'possession_team'], how='left')
dbr['ft_only'] = dbr['n_fg'].isna()

ft_only = dbr[dbr['ft_only']].copy()
print(f"\nFT-only dead_ball_rebound possessions: {len(ft_only):,}")
print("raw_outcome distribution for FT-only:")
print(ft_only['raw_outcome'].value_counts().head(15))
print("\npoints distribution for FT-only:")
print(ft_only['points'].value_counts().sort_index())


# ── Section 2: Deep-dive the 302 1-pt possessions ────────────────────────────
one_pt = dbr[dbr['points'] == 1].copy()
print(f"\n1-pt dead_ball_rebound possessions: {len(one_pt):,}")
print("raw_outcome distribution:")
print(one_pt['raw_outcome'].value_counts().head(15))
print("ft_only breakdown:")
print(one_pt['ft_only'].value_counts())

# For each, find the preceding possession (possession_id - 1) in the same game
# and show the raw play rows for both possessions

# Build play-level lookup: (gameId, possession_id, possession_team) -> list of plays
play_lookup = (
    pr_scored[['gameId', 'possession_id', 'possession_team', 'play_id',
               'playType', 'playText', 'team_y', 'period', 'secondsRemaining', 'scoringPlay', 'scoreValue']]
    .rename(columns={'team_y': 'team'})
    .copy()
)

def get_plays(game_id, poss_id, poss_team=None):
    mask = (play_lookup['gameId'] == game_id) & (play_lookup['possession_id'] == poss_id)
    if poss_team:
        mask &= (play_lookup['possession_team'] == poss_team)
    return play_lookup[mask][['playType', 'playText', 'team', 'period', 'secondsRemaining', 'scoringPlay', 'scoreValue']]

# Look at FT-only 1-pt possessions (most suspicious)
ft_only_1pt = one_pt[one_pt['ft_only']].copy()
print(f"\nFT-only 1-pt dead_ball_rebound possessions: {len(ft_only_1pt):,}")

# Sample up to 30
sample = ft_only_1pt.sample(min(30, len(ft_only_1pt)), random_state=42)

print("\n" + "="*80)
print("SAMPLE: FT-only 1-pt dead_ball_rebound possessions")
print("Showing PREV possession plays then CURRENT possession plays")
print("="*80)

for i, (_, row) in enumerate(sample.iterrows()):
    gid = row['gameId']
    pid = row['possession_id']
    pteam = row['possession_team']
    prev_pid = pid - 1

    print(f"\n--- Case {i+1}: gameId={gid}, poss_id={pid}, team={pteam} ---")
    print(f"  raw_outcome={row.get('raw_outcome','?')}  period={row.get('period','?')}  start_s={row.get('start_seconds','?'):.0f}  end_s={row.get('end_seconds','?'):.0f}")

    # Previous possession (any team)
    prev_plays = get_plays(gid, prev_pid)
    print(f"  PREV possession (id={prev_pid}):")
    if len(prev_plays) == 0:
        print("    (no plays found)")
    else:
        for _, p in prev_plays.sort_values('secondsRemaining', ascending=False).iterrows():
            flag = "* SCORE" if p['scoringPlay'] else ""
            print(f"    [{p['period']}] {p['secondsRemaining']:.0f}s  {p['playType']:25s}  {str(p['playText'])[:60].encode('ascii','replace').decode()[:60]:60s}  {flag}")

    # Current possession
    curr_plays = get_plays(gid, pid, pteam)
    print(f"  CURR possession (id={pid}, team={pteam}):")
    if len(curr_plays) == 0:
        print("    (no plays found)")
    else:
        for _, p in curr_plays.sort_values('secondsRemaining', ascending=False).iterrows():
            flag = f"* SCORE +{p['scoreValue']:.0f}" if p['scoringPlay'] else ""
            print(f"    [{p['period']}] {p['secondsRemaining']:.0f}s  {p['playType']:25s}  {str(p['playText'])[:60].encode('ascii','replace').decode()[:60]:60s}  {flag}")

    if i >= 19:  # stop after 20
        break


# ── Section 3: Non-FT 1-pt possessions ───────────────────────────────────────
non_ft_1pt = one_pt[~one_pt['ft_only']].copy()
print(f"\n\nNon-FT 1-pt dead_ball_rebound possessions (with FG attempts): {len(non_ft_1pt):,}")
if len(non_ft_1pt) > 0:
    print("raw_outcome:")
    print(non_ft_1pt['raw_outcome'].value_counts().head(10))

    sample2 = non_ft_1pt.sample(min(10, len(non_ft_1pt)), random_state=1)
    print("\n" + "="*80)
    print("SAMPLE: Non-FT 1-pt dead_ball_rebound possessions")
    print("="*80)
    for i, (_, row) in enumerate(sample2.iterrows()):
        gid = row['gameId']
        pid = row['possession_id']
        pteam = row['possession_team']
        prev_pid = pid - 1
        print(f"\n--- Case {i+1}: gameId={gid}, poss_id={pid}, team={pteam} ---")
        curr_plays = get_plays(gid, pid, pteam)
        prev_plays = get_plays(gid, prev_pid)
        print(f"  PREV (id={prev_pid}):")
        for _, p in prev_plays.sort_values('secondsRemaining', ascending=False).iterrows():
            print(f"    [{p['period']}] {p['secondsRemaining']:.0f}s  {p['playType']:25s}  {str(p['playText'])[:60].encode('ascii','replace').decode()}")
        print(f"  CURR (id={pid}):")
        for _, p in curr_plays.sort_values('secondsRemaining', ascending=False).iterrows():
            flag = f"* +{p['scoreValue']:.0f}" if p['scoringPlay'] else ""
            print(f"    [{p['period']}] {p['secondsRemaining']:.0f}s  {p['playType']:25s}  {str(p['playText'])[:60].encode('ascii','replace').decode()}  {flag}")


# ── Section 4: Missed-FT check — does the PREV possession end with a missed FT? ─
print("\n\n" + "="*80)
print("SECTION 4: For ALL FT-only 1-pt DBR possessions, how does the PREV poss end?")
print("="*80)
# Get the final play type of the previous possession for each 1-pt FT-only case
prev_poss_ids = ft_only_1pt[['gameId', 'possession_id']].copy()
prev_poss_ids['prev_poss_id'] = ft_only_1pt['possession_id'].values - 1

prev_final_plays = []
for _, row in prev_poss_ids.iterrows():
    plays = get_plays(row['gameId'], row['prev_poss_id'])
    if len(plays) == 0:
        prev_final_plays.append('(none)')
        continue
    # Last play by clock
    last = plays.sort_values('secondsRemaining').iloc[0]
    prev_final_plays.append(last['playType'])

ft_only_1pt = ft_only_1pt.copy()
ft_only_1pt['prev_final_playType'] = prev_final_plays

print("Last playType of the PREVIOUS possession for FT-only 1-pt DBR cases:")
print(ft_only_1pt['prev_final_playType'].value_counts())

print("\nDone.")
