# -*- coding: utf-8 -*-
"""
diag_dbr_samples.py

Show play-by-play samples of dead_ball_rebound possessions, bucketed by
what actually starts the possession (first meaningful play).

Run:  python diag_dbr_samples.py
"""

import glob, os, pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def _concat(subdir, **kw):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, subdir, '*.csv')))
    return pd.concat([pd.read_csv(p, **kw) for p in paths], ignore_index=True) if paths else pd.DataFrame()

_SKIP = {'Substitution', 'Official TV Timeout', 'OfficialTVTimeOut',
         'ShortTimeOut', 'RegularTimeOut', ''}

print("Loading data...")
pe = _concat('possessions_enriched').drop_duplicates(['gameId','possession_id','possession_team'])
pr = _concat('possessions').drop_duplicates(['play_id','gameId','possession_team'])
pl = _concat('plays', usecols=['id','scoreValue','scoringPlay','playType','playText',
                                'period','secondsRemaining','gameId']).drop_duplicates('id')
print(f"  pe={len(pe):,}  pr={len(pr):,}  pl={len(pl):,}")

# Join plays onto possessions
pr_full = pr.merge(
    pl.rename(columns={'id': 'play_id'}),
    on=['play_id', 'gameId'], how='left'
)
pr_full['points'] = pr_full['scoreValue'].where(pr_full['scoringPlay'] == True, 0).fillna(0)

# Points per possession
ppts = pr_full.groupby(['gameId','possession_id','possession_team'], as_index=False)['points'].sum()
df = pe.merge(ppts, on=['gameId','possession_id','possession_team'], how='left')
df['points'] = df['points'].fillna(0)

# ── DBR possessions ───────────────────────────────────────────────────────────
dbr = df[df['prev_poss_ender'] == 'dead_ball_rebound'].copy()
print(f"\ndead_ball_rebound possessions: {len(dbr):,}")

# ── First MEANINGFUL play of each DBR possession (skip subs/timeouts) ─────────
curr_plays_full = (
    pr_full[pr_full['playType'].notna() & ~pr_full['playType'].isin(_SKIP)]
    .merge(dbr[['gameId','possession_id','possession_team']], on=['gameId','possession_id','possession_team'])
    .dropna(subset=['secondsRemaining'])
    .sort_values('secondsRemaining', ascending=False)
    .drop_duplicates(['gameId','possession_id','possession_team'], keep='first')
    [['gameId','possession_id','possession_team','playType','playText']]
    .rename(columns={'playType': 'first_meaningful_play', 'playText': 'first_play_text'})
)

dbr = dbr.merge(curr_plays_full, on=['gameId','possession_id','possession_team'], how='left')

print("\nFirst meaningful play of DBR possession:")
print(dbr['first_meaningful_play'].value_counts().head(15))
print()

# ── Helper: print prev + curr possession plays side by side ───────────────────
def show_possession(game_id, poss_id, poss_team, label=""):
    mask = (pr_full['gameId'] == game_id) & (pr_full['possession_id'] == poss_id)
    if poss_team:
        mask &= (pr_full['possession_team'] == poss_team)
    rows = pr_full[mask].sort_values('secondsRemaining', ascending=False)
    print(f"  {label} (id={poss_id}{', team='+str(poss_team) if poss_team else ''}):")
    if len(rows) == 0:
        print("    (no plays)")
        return
    for _, p in rows.iterrows():
        score_flag = f"  *** +{p['scoreValue']:.0f}" if p['scoringPlay'] else ""
        txt = str(p.get('playText', '')).encode('ascii', 'replace').decode()[:65]
        print(f"    [{p['period']}] {p['secondsRemaining']:5.0f}s  {str(p['playType']):26s}  {txt}{score_flag}")

def show_sample(bucket_label, sample_df, n=8):
    sample = sample_df.sample(min(n, len(sample_df)), random_state=42)
    print("=" * 90)
    print(f"BUCKET: {bucket_label}  ({len(sample_df):,} total)")
    print("=" * 90)
    for i, (_, row) in enumerate(sample.iterrows()):
        gid  = row['gameId']
        pid  = row['possession_id']
        pt   = row['possession_team']
        prev = pid - 1
        print(f"\n  --- {i+1}. gameId={gid}  poss={pid}  team={pt}  pts={row['points']:.0f}  outcome={row.get('raw_outcome','?')} ---")
        show_possession(gid, prev, None,  "PREV")
        show_possession(gid, pid,  pt,    "CURR")
    print()

# ── Bucket 1: fouled on inbound ───────────────────────────────────────────────
foul_inbound = dbr[dbr['first_meaningful_play'] == 'PersonalFoul']
show_sample("PersonalFoul first (fouled on inbound)", foul_inbound)

# ── Bucket 2: FT first (already in FT sequence) ───────────────────────────────
ft_first = dbr[dbr['first_meaningful_play'] == 'MadeFreeThrow']
show_sample("MadeFreeThrow first (already in FT sequence)", ft_first)

# ── Bucket 3: JumpShot first (clean half-court poss) ─────────────────────────
jump_first = dbr[dbr['first_meaningful_play'] == 'JumpShot']
show_sample("JumpShot first (clean possession)", jump_first)

# ── Bucket 4: Turnover first ──────────────────────────────────────────────────
to_first = dbr[dbr['first_meaningful_play'].isin({'Lost Ball Turnover', 'Bad Pass Turnover'})]
show_sample("Turnover first (lost it on inbound)", to_first)

# ── Bucket 5: LayUp first ─────────────────────────────────────────────────────
layup_first = dbr[dbr['first_meaningful_play'] == 'LayUpShot']
show_sample("LayUpShot first", layup_first)
