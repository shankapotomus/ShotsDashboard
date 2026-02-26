# -*- coding: utf-8 -*-
import glob, os, pandas as pd

def _concat(subdir, **kw):
    paths = sorted(glob.glob(os.path.join(subdir, '*.csv')))
    return pd.concat([pd.read_csv(p, **kw) for p in paths], ignore_index=True) if paths else pd.DataFrame()

print("Loading...")
pe = _concat('possessions_enriched').drop_duplicates(['gameId','possession_id','possession_team'])
pr = _concat('possessions').drop_duplicates(['play_id','gameId','possession_team'])
pl = _concat('plays', usecols=['id','scoreValue','scoringPlay','playType','period','secondsRemaining','gameId']).drop_duplicates('id')

pr_full = pr.merge(pl.rename(columns={'id':'play_id'}), on=['play_id','gameId'], how='left')

dbr = pe[pe['prev_poss_ender']=='dead_ball_rebound'].copy()
dbr['prev_poss_id'] = dbr['possession_id'] - 1

# Get prev possession's team from possessions_enriched
pe_prev = pe[['gameId','possession_id','possession_team']].rename(
    columns={'possession_id':'prev_poss_id', 'possession_team':'prev_team'})
dbr = dbr.merge(pe_prev, on=['gameId','prev_poss_id'], how='left')

# Last play: sort by secondsRemaining asc (lowest = latest), play_id desc (tiebreak to highest id)
pr_last = (
    pr_full.dropna(subset=['secondsRemaining','playType'])
    .sort_values(['secondsRemaining', 'play_id'], ascending=[True, False])
    .drop_duplicates(['gameId','possession_id','possession_team'], keep='first')
    .rename(columns={'possession_id':'prev_poss_id',
                     'possession_team':'prev_team',
                     'playType':'prev_last_play'})
    [['gameId','prev_poss_id','prev_team','prev_last_play']]
)

dbr2 = dbr.merge(pr_last, on=['gameId','prev_poss_id','prev_team'], how='left')
print('\n=== Last PLAY of prev possession (secondsRemaining asc, play_id desc) ===')
print(dbr2['prev_last_play'].value_counts().head(15))
n_dbr = (dbr2['prev_last_play'] == 'Dead Ball Rebound').sum()
print(f'\nTotal: {len(dbr2):,}  |  Dead Ball Rebound: {n_dbr:,}  ({100*n_dbr/len(dbr2):.1f}%)')

# What's NOT a Dead Ball Rebound?
non_dbr = dbr2[dbr2['prev_last_play'] != 'Dead Ball Rebound'].copy()
print(f'\nStill non-DBR last plays: {len(non_dbr):,}')
print(non_dbr['prev_last_play'].value_counts().head(10))

# Sample 10 non-DBR cases and show the full prev possession plays
print('\n--- Sampling 10 non-DBR-ending prev possessions ---')
sample = non_dbr.sample(min(10, len(non_dbr)), random_state=7)

for _, row in sample.iterrows():
    gid  = row['gameId']
    ppid = row['prev_poss_id']
    pteam = row['prev_team']
    print(f'\n  gameId={gid}  prev_poss_id={ppid}  prev_team={pteam}  prev_last={row["prev_last_play"]}')
    plays = (
        pr_full[(pr_full['gameId']==gid) &
                (pr_full['possession_id']==ppid) &
                (pr_full['possession_team']==pteam)]
        .sort_values(['secondsRemaining','play_id'], ascending=[False, True])
    )
    for _, p in plays.iterrows():
        txt = str(p.get('playText', '')).encode('ascii','replace').decode()[:65]
        print(f"    [{p['period']}] {p['secondsRemaining']:5.0f}s  id={p['play_id']:9.0f}  {str(p['playType']):26s}  {txt}")
