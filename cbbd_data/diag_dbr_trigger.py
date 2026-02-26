# -*- coding: utf-8 -*-
"""What triggered the Dead Ball Rebound in the previous possession?"""
import glob, os, pandas as pd

def _concat(subdir, **kw):
    paths = sorted(glob.glob(os.path.join(subdir, '*.csv')))
    return pd.concat([pd.read_csv(p, **kw) for p in paths], ignore_index=True) if paths else pd.DataFrame()

print("Loading...")
pe = _concat('possessions_enriched').drop_duplicates(['gameId','possession_id','possession_team'])
pr = _concat('possessions').drop_duplicates(['play_id','gameId','possession_team'])
pl = _concat('plays', usecols=['id','playType','playText','period','secondsRemaining','gameId']).drop_duplicates('id')

pr_full = pr.merge(pl.rename(columns={'id':'play_id'}), on=['play_id','gameId'], how='left')

dbr = pe[pe['prev_poss_ender']=='dead_ball_rebound'].copy()
dbr['prev_poss_id'] = dbr['possession_id'] - 1

pe_prev = pe[['gameId','possession_id','possession_team']].rename(
    columns={'possession_id':'prev_poss_id','possession_team':'prev_team'})
dbr = dbr.merge(pe_prev, on=['gameId','prev_poss_id'], how='left')

# Sort all prev-possession plays correctly: period asc, secondsRemaining desc, play_id asc
# (same order as the tracker itself uses)
_SKIP = {'Substitution','Official TV Timeout','OfficialTVTimeOut','ShortTimeOut','RegularTimeOut',''}

prev_all = pr_full.merge(
    dbr[['gameId','prev_poss_id','prev_team']].drop_duplicates(),
    left_on=['gameId','possession_id','possession_team'],
    right_on=['gameId','prev_poss_id','prev_team'],
    how='inner'
).sort_values(['gameId','prev_poss_id','secondsRemaining','play_id'],
              ascending=[True,True,True,False])   # secondsRemaining asc = latest first; play_id desc = tiebreak

# The DBR itself is the last row — we want the play BEFORE the last meaningful play
# Group by (gameId, prev_poss_id, prev_team), get second-to-last non-skip play
def get_trigger(group):
    # Filter out skip types
    g = group[~group['playType'].isin(_SKIP) & group['playType'].notna()]
    g = g.sort_values(['secondsRemaining','play_id'], ascending=[True, False])
    # g.iloc[0] = the DBR (last play), g.iloc[1] = play before it
    if len(g) < 2:
        return 'only_one_play'
    return g.iloc[1]['playType']

trigger = (
    prev_all
    .groupby(['gameId','prev_poss_id','prev_team'], group_keys=False)
    .apply(get_trigger)
    .reset_index(name='trigger_play')
)

# Merge back
dbr2 = dbr.merge(trigger, on=['gameId','prev_poss_id','prev_team'], how='left')

print('\n=== Play that TRIGGERED the Dead Ball Rebound (prev possession) ===')
print(dbr2['trigger_play'].value_counts().head(20))
print(f'\nTotal: {len(dbr2):,}')
