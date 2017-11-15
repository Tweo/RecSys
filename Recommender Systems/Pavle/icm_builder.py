# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity

from similarity import Cosine
from scipy.io import mmwrite, mmread

import time

# import proba as prb
#read interactions
int_data=pd.read_csv('./Data/train_final.csv', sep='\t', header=0)
int_data = int_data.sort_values(['playlist_id', 'track_id'], ascending=False)
int_data = int_data.drop_duplicates(subset=['playlist_id', 'track_id'], keep='first')


items = int_data['track_id'].unique()
users = int_data['playlist_id'].unique()

item_to_idx = pd.Series(data=np.arange(len(items)), index=items)
user_to_idx = pd.Series(data=np.arange(len(users)), index=users)

idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
idx_to_user = pd.Series(index=user_to_idx.data, data=user_to_idx.index)

#target users
tusers = pd.read_csv('./Data/target_playlists.csv', header=0)
tusers_that_rated = tusers[tusers['playlist_id'].isin(users) == True]
tusers_that_rated = tusers_that_rated.values.ravel()

#get item_ids that the traget users rated
#compute similarity only for these items
tusers_data = int_data[int_data['playlist_id'].isin(tusers_that_rated) == True]
items_to_compute = tusers_data['track_id'].unique()

#read item profiles
data=pd.read_csv('./Data/tracks_final.csv', sep='\t', header=0, usecols=['track_id', 'artist_id'])# 'discipline_id', 'industry_id', 'country', 'region', 'latitude', 'longitude', 'employment', 'active_during_test', 'tags', 'title'])
data = data.fillna(0)

# #not activeitems
# not_active_items = data.iloc[:][data['active_during_test'] == 0]
# not_active_items = not_active_items['id'].values



c = data['artist_id'].unique()
# d = data['discipline_id'].unique()
# i = data['industry_id'].unique()
# cn = data['country'].unique()
# r = data['region'].unique()
# la = data['latitude'].unique()
# lg = data['longitude'].unique()
# e = data['employment'].unique()
# tags = prb.get_disj_tags()
# titles = prb.get_disj_titles()

pitem_to_idx = pd.Series(data=np.arange(data.shape[0]), index=data['track_id'])
idx_to_pitem = pd.Series(index=pitem_to_idx.data, data=pitem_to_idx.index)



c_to_idx = pd.Series(index=c, data=np.arange(c.shape[0]))
# d_to_idx = pd.Series(index=d, data=np.arange(c.shape[0], c.shape[0] + d.shape[0]))
# i_to_idx = pd.Series(index=i, data=np.arange(c.shape[0] + d.shape[0], c.shape[0] + d.shape[0] + i.shape[0]))
# cn_to_idx = pd.Series(index=cn, data=np.arange(c.shape[0] + d.shape[0] + i.shape[0], c.shape[0] + d.shape[0] + i.shape[0] + cn.shape[0]))
# r_to_idx = pd.Series(index=r, data=np.arange(c.shape[0] + d.shape[0] + i.shape[0] +cn.shape[0], c.shape[0] + d.shape[0] + i.shape[0] + cn.shape[0] + r.shape[0]))
# la_to_idx = pd.Series(index=la, data=np.arange(c.shape[0] + d.shape[0] + i.shape[0] +cn.shape[0] + r.shape[0], c.shape[0] + d.shape[0] + i.shape[0] + cn.shape[0] + r.shape[0] + la.shape[0]))
# lg_to_idx = pd.Series(index=lg, data=np.arange(c.shape[0] + d.shape[0] + i.shape[0] +cn.shape[0] + r.shape[0] + la.shape[0], c.shape[0] + d.shape[0] + i.shape[0] + cn.shape[0] + r.shape[0] + la.shape[0] + lg.shape[0]))
# e_to_idx = pd.Series(index=e, data=np.arange(c.shape[0] + d.shape[0] + i.shape[0] +cn.shape[0] + r.shape[0] + la.shape[0] + lg.shape[0], c.shape[0] + d.shape[0] + i.shape[0] + cn.shape[0] + r.shape[0] + la.shape[0] + lg.shape[0] + e.shape[0]))
# tags_to_idx = pd.Series(index=tags, data=np.arange(c.shape[0] + d.shape[0] + i.shape[0] +cn.shape[0] + r.shape[0] + la.shape[0] + lg.shape[0], c.shape[0] + d.shape[0] + i.shape[0] + cn.shape[0] + r.shape[0] + la.shape[0] + lg.shape[0] + e.shape[0] + len(tags)))
# titles_to_idx = pd.Series(index=titles, data=np.arange(c.shape[0] + d.shape[0] + i.shape[0] +cn.shape[0] + r.shape[0] + la.shape[0] + lg.shape[0], c.shape[0] + d.shape[0] + i.shape[0] + cn.shape[0] + r.shape[0] + la.shape[0] + lg.shape[0] + e.shape[0] + len(tags) +len(tags)))


icm = sps.csc_matrix((data.shape[0], c.shape[0]))# + d.shape[0] + i.shape[0] + cn.shape[0] + r.shape[0] + la.shape[0] + lg.shape[0] + e.shape[0] + len(tags) + len(titles)))

#fancy indexing
icm[np.arange(0,data.shape[0]), c_to_idx[data.iloc[:,1].values]] = 1
# icm[np.arange(0,data.shape[0]), d_to_idx[data.iloc[:,2].values]] = 1
# icm[np.arange(0,data.shape[0]), i_to_idx[data.iloc[:,3].values]] = 1
# icm[np.arange(0,data.shape[0]), cn_to_idx[data.iloc[:,4].values]] = 1
# icm[np.arange(0,data.shape[0]), r_to_idx[data.iloc[:,5].values]] = 1
# icm[np.arange(0,data.shape[0]), la_to_idx[data.iloc[:,6].values]] = 1
# icm[np.arange(0,data.shape[0]), lg_to_idx[data.iloc[:,7].values]] = 1
# icm[np.arange(0,data.shape[0]), e_to_idx[data.iloc[:,8].values]] = 1    
# icm[np.arange(0,data.shape[0]), tags_to_idx[data.iloc[:,9].values]] = 1
# icm[np.arange(0,data.shape[0]), titles_to_idx[data.iloc[:,10].values]] = 1    

#icm for rated items by target users
tdata = data[data['track_id'].isin(items_to_compute) == True]

titems = tdata['track_id'].unique()

titem_to_idx = pd.Series(data=np.arange(len(titems)), index=titems)
idx_to_titem = pd.Series(index=titem_to_idx.data, data=titem_to_idx.index)

ticm = sps.csc_matrix((tdata.shape[0], c.shape[0])) # + d.shape[0] + i.shape[0] + cn.shape[0] + r.shape[0] + la.shape[0] + lg.shape[0] + e.shape[0]))

ticm[np.arange(0,tdata.shape[0]), c_to_idx[tdata.iloc[:,1].values]] = 1
# ticm[np.arange(0,tdata.shape[0]), d_to_idx[tdata.iloc[:,2].values]] = 1
# ticm[np.arange(0,tdata.shape[0]), i_to_idx[tdata.iloc[:,3].values]] = 1
# ticm[np.arange(0,tdata.shape[0]), cn_to_idx[tdata.iloc[:,4].values]] = 1
# ticm[np.arange(0,tdata.shape[0]), r_to_idx[tdata.iloc[:,5].values]] = 1
# ticm[np.arange(0,tdata.shape[0]), la_to_idx[tdata.iloc[:,6].values]] = 1
# ticm[np.arange(0,tdata.shape[0]), lg_to_idx[tdata.iloc[:,7].values]] = 1
# ticm[np.arange(0,tdata.shape[0]), e_to_idx[tdata.iloc[:,8].values]] = 1
# ticm[np.arange(0,tdata.shape[0]), tags_to_idx[tdata.iloc[:,9].values]] = 1
# ticm[np.arange(0,tdata.shape[0]), titles_to_idx[tdata.iloc[:,10].values]] = 1

def compute_sim():
	c = Cosine()

	sim = c.compute(icm, ticm)

	count = 1
	for i in titems:
		sim[titem_to_idx[i], pitem_to_idx[i]] = 0.0
		print("Finished for: ", count)
		count += 1

	mmwrite("./Data/item_similarity.mtx", sim)

def filter_seen(user_id, ranking, rated_items):

        seen = pitem_to_idx[rated_items].values
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]

def filter_active(ranking):

        active_mask = np.in1d(ranking, pitem_to_idx[not_active_items], assume_unique=True, invert=True)
        return ranking[active_mask]

#estimate rating of all items to user
def recommend(user_id, n=None, exclude_seen=True):

	rated = int_data[int_data['playlist_id'] == user_id]

	rated_items = rated['track_id'].values
	ratings = np.ones(rated.shape[0])#rated['interaction_type'].values
	
	s = sim[titem_to_idx[rated_items], :].toarray()
	suma = s.sum(axis = 0)

	ratings = ratings.reshape(1, ratings.shape[0]).T

	ratings = np.tile(ratings, (1, s.shape[1]))


	s = s*ratings
	s = s.sum(axis = 0)

	s = s/suma

	if exclude_seen:
            s = filter_seen(user_id, s, rated_items)
            # s = filter_active(s)

	s = np.argsort(s)[::-1]


	return s[:n]


# Main


compute_sim()

# sim = mmread("./Data/item_similarity.mtx")
# sim = sim.tocsc()
# #print(sim)


# result = np.zeros((tusers_that_rated.shape[0], 6))

# r = 0

# for u in tusers_that_rated:

# 	result[r,0] = u
# 	result_idx = recommend(u, 5, False)
# 	result[r, 1:] = idx_to_pitem[result_idx].values
# 	r+=1
# 	print("Finished for: ", r)

# np.savetxt('./data/result_content1.csv', result, fmt='%d, %d %d %d %d %d')


