import pandas as pd 
import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity

from similarity import Cosine
from scipy.io import mmwrite, mmread

import time

import pandas as pd 
import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity

from similarity import Cosine
from scipy.io import mmwrite, mmread

import time

int_data=pd.read_csv('./Data/train_final.csv', sep='\t', header=0)

int_data = int_data.sort_values(['playlist_id', 'track_id'], ascending=False)
int_data = int_data.drop_duplicates(subset=['playlist_id', 'track_id'], keep='first')

tracks = int_data.track_id.unique() # items
playlists = int_data.playlist_id.unique() # users

track_to_idx = pd.Series(data=np.arange(len(tracks)), index=tracks)
playlist_to_idx = pd.Series(data=np.arange(len(playlists)), index=playlists)

idx_to_track = pd.Series(data=track_to_idx.index, index=track_to_idx.data)
idx_to_playlist = pd.Series(data=playlist_to_idx.index, index=playlist_to_idx.data)

target_playlists = pd.read_csv('./Data/target_playlists.csv', header=0)

created_playlists = target_playlists[target_playlists['playlist_id'].isin(playlists) == True]
created_playlists = created_playlists.values.ravel()

target_playlist_data = int_data[int_data['playlist_id'].isin(created_playlists)==True]
tracks_to_compute = target_playlist_data['track_id'].unique()

#read item profiles
data=pd.read_csv('./Data/tracks_final.csv', sep='\t', header=0, usecols=['track_id', 'artist_id'])# 'discipline_id', 'industry_id', 'country', 'region', 'latitude', 'longitude', 'employment', 'active_during_test', 'tags', 'title'])
data = data.fillna(0) #???

artists = data['artist_id'].unique()
tracks_d = data['track_id'].unique()

ptrack_to_idx = pd.Series(data=np.arange(data.shape[0]), index=data['track_id'])
idx_to_ptrack = pd.Series(index=ptrack_to_idx.data, data=ptrack_to_idx.index)

artist_to_idx = pd.Series(data=np.arange(data.shape[0]), index=data['track_id'])
idx_to_artist = pd.Series(data=artist_to_idx.index, index=artist_to_idx.data)

artists_to_idx = pd.Series(index=artists, data=np.arange(artists.shape[0]))

icm = sps.csc_matrix((data.shape[0], artists.shape[0]))

#fancy indexing
icm[np.arange(0,data.shape[0]), artists_to_idx[data.iloc[:,1].values]] = 1

#icm for rated items by target users
tdata = data[data['track_id'].isin(tracks_to_compute) == True]

ttracks = tdata['track_id'].unique()

ttrack_to_idx = pd.Series(data=np.arange(len(ttracks)), index=ttracks)
idx_to_ttrack = pd.Series(index=ttrack_to_idx.data, data=ttrack_to_idx.index)

ticm = sps.csc_matrix((tdata.shape[0], artists.shape[0]))

ticm[np.arange(0,tdata.shape[0]), artist_to_idx[tdata.iloc[:,1].values]] = 1

def compute_sim():
    c = Cosine()

    sim = c.compute(icm, ticm)

    count = 1
    for i in titems:
        sim[titem_to_idx[i], pitem_to_idx[i]] = 0.0
        print("Finished for: ", count)
        count += 1

    mmwrite("./Data/item_similarity1.mtx", sim)

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

# sim = mmread("./Data/item_similarity1.mtx")
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

# np.savetxt('./data/result_content2.csv', result, fmt='%d, %d %d %d %d %d')
