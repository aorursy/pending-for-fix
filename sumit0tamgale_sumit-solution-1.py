#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# In[2]:


# Load module from another directory
import shutil
shutil.copyfile(src="../input/redcarpet.py", dst="../working/redcarpet.py")
from redcarpet import mat_to_sets


# In[3]:


item_file = "../input/talent.pkl"
item_records, COLUMN_LABELS, READABLE_LABELS, ATTRIBUTES = pickle.load(open(item_file, "rb"))
#print(item_records)
item_df = pd.DataFrame(item_records)[ATTRIBUTES + COLUMN_LABELS].fillna(value=0)
ITEM_NAMES = item_df["name"].values
ITEM_IDS = item_df["id"].values
s_items = mat_to_sets(item_df[COLUMN_LABELS].values)
assert len(item_df) == len(s_items), "Item matrix is not the same length as item category set list."
print("Talent:", len(item_df))
print("Categories:", len(COLUMN_LABELS))
print(item_df.shape)
item_df.head()


# In[4]:


def cameo_name(i):
    """
    Show the name and URL of Cameo talent based on its index `i`.
    """
    return "{} (cameo.com/{})".format(ITEM_NAMES[i], ITEM_IDS[i])


# In[5]:


csr_train, csr_test, csr_input, csr_hidden = pickle.load(open("../input/train_test_mat.pkl", "rb"))
m_split = [np.array(csr.todense()) for csr in [csr_train, csr_test, csr_input, csr_hidden]]
m_train, m_test, m_input, m_hidden = m_split
s_train, s_test, s_input, s_hidden = pickle.load(open("../input/train_test_set.pkl", "rb"))
assert len(m_train) == len(s_train), "Train matrix is not the same length as train sets."
assert len(m_test) == len(s_test), "Test matrix is not the same length as test sets."
assert len(m_input) == len(s_input), "Input matrix is not the same length as input sets."
assert len(m_hidden) == len(s_hidden), "Hidden matrix is not the same length as hidden sets."
print("Train Users", len(m_train))
print("Test Users", len(m_test))
print("Minimum Test Items per User:", min(m_test.sum(axis=1)))
print("Minimum Input Items per User:", min(m_input.sum(axis=1)))
print("Minimum Hidden Items per User:", min(m_hidden.sum(axis=1)))
like_df = pd.DataFrame(m_train, columns=ITEM_NAMES)
like_df.head()


# In[6]:


print(like_df.iloc(0)[])
'''def get categories_list(user):
    for like_df
    item_df[item_df['name']=='Perez Hilton']['categories'][0])'''


# In[7]:


from redcarpet import mapk_score, uhr_score


# In[8]:


help(mapk_score)


# In[9]:


help(uhr_score)


# In[10]:


from redcarpet import jaccard_sim, cosine_sim


# In[11]:


help(jaccard_sim)


# In[12]:


help(cosine_sim)


# In[13]:


from redcarpet import collaborative_filter, content_filter, weighted_hybrid


# In[14]:


help(collaborative_filter)


# In[15]:


help(content_filter)


# In[16]:


help(weighted_hybrid)


# In[17]:


from redcarpet import get_recs
from redcarpet import show_user_recs, show_item_recs, show_user_detail
from redcarpet import show_apk_dist, show_hit_dist, show_score_dist


# In[18]:


help(get_recs)


# In[19]:


k_top = 10


# In[20]:


print("Model: Collaborative Filtering with Jacccard Similarity (j=10)")
collab_jac10 = collaborative_filter(s_train, s_input, sim_fn=jaccard_sim, j=10, k=k_top)
print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(collab_jac10), k=k_top)))
print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(collab_jac10), k=k_top)))


# In[21]:


print("Model: Collaborative Filtering with Jacccard Similarity (j=40)")
collab_jac40 = collaborative_filter(s_train, s_input, sim_fn=cosine_sim, j=25, k=k_top)
print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(collab_jac40), k=k_top)))
print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(collab_jac40), k=k_top)))


# In[22]:


idf = show_item_recs(s_hidden, collab_jac10, k=k_top)
idf.sort_values(by=["Hits", "Hit Rate"], ascending=[False, False]).head()


# In[23]:


udf = show_user_recs(s_hidden, collab_jac10, k=k_top)
udf.sort_values(by=["APK", "Hits"], ascending=[False, False]).head()


# In[24]:


show_user_detail(s_input, s_hidden, collab_jac10, uid=0, name_fn=cameo_name)


# In[25]:


print("Model: Collaborative Filtering with Cosine Similarity (j=10)")
collab_cos10 = collaborative_filter(s_train, s_input, sim_fn=cosine_sim, j=10, k=k_top)
print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(collab_cos10), k=k_top)))
print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(collab_cos10), k=k_top)))


# In[26]:


print("Model: Collaborative Filtering with Cosine Similarity (j=40)")
collab_cos40 = collaborative_filter(s_train, s_input, sim_fn=cosine_sim, j=40, k=k_top)
print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(collab_cos40), k=k_top)))
print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(collab_cos40), k=k_top)))


# In[27]:


results = [
    (collab_jac10, "Jaccard (j=10)"),
    (collab_cos10, "Cosine (j=10)")
]


# In[28]:


show_apk_dist(s_hidden, results, k=k_top)


# In[29]:


results = [
    (collab_jac40, "Jaccard (j=40)"),
    (collab_cos40, "Cosine (j=40)")
]


# In[30]:


show_apk_dist(s_hidden, results, k=k_top)


# In[31]:


show_hit_dist(s_hidden, results, k=k_top)


# In[32]:


show_score_dist(results, k=10, bins=np.arange(0.0, 1.1, 0.1))


# In[33]:


print("Model: Hybrid Collaborative Filtering")
print("Similarity: Hybrid (0.2 * Jaccard + 0.8 * Cosine)")
collab_hybrid = weighted_hybrid([
    (collab_jac10, 0.2),
    (collab_cos10, 0.8)
])
print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(collab_hybrid), k=k_top)))
print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(collab_hybrid), k=k_top)))


# In[34]:


print("Model: Hybrid Collaborative Filtering")
print("Similarity: Hybrid (0.2 * Jaccard + 0.8 * Cosine)")
collab_hybrid40 = weighted_hybrid([
    (collab_jac40, 0.2),
    (collab_cos40, 0.8)
])
print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(collab_hybrid40), k=k_top)))
print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(collab_hybrid40), k=k_top)))


# In[35]:


from redcarpet import write_kaggle_recs, download_kaggle_recs


# In[36]:


# Load hold out set
s_hold_input = pickle.load(open("../input/hold_set.pkl", "rb"))
print("Hold Out Set: N = {}".format(len(s_hold_input)))
s_all_input = s_input + s_hold_input
print("All Input:    N = {}".format(len(s_all_input)))


# In[37]:


print("Final Model")
print("Strategy: Collaborative")
print("Similarity: Cosine (j=10)")
# Be sure to use the entire s_input
final_scores = collaborative_filter(s_train, s_all_input, sim_fn=cosine_sim, j=40)
final_recs = get_recs(final_scores)


# In[38]:


print("Final Model")
print("Strategy: Collaborative")
print("Similarity: Cosine (j=40)")
# Be sure to use the entire s_input
final_scores = collaborative_filter(s_train, s_all_input, sim_fn=cosine_sim, j=40)
final_recs = get_recs(final_scores)


# In[39]:


outfile = "kaggle_submission_hybrid_collab.csv"
n_lines = write_kaggle_recs(final_recs, outfile)
print("Wrote predictions for {} users to {}.".format(n_lines, outfile))
download_kaggle_recs(final_recs, outfile)


# In[40]:




