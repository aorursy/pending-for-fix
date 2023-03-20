#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array
    :param data: array-like
    :param window_size: size
    :return: the sequence of windows
    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending
    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    target_distribution = np.arange(1, total_events + 1, dtype='float') / total_events
    subarray_distribution = np.cumsum(np.bincount(subindices, minlength=total_events), dtype='float')
    subarray_distribution /= 1.0 * subarray_distribution[-1]
    return np.mean((target_distribution - subarray_distribution) ** 2)


def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.
    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = np.array(predictions)
    masses = np.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[np.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = np.argsort(np.argsort(predictions, kind='mergesort'), kind='mergesort')

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return np.mean(cvms)


def __roc_curve_splitted(data_zero, data_one, sample_weights_zero, sample_weights_one):
    """
    Compute roc curve
    :param data_zero: 0-labeled data
    :param data_one:  1-labeled data
    :param sample_weights_zero: weights for 0-labeled data
    :param sample_weights_one:  weights for 1-labeled data
    :return: roc curve
    """
    labels = [0] * len(data_zero) + [1] * len(data_one)
    weights = np.concatenate([sample_weights_zero, sample_weights_one])
    data_all = np.concatenate([data_zero, data_one])
    fpr, tpr, _ = roc_curve(labels, data_all, sample_weight=weights)
    return fpr, tpr


def compute_ks(data_prediction, mc_prediction, weights_data, weights_mc):
    """
    Compute Kolmogorov-Smirnov (ks) distance between real data predictions cdf and Monte Carlo one.
    :param data_prediction: array-like, real data predictions
    :param mc_prediction: array-like, Monte Carlo data predictions
    :param weights_data: array-like, real data weights
    :param weights_mc: array-like, Monte Carlo weights
    :return: ks value
    """
    assert len(data_prediction) == len(weights_data), 'Data length and weight one must be the same'
    assert len(mc_prediction) == len(weights_mc), 'Data length and weight one must be the same'

    data_prediction, mc_prediction = np.array(data_prediction), np.array(mc_prediction)
    weights_data, weights_mc = np.array(weights_data), np.array(weights_mc)

    assert np.all(data_prediction >= 0.) and np.all(data_prediction <= 1.), 'Data predictions are out of range [0, 1]'
    assert np.all(mc_prediction >= 0.) and np.all(mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'

    weights_data /= np.sum(weights_data)
    weights_mc /= np.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = np.max(np.abs(fpr - tpr))
    return Dnm


def roc_auc_truncated(labels, predictions, tpr_thresholds=(0.2, 0.4, 0.6, 0.8),
                      roc_weights=(4, 3, 2, 1, 0)):
    """
    Compute weighted area under ROC curve.
    :param labels: array-like, true labels
    :param predictions: array-like, predictions
    :param tpr_thresholds: array-like, true positive rate thresholds delimiting the ROC segments
    :param roc_weights: array-like, weights for true positive rate segments
    :return: weighted AUC
    """
    assert np.all(predictions >= 0.) and np.all(predictions <= 1.), 'Data predictions are out of range [0, 1]'
    assert len(tpr_thresholds) + 1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = 0.
    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
    for index in range(1, len(tpr_thresholds)):
        tpr_cut = np.minimum(tpr, tpr_thresholds[index])
        tpr_previous = np.minimum(tpr, tpr_thresholds[index - 1])
        area += roc_weights[index - 1] * (auc(fpr, tpr_cut) - auc(fpr, tpr_previous))
    tpr_thresholds = np.array(tpr_thresholds)
    # roc auc normalization to be 1 for an ideal classifier
    area /= np.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * np.array(roc_weights))
    return area


# In[3]:


ls ../input/flavours-of-physics-kernels-only


# In[4]:


print("Load the train/test/eval data using pandas")
train = pd.read_csv("../input/flavours-of-physics-kernels-only/training.csv.zip")
test  = pd.read_csv("../input/flavours-of-physics-kernels-only/test.csv.zip")


# In[5]:


# -------------- loading data files -------------- #



#--------------- feature engineering -------------- #
def add_features(df):
    # features used by the others on Kaggle
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    #df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError'] # modified to:
    df['flight_dist_sig2'] = (df['FlightDistance']/df['FlightDistanceError'])**2
    # features from phunter
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['NEW_IP_dira'] = df['IP']*df['dira']
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)
    # My:
    # new combined features just to minimize their number;
    # their physical sense doesn't matter
    df['NEW_iso_abc'] = df['isolationa']*df['isolationb']*df['isolationc']
    df['NEW_iso_def'] = df['isolationd']*df['isolatione']*df['isolationf']
    df['NEW_pN_IP'] = df['p0_IP']+df['p1_IP']+df['p2_IP']
    df['NEW_pN_p']  = df['p0_p']+df['p1_p']+df['p2_p']
    df['NEW_IP_pNpN'] = df['IP_p0p2']*df['IP_p1p2']
    df['NEW_pN_IPSig'] = df['p0_IPSig']+df['p1_IPSig']+df['p2_IPSig']
    #My:
    # "super" feature changing the result from 0.988641 to 0.991099
    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']
    return df

print("Add features")
train = add_features(train)
test = add_features(test)


print("Eliminate features")
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',
              'SPDhits','CDF1', 'CDF2', 'CDF3',
              'isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt',
              'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',
              'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf',
              'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT',
              'p0_IP', 'p1_IP', 'p2_IP',
              'IP_p0p2', 'IP_p1p2',
              'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof',
              'p0_IPSig', 'p1_IPSig', 'p2_IPSig',
              'DOCAone', 'DOCAtwo', 'DOCAthree']

features = list(f for f in train.columns if f not in filter_out)


# In[6]:


mungedtrain = train[features].astype(complex)
mungedtest = test[features].astype(complex)


# In[7]:


def Output(p):
    return 1./(1.+np.exp(-p))
def GP(data):
    return (0.476446 +
            0.147600*np.tanh((((((3.141593)) * ((-((data["ISO_SumBDT"])))))) - (((((((data["IPSig"]) + (data["p1p2_ip_ratio"]))/2.0)) + (((data["IPSig"]) * (((data["p0p2_ip_ratio"]) * 2.0)))))/2.0)))) +
            0.998090*np.tanh(((np.cos((np.cos((data["IPSig"]))))) - ((((((data["IPSig"]) + (((data["VertexChi2"]) * (((data["p0p2_ip_ratio"]) + (data["NEW_iso_def"]))))))/2.0)) + (((data["ISO_SumBDT"]) * 2.0)))))) +
            0.984380*np.tanh(((np.cos((((data["p1p2_ip_ratio"]) + (data["p1p2_ip_ratio"]))))) - ((((((data["p0p2_ip_ratio"]) + ((((data["iso"]) + (data["FlightDistanceError"]))/2.0)))/2.0)) - (((np.cos((data["p_track_Chi2Dof_MAX"]))) * 2.0)))))) +
            1.0*np.tanh(np.cos(((((data["p_track_Chi2Dof_MAX"]) + ((((((data["FlightDistanceError"]) * (((np.sin((data["p0p2_ip_ratio"]))) * 2.0)))) + ((((data["VertexChi2"]) + (data["iso"]))/2.0)))/2.0)))/2.0)))) +
            0.938840*np.tanh(((np.tanh(((((data["IP"]) + (((data["NEW_pN_IP"]) - (np.cos((data["NEW_pN_IP"]))))))/2.0)))) - (((np.tanh((((data["NEW_IP_dira"]) + (data["IP"]))))) * 2.0)))) +
            0.824192*np.tanh(((((((((np.cos((data["p_track_Chi2Dof_MAX"]))) * (data["p_track_Chi2Dof_MAX"]))) * (data["p_track_Chi2Dof_MAX"]))) - (data["DCA_MAX"]))) - (((((np.cos((data["p_track_Chi2Dof_MAX"]))) * (data["FlightDistanceError"]))) * 2.0)))) +
            0.981420*np.tanh((-(((((((data["NEW_IP_dira"]) + (data["ISO_SumBDT"]))/2.0)) * (((np.sin((((((data["NEW_IP_dira"]) * ((31.006277)))) - (data["iso_bdt_min"]))))) - (data["DCA_MAX"])))))))) +
            0.779900*np.tanh(((((data["p0p2_ip_ratio"]) / 2.0)) * (((((np.tanh((((data["flight_dist_sig"]) - (((data["IPSig"]) * 2.0)))))) - (np.cos((data["p0p2_ip_ratio"]))))) - (np.cos((data["NEW_pN_IP"]))))))) +
            0.830724*np.tanh(((np.cos((((np.cos((data["p1p2_ip_ratio"]))) * 2.0)))) * (((np.cos((((data["p0p2_ip_ratio"]) * 2.0)))) + (np.cos((((((np.cos((data["p1p2_ip_ratio"]))) * 2.0)) * 2.0)))))))) +
            0.999648*np.tanh(((((data["LifeTime"]) * (((((((((((data["IP"]) * (((data["flight_dist_sig"]) * 2.0)))) * 2.0)) + ((7.07893085479736328)))) - (data["flight_dist_sig"]))) + (data["IP"]))))) * 2.0)) +
            0.824504*np.tanh(((((data["DCA_MAX"]) * ((-((data["DCA_MAX"])))))) * (((((data["DCA_MAX"]) * ((9.869604)))) * (((data["p1p2_ip_ratio"]) + (((data["NEW_iso_abc"]) * (data["DCA_MAX"]))))))))) +
            0.981804*np.tanh(np.tanh((np.tanh((((((np.sin((np.sin((data["p_track_Chi2Dof_MAX"]))))) + (np.sin(((-((((data["FlightDistanceError"]) * 2.0))))))))) * (np.cos(((-((data["p_track_Chi2Dof_MAX"])))))))))))) +
            1.0*np.tanh(((np.sin((data["FlightDistanceError"]))) * (((((((((data["NEW_IP_pNpN"]) * (data["IP"]))) * (data["FlightDistanceError"]))) + (np.sin((data["iso_min"]))))) * (data["FlightDistanceError"]))))) +
            0.999180*np.tanh(((data["LifeTime"]) * ((((((((-((data["iso"])))) * (((((data["VertexChi2"]) * 2.0)) * 2.0)))) - (data["FlightDistance"]))) + ((((((9.869604)) * 2.0)) * 2.0)))))) +
            0.897916*np.tanh(((data["NEW_IP_dira"]) * (((data["NEW_IP_dira"]) + (((np.cos((data["ISO_SumBDT"]))) + ((((((np.cos((data["iso"]))) + (((data["ISO_SumBDT"]) * 2.0)))/2.0)) * 2.0)))))))) +
            0.986864*np.tanh((-((np.sin((((((((np.cos(((((((data["VertexChi2"]) * (np.cos((np.cos((data["p1p2_ip_ratio"]))))))) + (data["p0p2_ip_ratio"]))/2.0)))) / 2.0)) / 2.0)) / 2.0))))))) +
            0.990504*np.tanh((-((((data["DCA_MAX"]) * (((np.cos((data["IPSig"]))) + (((data["IP"]) * (((data["VertexChi2"]) - (data["iso"])))))))))))) +
            0.987828*np.tanh((((((((((data["NEW_IP_pNpN"]) * (data["p1p2_ip_ratio"]))) * (data["p1p2_ip_ratio"]))) + (np.sin(((-(((((((0.636620)) / 2.0)) + (data["iso_bdt_min"])))))))))/2.0)) * (data["p_track_Chi2Dof_MAX"]))) +
            0.928492*np.tanh((((((data["iso_min"]) * (np.cos((((data["DCA_MAX"]) - (data["p_track_Chi2Dof_MAX"]))))))) + (((np.tanh((((data["DCA_MAX"]) - (((data["flight_dist_sig"]) * (data["LifeTime"]))))))) / 2.0)))/2.0)) +
            0.939444*np.tanh(((((((((data["NEW_IP_dira"]) * (data["ISO_SumBDT"]))) + (((np.sin((((np.sin((((data["iso_bdt_min"]) * ((-(((31.006277))))))))) * 2.0)))) / 2.0)))/2.0)) + (data["NEW_IP_pNpN"]))/2.0)) +
            1.0*np.tanh(((data["NEW_IP_pNpN"]) * (np.tanh((((((((((data["NEW_iso_abc"]) + ((-(((0.318310))))))/2.0)) + (((((np.sin(((-((data["iso"])))))) * 2.0)) * 2.0)))/2.0)) * 2.0)))))) +
            1.0*np.tanh(((((((data["NEW_FD_LT"]) * (((data["NEW_FD_SUMP"]) * (((data["NEW_FD_SUMP"]) * ((-((data["NEW_FD_LT"])))))))))) * (data["NEW_FD_SUMP"]))) * (np.cos((((data["ISO_SumBDT"]) / 2.0)))))) +
            0.907896*np.tanh((-((((data["IP"]) * (np.cos((((((data["iso_bdt_min"]) - ((((-((data["iso_bdt_min"])))) - (data["NEW_pN_IP"]))))) * 2.0))))))))) +
            0.685904*np.tanh(((((np.cos((data["p0p2_ip_ratio"]))) + (data["ISO_SumBDT"]))) * (((((-((((np.cos((data["ISO_SumBDT"]))) + (data["NEW_IP_dira"])))))) + (data["p1p2_ip_ratio"]))/2.0)))) +
            0.709252*np.tanh(((((data["p_track_Chi2Dof_MAX"]) * (((data["p_track_Chi2Dof_MAX"]) * (data["DCA_MAX"]))))) * ((-((((data["p1p2_ip_ratio"]) - (np.cos((np.sin((((((data["DCA_MAX"]) * 2.0)) * 2.0))))))))))))) +
            0.989888*np.tanh(((np.cos(((((((((np.cos((data["NEW_pN_IP"]))) * 2.0)) + (((np.cos((data["NEW_pN_IP"]))) * 2.0)))) + (data["IPSig"]))/2.0)))) * ((-(((((0.318310)) / 2.0))))))) +
            0.990684*np.tanh(((((data["iso"]) - (data["VertexChi2"]))) * (((data["VertexChi2"]) * (((data["iso"]) * (((np.tanh((np.tanh((data["iso"]))))) * (data["LifeTime"]))))))))) +
            0.967560*np.tanh(((np.sin((((data["p1p2_ip_ratio"]) + (((data["p1p2_ip_ratio"]) - ((((3.141593)) / 2.0)))))))) * (((data["p1p2_ip_ratio"]) * (((data["p1p2_ip_ratio"]) * (np.tanh((data["NEW_IP_pNpN"]))))))))) +
            0.999342*np.tanh((((((((((np.cos((((data["IPSig"]) * 2.0)))) / 2.0)) + (((((data["NEW_pN_p"]) * 2.0)) * (((data["IPSig"]) * (np.sin(((3.141593)))))))))/2.0)) / 2.0)) / 2.0)) +
            0.998932*np.tanh(((np.sin((((data["pt"]) * (np.tanh((data["NEW_FD_SUMP"]))))))) * (((data["pt"]) * (((data["DCA_MAX"]) * (data["NEW_FD_SUMP"]))))))) +
            0.999884*np.tanh((((((((((31.006277)) * (data["p0p2_ip_ratio"]))) - ((((31.006277)) / 2.0)))) - (((((data["flight_dist_sig"]) * (data["iso_min"]))) * (data["iso_min"]))))) * (data["LifeTime"]))) +
            0.929116*np.tanh(((((((data["DCA_MAX"]) * (((((((np.tanh((data["NEW_iso_def"]))) - (data["NEW_IP_pNpN"]))) - (data["NEW_IP_dira"]))) - (data["NEW_IP_dira"]))))) * 2.0)) + (((data["NEW_IP_pNpN"]) / 2.0)))) +
            0.958496*np.tanh(((data["NEW_IP_pNpN"]) * ((((np.sin((((((data["p0p2_ip_ratio"]) + (data["p1p2_ip_ratio"]))) - (data["IPSig"]))))) + ((((np.sin((data["NEW_FD_LT"]))) + (data["p1p2_ip_ratio"]))/2.0)))/2.0)))) +
            0.999890*np.tanh(((data["IP"]) * ((((data["ISO_SumBDT"]) + (np.sin((((((((data["ISO_SumBDT"]) + (data["NEW_pN_IP"]))) + (data["ISO_SumBDT"]))) + (np.cos((data["ISO_SumBDT"]))))))))/2.0)))) +
            0.972660*np.tanh(((((np.tanh(((((data["NEW_pN_IP"]) + (data["NEW_IP_pNpN"]))/2.0)))) / 2.0)) * ((((((data["NEW_IP_pNpN"]) + (np.sin((((np.sin((((data["NEW_pN_IP"]) * 2.0)))) * 2.0)))))/2.0)) / 2.0)))) +
            0.778268*np.tanh(((((((data["LifeTime"]) * 2.0)) * (((data["flight_dist_sig"]) / 2.0)))) * (np.sin((((((-(((0.636620))))) + (((data["LifeTime"]) - (data["IPSig"]))))/2.0)))))) +
            0.999978*np.tanh(np.sin(((-((((((((data["NEW5_lt"]) * (((((((data["p1p2_ip_ratio"]) * 2.0)) * 2.0)) + (data["IPSig"]))))) * (data["iso_bdt_min"]))) * ((9.869604))))))))) +
            0.999834*np.tanh(((data["LifeTime"]) * ((-((((((((((data["LifeTime"]) + (data["FlightDistance"]))) + (data["FlightDistance"]))) * (data["iso_bdt_min"]))) + (((data["iso"]) + (data["FlightDistance"])))))))))) +
            0.999632*np.tanh(np.cos(((((((((data["NEW_FD_SUMP"]) + ((9.869604)))/2.0)) + (np.tanh(((-((((((data["p_track_Chi2Dof_MAX"]) * 2.0)) - (((data["FlightDistanceError"]) - (data["p_track_Chi2Dof_MAX"])))))))))))) * 2.0)))) +
            0.999972*np.tanh(((data["p_track_Chi2Dof_MAX"]) * (((((data["FlightDistanceError"]) * (((data["iso_bdt_min"]) * (data["iso_bdt_min"]))))) * (np.sin((((np.sin((((data["IPSig"]) / 2.0)))) * (data["FlightDistanceError"]))))))))) +
            0.988372*np.tanh((((((np.cos((((data["ISO_SumBDT"]) * (((data["ISO_SumBDT"]) * (data["iso"]))))))) + (data["ISO_SumBDT"]))/2.0)) * (((data["ISO_SumBDT"]) * (np.tanh((data["iso"]))))))) +
            0.990676*np.tanh(((np.tanh((np.cos((((data["NEW_iso_def"]) - ((((-((((data["ISO_SumBDT"]) + (data["ISO_SumBDT"])))))) * (((data["ISO_SumBDT"]) - (data["NEW_iso_def"]))))))))))) * (data["NEW_IP_dira"]))) +
            0.985188*np.tanh((((-((((((((((np.sin((data["iso"]))) + (np.sin((((data["FlightDistanceError"]) * 2.0)))))) / 2.0)) / 2.0)) / 2.0))))) / 2.0)) +
            0.988216*np.tanh(((data["NEW_IP_pNpN"]) * ((-((((np.cos((data["FlightDistance"]))) * (np.cos(((((data["FlightDistance"]) + (((np.sin((np.tanh((((data["NEW_IP_pNpN"]) * 2.0)))))) * 2.0)))/2.0))))))))))) +
            0.945212*np.tanh(((((((((((data["LifeTime"]) * 2.0)) * 2.0)) * (((((-(((1.570796))))) + (((data["VertexChi2"]) * ((((-((data["iso_bdt_min"])))) * 2.0)))))/2.0)))) * 2.0)) * 2.0)) +
            0.972988*np.tanh(((data["IP"]) * (np.cos((((((np.cos((((((data["DCA_MAX"]) + (((data["IP"]) - (data["ISO_SumBDT"]))))) * 2.0)))) * 2.0)) * 2.0)))))) +
            0.761284*np.tanh(((np.cos((data["p_track_Chi2Dof_MAX"]))) * (np.sin((np.sin((((((((data["p1p2_ip_ratio"]) * (data["p1p2_ip_ratio"]))) - (np.sin((data["FlightDistanceError"]))))) / 2.0)))))))) +
            0.999672*np.tanh(((((data["NEW_iso_abc"]) * (np.sin((((data["dira"]) - (np.cos((((((data["NEW_IP_pNpN"]) * (((((data["iso_bdt_min"]) * (data["iso_bdt_min"]))) / 2.0)))) / 2.0)))))))))) * 2.0)) +
            1.0*np.tanh(((data["NEW_FD_SUMP"]) * ((((-(((((31.006277)) * 2.0))))) - ((((data["flight_dist_sig"]) + ((((((-((data["pt"])))) + (data["NEW_FD_SUMP"]))) * (data["DCA_MAX"]))))/2.0)))))) +
            1.0*np.tanh(((((((data["NEW_IP_pNpN"]) * (np.cos((data["NEW_iso_abc"]))))) * (np.tanh((((data["NEW_iso_abc"]) - (np.sin((data["NEW_pN_p"]))))))))) * (data["NEW_IP_pNpN"]))) +
            0.999718*np.tanh(((data["DCA_MAX"]) * ((-((np.sin(((((9.869604)) * (((data["NEW_FD_SUMP"]) * (((((((data["pt"]) + (np.sin((data["NEW_iso_def"]))))/2.0)) + (data["pt"]))/2.0))))))))))))) +
            0.999594*np.tanh((((((((((data["FlightDistance"]) * (data["NEW_IP_pNpN"]))) * (np.sin((data["NEW_iso_def"]))))) + (np.cos((data["FlightDistance"]))))/2.0)) * (((data["NEW_IP_pNpN"]) * (np.sin((data["NEW_IP_pNpN"]))))))) +
            0.993524*np.tanh(np.sin((((data["NEW_FD_SUMP"]) * (((data["NEW_iso_abc"]) - (((data["flight_dist_sig"]) * (np.cos(((((((data["IPSig"]) + (data["NEW_IP_pNpN"]))/2.0)) - (data["NEW_iso_def"]))))))))))))) +
            0.896484*np.tanh(((data["p1p2_ip_ratio"]) * (((np.tanh(((((data["FlightDistance"]) + (((np.cos((data["p1p2_ip_ratio"]))) * 2.0)))/2.0)))) - (np.tanh((data["FlightDistance"]))))))) +
            0.990364*np.tanh(((((data["NEW5_lt"]) * 2.0)) * (((((data["IPSig"]) - (((data["IPSig"]) * (np.cos((data["p1p2_ip_ratio"]))))))) * (((((np.sin((data["VertexChi2"]))) * 2.0)) * 2.0)))))) +
            0.980676*np.tanh((-((((((data["NEW5_lt"]) * (((data["VertexChi2"]) + (data["VertexChi2"]))))) * (((((((9.0)) * (data["ISO_SumBDT"]))) + (data["VertexChi2"]))/2.0))))))) +
            0.988828*np.tanh(((data["NEW_IP_pNpN"]) * ((((-((np.cos((((np.sin((((data["flight_dist_sig"]) - (((np.sin((data["NEW_IP_dira"]))) * 2.0)))))) - (data["NEW_pN_p"])))))))) / 2.0)))) +
            0.935788*np.tanh(((((data["DCA_MAX"]) * 2.0)) * (((np.cos((((((((data["p0p2_ip_ratio"]) - ((-((np.sin((((data["DCA_MAX"]) * 2.0))))))))) * 2.0)) * 2.0)))) / 2.0)))) +
            0.759384*np.tanh(((((((((((np.sin((data["p0p2_ip_ratio"]))) / 2.0)) * (data["FlightDistanceError"]))) / 2.0)) * (np.sin((data["p0p2_ip_ratio"]))))) * (np.sin((data["p0p2_ip_ratio"]))))) +
            0.980104*np.tanh(((data["iso_bdt_min"]) * ((((((np.tanh((((((((((data["IP"]) / 2.0)) - (data["DCA_MAX"]))) * (data["VertexChi2"]))) * (data["VertexChi2"]))))) / 2.0)) + (data["DCA_MAX"]))/2.0)))) +
            0.968704*np.tanh(((data["NEW_IP_pNpN"]) * ((((data["p1p2_ip_ratio"]) + ((((((((data["NEW_IP_pNpN"]) + (((data["NEW_IP_pNpN"]) * (np.sin((data["NEW_pN_IP"]))))))/2.0)) - (data["IP"]))) - (data["IP"]))))/2.0)))) +
            0.972400*np.tanh((-((((((data["NEW_FD_SUMP"]) * 2.0)) * (((((data["p_track_Chi2Dof_MAX"]) * 2.0)) + ((((-((((data["IPSig"]) * 2.0))))) + (data["FlightDistance"])))))))))) +
            0.856012*np.tanh(((np.sin(((((((11.26443672180175781)) * (((data["NEW5_lt"]) * (np.sin(((((((data["pt"]) + (np.tanh((data["VertexChi2"]))))/2.0)) + (data["VertexChi2"]))))))))) * 2.0)))) * 2.0)) +
            0.897284*np.tanh(np.sin((((((data["DCA_MAX"]) * (((data["IPSig"]) * (((data["DCA_MAX"]) * (((data["IPSig"]) * (data["NEW5_lt"]))))))))) * (data["IPSig"]))))) +
            0.993618*np.tanh(((((((data["IP"]) * (((((((data["DCA_MAX"]) - (data["iso_min"]))) * (data["p1p2_ip_ratio"]))) - (data["iso_min"]))))) - (data["IP"]))) * (((data["DCA_MAX"]) * 2.0)))) +
            0.984692*np.tanh((((((np.cos(((((10.0)) - (np.tanh((((((data["ISO_SumBDT"]) - (data["NEW_iso_def"]))) * (((data["flight_dist_sig"]) - ((10.0)))))))))))) + (data["NEW_IP_pNpN"]))/2.0)) / 2.0)) +
            0.905340*np.tanh(((data["DCA_MAX"]) * ((((np.cos((data["NEW_pN_IP"]))) + (((data["NEW5_lt"]) * (((((data["IPSig"]) - ((9.869604)))) * ((((8.0)) * ((8.0)))))))))/2.0)))) +
            0.910916*np.tanh(((((np.sin((((((((-(((((((31.006277)) * (data["NEW5_lt"]))) * 2.0))))) + (((data["NEW_iso_def"]) * ((0.318310)))))/2.0)) / 2.0)))) / 2.0)) / 2.0)) +
            0.993148*np.tanh(((((data["FlightDistance"]) + (np.cos((data["IPSig"]))))) * (((data["p1p2_ip_ratio"]) * (np.sin((((data["iso"]) * (((data["p0p2_ip_ratio"]) * ((-((data["LifeTime"])))))))))))))) +
            0.842268*np.tanh(np.tanh((((np.tanh((((((np.cos((data["p0p2_ip_ratio"]))) / 2.0)) * ((((0.318310)) * (((data["p1p2_ip_ratio"]) + ((-((np.cos((data["p0p2_ip_ratio"])))))))))))))) / 2.0)))) +
            1.0*np.tanh(((data["DCA_MAX"]) * (np.sin(((((data["iso_bdt_min"]) + (((((data["NEW_FD_SUMP"]) * 2.0)) * (((data["pt"]) * (((data["NEW_FD_SUMP"]) * (data["pt"]))))))))/2.0)))))) +
            0.825612*np.tanh((((((data["DCA_MAX"]) + (data["IP"]))/2.0)) * ((((np.sin(((((9.869604)) * ((-((data["p_track_Chi2Dof_MAX"])))))))) + (((data["IP"]) * ((-((data["p_track_Chi2Dof_MAX"])))))))/2.0)))) +
            0.993998*np.tanh(((((((data["p0p2_ip_ratio"]) * (np.cos((((np.tanh((np.cos((data["NEW5_lt"]))))) * 2.0)))))) / 2.0)) - (((((((data["NEW5_lt"]) * 2.0)) * (data["p_track_Chi2Dof_MAX"]))) * 2.0)))) +
            0.938832*np.tanh(((((((data["pt"]) * (data["NEW_FD_SUMP"]))) - (data["NEW_IP_dira"]))) * (((data["DCA_MAX"]) * (((data["p0p2_ip_ratio"]) - (((data["p1p2_ip_ratio"]) * (data["p0p2_ip_ratio"]))))))))) +
            0.891984*np.tanh((((-((((data["VertexChi2"]) * ((((data["dira"]) + (np.tanh(((((31.006277)) - (data["pt"]))))))/2.0))))))) * (((((data["dira"]) - ((31.006277)))) * 2.0)))) +
            0.900620*np.tanh(((data["iso"]) * (np.sin((np.tanh((((((((data["NEW5_lt"]) * (data["iso"]))) * (np.cos((((data["flight_dist_sig2"]) * (data["iso_bdt_min"]))))))) * (data["iso"]))))))))) +
            0.999714*np.tanh(((((data["FlightDistance"]) + (((((data["ISO_SumBDT"]) * (data["VertexChi2"]))) + (np.tanh((data["pt"]))))))) * (((data["pt"]) * (np.sin(((3.141593)))))))) +
            0.510884*np.tanh(((((np.tanh((np.tanh((np.tanh((data["NEW_IP_pNpN"]))))))) + ((((-((data["NEW_IP_pNpN"])))) * ((-((np.sin((((data["iso"]) * 2.0))))))))))) / 2.0)) +
            0.844724*np.tanh(((data["p1p2_ip_ratio"]) * (((((data["LifeTime"]) * (((((np.cos((data["p1p2_ip_ratio"]))) * (data["IPSig"]))) + ((-((((data["iso_bdt_min"]) * (data["flight_dist_sig"])))))))))) * 2.0)))) +
            0.999822*np.tanh((((((((31.006277)) + (((data["NEW_pN_IPSig"]) * (np.cos((((np.cos((data["iso_bdt_min"]))) * ((31.006277)))))))))) + (data["NEW_pN_IPSig"]))) * ((-((data["NEW_FD_SUMP"])))))) +
            0.738600*np.tanh(((data["IP"]) * (np.sin((np.sin((((data["DCA_MAX"]) * ((((data["NEW_iso_abc"]) + (((np.sin((data["IP"]))) * ((((data["NEW_iso_abc"]) + (data["NEW5_lt"]))/2.0)))))/2.0)))))))))) +
            0.787448*np.tanh(((data["DCA_MAX"]) * (((np.tanh((((data["DCA_MAX"]) * (np.cos((data["VertexChi2"]))))))) - (((data["IP"]) - (((data["IP"]) * (np.cos((data["NEW_pN_p"]))))))))))) +
            0.999976*np.tanh(((((((((data["LifeTime"]) * 2.0)) * 2.0)) * (((((data["NEW_iso_def"]) * (np.cos((((data["p1p2_ip_ratio"]) * 2.0)))))) - (np.cos((((data["p1p2_ip_ratio"]) * 2.0)))))))) * 2.0)) +
            0.999974*np.tanh(((((data["flight_dist_sig"]) * (data["NEW_FD_SUMP"]))) * ((((-((np.sin((data["IPSig"])))))) + (np.tanh((((data["flight_dist_sig"]) * (((data["IPSig"]) - ((3.141593)))))))))))) +
            0.986320*np.tanh(((((np.cos((data["FlightDistanceError"]))) - (data["iso_bdt_min"]))) * (((np.sin((((data["IP"]) * ((31.006277)))))) * (((data["IP"]) * (data["iso_bdt_min"]))))))) +
            0.981344*np.tanh((((-((((data["NEW5_lt"]) * (((data["p1p2_ip_ratio"]) * ((((((((((data["NEW_pN_IP"]) * 2.0)) + (data["p1p2_ip_ratio"]))/2.0)) + (data["p0p2_ip_ratio"]))) + (data["p0p2_ip_ratio"])))))))))) * 2.0)) +
            0.991896*np.tanh(((data["DCA_MAX"]) * (np.sin((((np.cos((((np.cos((data["DCA_MAX"]))) * (data["p0p2_ip_ratio"]))))) - (((data["FlightDistance"]) * (np.sin((((data["p0p2_ip_ratio"]) / 2.0)))))))))))) +
            0.928168*np.tanh(((((((data["FlightDistanceError"]) * (data["FlightDistance"]))) * (data["NEW5_lt"]))) * (np.sin((((data["FlightDistance"]) * (((((data["NEW5_lt"]) * (data["FlightDistanceError"]))) * 2.0)))))))) +
            0.999790*np.tanh(((data["iso_bdt_min"]) * ((-(((((np.sin((((data["DCA_MAX"]) - (data["IP"]))))) + (np.tanh((((data["NEW_iso_def"]) + (((data["DCA_MAX"]) - (data["iso_min"]))))))))/2.0))))))) +
            0.836476*np.tanh(((((((data["NEW_FD_SUMP"]) * (np.sin((((((((data["FlightDistance"]) - (data["iso_min"]))) - (np.cos((data["FlightDistance"]))))) / 2.0)))))) * (data["NEW_iso_abc"]))) * 2.0)) +
            0.966008*np.tanh(((data["iso_min"]) * (((((np.cos((data["p0p2_ip_ratio"]))) * 2.0)) * (((((((data["ISO_SumBDT"]) * (data["NEW_pN_IPSig"]))) - (data["LifeTime"]))) * (data["LifeTime"]))))))) +
            0.999268*np.tanh(((((data["iso"]) - (data["p_track_Chi2Dof_MAX"]))) * (np.sin((((data["DCA_MAX"]) * ((((data["DCA_MAX"]) + (((data["NEW_IP_pNpN"]) * 2.0)))/2.0)))))))) +
            0.999957*np.tanh(((data["LifeTime"]) * (((((data["p_track_Chi2Dof_MAX"]) * (((((data["iso"]) - (((data["p_track_Chi2Dof_MAX"]) * (((data["p_track_Chi2Dof_MAX"]) * (data["iso"]))))))) * 2.0)))) * (data["p_track_Chi2Dof_MAX"]))))) +
            0.960156*np.tanh(((((((data["NEW_FD_SUMP"]) * (np.sin((np.tanh((((data["DCA_MAX"]) * (np.tanh((((data["p1p2_ip_ratio"]) * 2.0)))))))))))) * (((data["pt"]) + (data["NEW_IP_pNpN"]))))) / 2.0)) +
            0.999967*np.tanh((((-((data["IP"])))) * (((((((np.sin((((data["NEW_iso_def"]) * 2.0)))) + (((data["p0p2_ip_ratio"]) * (data["DCA_MAX"]))))/2.0)) + (((data["NEW_iso_def"]) * (data["DCA_MAX"]))))/2.0)))) +
            0.912300*np.tanh((((((9.869604)) - (((data["DCA_MAX"]) * (data["NEW_iso_def"]))))) * (((((data["NEW_iso_def"]) - (((data["DCA_MAX"]) * ((9.869604)))))) * (data["NEW5_lt"]))))) +
            0.919472*np.tanh(((((data["DCA_MAX"]) * (np.cos((((((data["flight_dist_sig"]) + ((((((data["flight_dist_sig"]) - (np.sin((data["flight_dist_sig"]))))) + (data["p0p2_ip_ratio"]))/2.0)))) * 2.0)))))) / 2.0)) +
            0.983644*np.tanh((((((data["IPSig"]) * (((((((((3.141593)) * (((np.sin(((3.141593)))) * (data["NEW_FD_LT"]))))) + (data["NEW5_lt"]))/2.0)) * 2.0)))) + (data["NEW5_lt"]))/2.0)) +
            0.889412*np.tanh(((((((np.sin((((data["p0p2_ip_ratio"]) + ((((np.sin((data["flight_dist_sig"]))) + (data["ISO_SumBDT"]))/2.0)))))) / 2.0)) * (((np.sin((data["p0p2_ip_ratio"]))) / 2.0)))) / 2.0)) +
            0.894832*np.tanh(((data["NEW_IP_dira"]) * (np.tanh((((data["NEW_iso_abc"]) * (np.tanh((np.tanh((np.tanh((np.tanh((np.cos((((data["ISO_SumBDT"]) - (np.tanh((data["NEW_iso_abc"]))))))))))))))))))))))

def GPReal(data):
     return (0.659360*np.tanh(np.real(((np.sin((np.sin(((((((data["p_track_Chi2Dof_MAX"]) * (data["p_track_Chi2Dof_MAX"]))) + (data["IPSig"]))/2.0)))))) - ((((((data["p_track_Chi2Dof_MAX"]) * (np.power((data["iso"]),2.)))) + ((((((data["IPSig"]) + (data["VertexChi2"]))/2.0)) - (((data["flight_dist_sig"]) / (data["IPSig"]))))))/2.0))))) +
             0.800000*np.tanh(np.real(((np.cos((((np.power((((((data["LifeTime"]) / (((np.tanh((complex(1.86195778846740723)))) * (np.conjugate(data["NEW_FD_SUMP"]))*(complex(0,1)))))) - (((((data["dira"]) + (data["LifeTime"]))) + (data["LifeTime"]))))),2.)) / 2.0)))) - (((data["VertexChi2"]) + (np.power((((np.tanh((data["LifeTime"]))) / (data["NEW_FD_SUMP"]))),3.))))))))
def GPComplex(data):
    return (0.659360*np.tanh(np.imag(((np.sin((np.sin(((((((data["p_track_Chi2Dof_MAX"]) * (data["p_track_Chi2Dof_MAX"]))) + (data["IPSig"]))/2.0)))))) - ((((((data["p_track_Chi2Dof_MAX"]) * (np.power((data["iso"]),2.)))) + ((((((data["IPSig"]) + (data["VertexChi2"]))/2.0)) - (((data["flight_dist_sig"]) / (data["IPSig"]))))))/2.0))))) +
             0.800000*np.tanh(np.imag(((np.cos((((np.power((((((data["LifeTime"]) / (((np.tanh((complex(1.86195778846740723)))) * (np.conjugate(data["NEW_FD_SUMP"]))*(complex(0,1)))))) - (((((data["dira"]) + (data["LifeTime"]))) + (data["LifeTime"]))))),2.)) / 2.0)))) - (((data["VertexChi2"]) + (np.power((((np.tanh((data["LifeTime"]))) / (data["NEW_FD_SUMP"]))),3.))))))))


# In[8]:


plt.figure(figsize=(15,15))
colors=['r','g']
plt.scatter(GPReal(mungedtrain),GPComplex(mungedtrain),s=1,color=[colors[int(c)] for c in train.signal.values])
plt.show()


# In[9]:


plt.figure(figsize=(15,15))
plt.scatter(GPReal(mungedtest[::12]),GPComplex(mungedtest[::12]),s=1)


# In[10]:


check_agreement = pd.read_csv('../input/flavours-of-physics-kernels-only/check_agreement.csv.zip')
check_correlation = pd.read_csv('../input/flavours-of-physics-kernels-only/check_correlation.csv.zip')
check_agreement.head()
check_agreement = add_features(check_agreement)
check_agreement = check_agreement[features+['weight','signal']]
check_agreement.head()
check_correlation = add_features(check_correlation)
check_correlation = check_correlation[features+['mass']]
check_agreement[check_agreement.columns[:-2]] = check_agreement[check_agreement.columns[:-2]].astype(complex)
agreement_probs = Output(.3*GPReal(check_agreement)+.7*GP(check_agreement))
print('Checking agreement...')
ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)
check_correlation[check_correlation.columns[:-1]] = check_correlation[check_correlation.columns[:-1]].astype(complex)
correlation_probs = Output(.3*GPReal(check_correlation)+.7*GP(check_correlation))
print ('Checking correlation...')
cvm = compute_cvm(correlation_probs, check_correlation['mass'])
print ('CvM metric', cvm, cvm < 0.002)

train_eval_probs = Output(.3*GPReal(mungedtrain)+.7*GP(mungedtrain))

print ('Calculating AUC...')
AUC = roc_auc_truncated(train['signal'], train_eval_probs)
print ('AUC', AUC)


# In[11]:


mungedtest.head()


# In[12]:


sub = pd.DataFrame({'id':test.id.values,'prediction':Output(.3*GPReal(mungedtest)+.7*GP(mungedtest.astype(float)))})
sub.head()


# In[13]:


sub.to_csv('submission.csv',index=False)

