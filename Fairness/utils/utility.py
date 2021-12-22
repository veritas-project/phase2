'''
Helper functions for life insurance underwriting

Copyright © 2021 Monetary Authority of Singapore

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
'''

from collections import namedtuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, classification_report
def evaluate(y_true, y_probs, threshold=0.5, verbose=True):
    """
    Inputs:
        y_true::np.array: typically y_true in practice, array of labels of ground truth
        y_probs::np.array: results from .predict_proba in practice, array of probabilistic score estimates of majority class
        threshold::float: threshold where probailistic scores above the threshold will be classified as 1
    
    Returns:
        Accuracy Score
        F1 Score
        ROC AUC Score
        PR AUC Score (Average Precision Score) - See https://sinyi-chou.github.io/python-sklearn-precision-recall/
        
    """
    y_score = y_probs.copy()
    y_pred = np.where(y_score > threshold, 1, 0)
    acc = accuracy_score(y_true, y_pred)
    if verbose:
        print(f"Accuracy score: {acc:.5f}")
    
    f1 = f1_score(y_true, y_pred)
    if verbose:
        print(f"F1-weighted: {f1:.5f}")
    
    roc_auc = roc_auc_score(y_true, y_score)
    if verbose:
        print(f"ROC AUC Score: {roc_auc:.5f}")
    
    pr_auc = average_precision_score(y_true, y_pred)
    if verbose:
        print(f"PR AUC Score: {pr_auc:.5f}")
    
    if verbose:
        print(f"Classification Report:\n{classification_report(y_true, y_pred)}")
        
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, roc_auc, pr_auc, cm
def bootstrap_conf_int(y_true, y_model, score_func, k=50):
    results = np.zeros(k)
    for i in range(k):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        results[i] = score_func(y_true[idx], y_model[idx])

    return results.mean(), 2 * results.std()


def format_uncertainty(mean_val, conf_int):
    return f"{mean_val:.3f} +/- {conf_int:.3f}"


class ModelRates:
    def __init__(self, y_true, y_prob):
        (ths, tpr, fpr, ppv, forr, base_ar, ar) = self._compute_rates(y_true, y_prob)
        self.tpr = interp1d(ths, tpr)
        self.fpr = interp1d(ths, fpr)
        self.ppv = interp1d(ths, ppv)
        self.forr = interp1d(ths, forr)
        self.approv_rate = interp1d(ths, ar)
        self.base_ar = base_ar

    @staticmethod
    def _compute_rates(y_true, y_prob):
        # Vectorizable computation of rates
        fpr, tpr, ths = roc_curve(y_true, y_prob, pos_label=1)
        ths[0] = 1.0  # roc_curve sets max threshold arbitrarily above 1
        ths = np.append(ths, [0.0])  # Add endpoints for ease of interpolation
        fpr = np.append(fpr, [1.0])
        tpr = np.append(tpr, [1.0])
        
        base_approv_rate = np.mean(y_true)
        base_reject_rate = 1 - base_approv_rate
        
        approv_rate = base_approv_rate * tpr + base_reject_rate * fpr
        reject_rate = 1 - approv_rate
        
        prob_tp = base_approv_rate * tpr
        ppv = np.divide(prob_tp, approv_rate, out=np.zeros_like(prob_tp), where=(approv_rate != 0))
        
        
        prob_fn0 = prob_tp * np.divide(1, tpr, out=np.zeros_like(prob_tp), where=(tpr != 0)) - prob_tp
        prob_fn = np.where(tpr == 0, approv_rate, prob_fn0)
        forr = np.divide(prob_fn, reject_rate, out=np.zeros_like(prob_fn), where=(reject_rate != 0))
        
        return ths, tpr, fpr, ppv, forr, base_approv_rate, approv_rate


class FairnessAnalysis:
    Metrics = namedtuple('Metrics', 'equal_opp fnr_parity fnr_ratio, fpr_parity fpr_ratio avg_odds ppv_parity fdr_ratio forr_ratio dem_parity dis_impact preval_ratio acc w_acc bal_acc f1 auc')

    metric_names = {'equal_opp': 'Equal Opportunity',
                    'fnr_parity': 'False Negative Rate Parity',
                    'fnr_ratio': 'False Negative Rate Ratio',
                    'fpr_parity': 'False Positive Rate Parity',
                    'fpr_ratio':'False Positive Ratio',
                    'avg_odds': 'Average Odds',
                    'ppv_parity': 'Positive Predictive Parity',
                    'fdr_ratio': 'False Discovery Rate Ratio',
                    'forr_ratio': 'False Omission Rate Ratio',
                    'dem_parity': 'Demographic Parity', 
                    'dis_impact': 'Disparate Impact',
                    'preval_ratio':'Prevalence Ratio',
                    'acc':'Accuracy',
                    'w_acc':'Weighted Accuracy',
                    'bal_acc': 'Balanced Accuracy',
                    'f1':'F1 Score',
                    'auc':'Area Under Curve'}
    
                                 
    def __init__(self, y_true, y_prob, group_mask):
        self.rates_a = ModelRates(y_true[group_mask], y_prob[group_mask])
        self.rates_b = ModelRates(y_true[~group_mask], y_prob[~group_mask])
        self.y_true = y_true
        self.y_prob = y_prob
        self.group_mask = group_mask
        assert np.sum(y_true == 1) + np.sum(y_true == 0) == len(y_true)  # Confirm 1, 0 labelling
        
    def compute(self, th_a=0.5, th_b=None):
        # Vectorizable
        if th_b is None:
            th_b = th_a

        # Fairness
        tpr_a, tpr_b = self.rates_a.tpr(th_a), self.rates_b.tpr(th_b)
        fpr_a, fpr_b = self.rates_a.fpr(th_a), self.rates_b.fpr(th_b)
        ppv_a, ppv_b = self.rates_a.ppv(th_a), self.rates_b.ppv(th_b)
        
        equal_opp = tpr_a - tpr_b
        fpr_parity = fpr_a - fpr_b
        
        avg_odds = 0.5 * (equal_opp + fpr_parity)
        dem_parity = self.rates_a.approv_rate(th_a) - self.rates_b.approv_rate(th_b)
        ppv_parity = ppv_a - ppv_b
        
        #fdr_parity = -ppv_parity
        fdr_ratio = (1-ppv_a)/(1-ppv_b)
        fnr_parity = -equal_opp
        #forr_parity = self.rates_a.forr(th_a) - self.rates_b.forr(th_b)
        forr_ratio = self.rates_a.forr(th_a)/self.rates_b.forr(th_b)
        dis_impact = self.rates_a.approv_rate(th_a)/self.rates_b.approv_rate(th_b)
        preval_ratio = self.rates_a.base_ar/self.rates_b.base_ar
        
        fpr_ratio = fpr_a / fpr_b
        fnr_ratio = (1 - tpr_a) / (1 - tpr_b)
             
        # Performance
        # Combine TPRs: P(R=1|Y=1) = P(R=1|Y=1,A=1)P(A=1|Y=1) + P(R=1|Y=1,A=0)P(A=0|Y=1)
        tpr = (tpr_a * np.mean(self.group_mask[self.y_true == 1]) +
               tpr_b * np.mean(~self.group_mask[self.y_true == 1]))
        # Combine FPRs: P(R=1|Y=0) = P(R=1|Y=0,A=1)P(A=1|Y=0) + P(R=1|Y=0,A=0)P(A=0|Y=0)
        fpr = (fpr_a * np.mean(self.group_mask[self.y_true == 0]) +
               fpr_b * np.mean(~self.group_mask[self.y_true == 0]))
        
        bal_acc = 0.5 * (tpr + 1 - fpr)
                  
        if isinstance(th_a, np.ndarray):
            # ni, nj = th_a.shape[1], th_b.shape[0]
            acc = w_acc = f1 = auc = None
        else:    
            n = len(self.y_true)
            n_a = np.sum(self.y_true)
            n_b = n - n_a
            
            pred = np.zeros_like(self.y_true)
            pred[self.group_mask] = (self.y_prob[self.group_mask] >= th_a).astype(int)
            pred[~self.group_mask] = (self.y_prob[~self.group_mask] >= th_b).astype(int)
            
            acc = accuracy_score(self.y_true, pred)
            acc_a = accuracy_score(self.y_true[self.y_true == 1], pred[self.y_true == 1])
            acc_b = accuracy_score(self.y_true[self.y_true == 0], pred[self.y_true == 0])
            
            w_acc = (acc_a * n_a + acc_b * n_b)/n
            
            f1 = f1_score(self.y_true, pred)
            auc = roc_auc_score(self.y_true, self.y_prob)
            
        return self.Metrics(equal_opp, fnr_parity, fnr_ratio, fpr_parity, fpr_ratio, avg_odds, ppv_parity, fdr_ratio, forr_ratio, dem_parity, dis_impact, preval_ratio, acc, w_acc, bal_acc, f1, auc)
    
    
    def compute_performance_rates(self, th_a=0.5, th_b=None):
        # Vectorizable
        if th_b is None:
            th_b = th_a

        # Performance metrics per group
        tpr_a, tpr_b = self.rates_a.tpr(th_a), self.rates_b.tpr(th_b)
        fpr_a, fpr_b = self.rates_a.fpr(th_a), self.rates_b.fpr(th_b)
        ppv_a, ppv_b = self.rates_a.ppv(th_a), self.rates_b.ppv(th_b)
        forr_a,forr_b  = self.rates_a.forr(th_a), self.rates_a.forr(th_b)
        fnr_a = (1 - tpr_a)
        fnr_b = (1 - tpr_b)
        bal_acc_a = 0.5 * (tpr_a + 1 - fpr_a)
        bal_acc_b = 0.5 * (tpr_b + 1 - fpr_b)
        perf_metrics_a = [tpr_a,fnr_a,fpr_a,ppv_a,bal_acc_a, forr_a]
        perf_metrics_b = [tpr_b,fnr_b,fpr_b,ppv_b,bal_acc_b, forr_b]
        perf_metrics = pd.DataFrame(list(zip(perf_metrics_a,perf_metrics_b)), columns=['Group_a','Group_b'],index=['tpr','fnr','fpr','ppv','bal_acc','forr'])
        return perf_metrics

class ModelPredictions:
    def __init__(self, y_prob, threshold_a, threshold_b, threshold_mask):
        self.y_pred = self._group_thresholds(y_prob, threshold_a, threshold_b, threshold_mask)

    @staticmethod
    def _group_thresholds(y_prob, threshold_a, threshold_b, threshold_mask):
        y_pred = np.zeros_like(y_prob)
        y_pred[threshold_mask] = (y_prob[threshold_mask] >= threshold_a).astype(int)
        y_pred[~threshold_mask] = (y_prob[~threshold_mask] >= threshold_b).astype(int)
        return y_pred

    
class FairnessAnalysisSecondary:
    Metrics = namedtuple('Metrics', 'equal_opp fnr_parity fnr_ratio, fpr_parity fpr_ratio avg_odds ppv_parity fdr_ratio forr_ratio dem_parity dis_impact preval_ratio acc w_acc bal_acc f1 auc')

    metric_names = {'equal_opp': 'Equal Opportunity',
                    'fnr_parity': 'False Negative Rate Parity',
                    'fnr_ratio': 'False Negative Rate Ratio',
                    'fpr_parity': 'False Positive Rate Parity',
                    'fpr_ratio':'False Positive Ratio',
                    'avg_odds': 'Average Odds',
                    'ppv_parity': 'Positive Predictive Parity',
                    'fdr_ratio': 'False Discovery Rate Ratio',
                    'forr_ratio': 'False Omission Rate Ratio',
                    'dem_parity': 'Demographic Parity', 
                    'dis_impact': 'Disparate Impact',
                    'preval_ratio':'Prevalence Ratio',
                    'acc':'Accuracy',
                    'w_acc':'Weighted Accuracy',
                    'bal_acc': 'Balanced Accuracy',
                    'f1':'F1 Score',
                    'auc':'Area Under Curve'}
    
    
    
                                 
    def __init__(self, y_true, y_prob, threshold_mask,threshold_a, threshold_b):
        y_predictions = ModelPredictions(y_prob, threshold_a, threshold_b, threshold_mask)
        self.y_pred = y_predictions.y_pred
        self.y_prob = y_prob
        self.y_true = y_true
        self.threshold_mask = threshold_mask
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        assert np.sum(y_true == 1) + np.sum(y_true == 0) == len(y_true)  # Confirm 1, 0 labelling
        
    @staticmethod
    def _compute_rates_secondary(y_true, y_pred):
        # Computation of rates based on predictions
        tp, fn, fp, tn = confusion_matrix(y_true,y_pred,labels=[1,0]).reshape(-1)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        base_approv_rate = np.mean(y_true)
        base_reject_rate = 1 - base_approv_rate
        
        approv_rate = base_approv_rate * tpr + base_reject_rate * fpr
        reject_rate = 1 - approv_rate
        
        prob_tp = base_approv_rate * tpr
        ppv = np.divide(prob_tp, approv_rate, out=np.zeros_like(prob_tp), where=(approv_rate != 0))
        
        
        prob_fn0 = prob_tp * np.divide(1, tpr, out=np.zeros_like(prob_tp), where=(tpr != 0))
        prob_fn = np.where(tpr == 0, approv_rate, prob_fn0)
        forr = np.divide(prob_fn, reject_rate, out=np.zeros_like(prob_fn), where=(reject_rate != 0))

        
        return tpr, fpr, ppv, forr, base_approv_rate, approv_rate
        
    def compute_secondary(self, fairness_mask):
        # Vectorizable

        # Fairness
        tpr_a, fpr_a, ppv_a, forr_a, base_approv_rate_a, approv_rate_a = self._compute_rates_secondary(self.y_true[fairness_mask], self.y_pred[fairness_mask])
        tpr_b, fpr_b, ppv_b, forr_b, base_approv_rate_b, approv_rate_b = self._compute_rates_secondary(self.y_true[~fairness_mask], self.y_pred[~fairness_mask])

        
        equal_opp = tpr_a - tpr_b
        fpr_parity = fpr_a - fpr_b
        
        avg_odds = 0.5 * (equal_opp + fpr_parity)
        dem_parity = approv_rate_a - approv_rate_b
        ppv_parity = ppv_a - ppv_b
        
        #fdr_parity = -ppv_parity
        fdr_ratio = (1-ppv_a)/(1-ppv_b)
        fnr_parity = -equal_opp
        forr_ratio = forr_a / forr_b
        
        dis_impact = approv_rate_a/approv_rate_b
        preval_ratio = base_approv_rate_a/base_approv_rate_b
        
        fpr_ratio = fpr_a / fpr_b
        fnr_ratio = (1 - tpr_a) / (1 - tpr_b)
             
        # Performance
        # Combine TPRs: P(R=1|Y=1) = P(R=1|Y=1,A=1)P(A=1|Y=1) + P(R=1|Y=1,A=0)P(A=0|Y=1)
        tpr = (tpr_a * np.mean(fairness_mask[self.y_true == 1]) +
               tpr_b * np.mean(~fairness_mask[self.y_true == 1]))
        # Combine FPRs: P(R=1|Y=0) = P(R=1|Y=0,A=1)P(A=1|Y=0) + P(R=1|Y=0,A=0)P(A=0|Y=0)
        fpr = (fpr_a * np.mean(fairness_mask[self.y_true == 0]) +
               fpr_b * np.mean(~fairness_mask[self.y_true == 0]))
        
        bal_acc = 0.5 * (tpr + 1 - fpr)
                  
#         if isinstance(th_a, np.ndarray):
#             # ni, nj = th_a.shape[1], th_b.shape[0]
#             acc = w_acc = f1 = auc = None
        #else:    
        n = len(self.y_true)
        n_a = np.sum(self.y_true)
        n_b = n - n_a

#             pred = np.zeros_like(self.y_true)
#             pred[self.group_mask] = (self.y_prob[self.group_mask] >= th_a).astype(int)
#             pred[~self.group_mask] = (self.y_prob[~self.group_mask] >= th_b).astype(int)

        acc = accuracy_score(self.y_true, self.y_pred)
        acc_a = accuracy_score(self.y_true[self.y_true == 1], self.y_pred[self.y_true == 1])
        acc_b = accuracy_score(self.y_true[self.y_true == 0], self.y_pred[self.y_true == 0])

        w_acc = (acc_a * n_a + acc_b * n_b)/n

        f1 = f1_score(self.y_true, self.y_pred)
        auc = roc_auc_score(self.y_true, self.y_prob)
            
        return self.Metrics(equal_opp, fnr_parity, fnr_ratio, fpr_parity, fpr_ratio, avg_odds, ppv_parity, fdr_ratio, forr_ratio, dem_parity, dis_impact, preval_ratio, acc, w_acc, bal_acc, f1, auc)
    
    