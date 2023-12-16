# coding: utf-8

from cmath import nan
# from tkinter.messagebox import NO
import pandas as pd
import numpy as np
from argparse import ArgumentParser


class Agreement:
    """
    class Agreement
        Supports computing Fleiss' kappa for multiple annotators,
        where we might have different annotators assigned to different
        subsets of the data.
    Method
        For a set of possible labels, we construct a count matrix of
        size A of MxN, where M is the number of items and N is the total number
        of annotators. the i,jth index of A indicates how many annotators
        assigned the label of column j to the item of index i.
    """

    def __init__(self, disagg_df, cols, text_id_col="text_id"):
        """ Takes a long dataframe (one annotation, one tweet per row)

        Params
            disagg_df: pd.DataFrame, contains columns:
                text_id (int) unique id of tweets/observations
                binary (0,1) label columns

        """
        self.df = disagg_df
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        ids = list(set(self.df[f"{text_id_col}"].values.tolist()))
        self.map = {idx: i for i, idx in enumerate(ids)}
        self.cols = cols
        self.N = len(ids)
        self.K = 2
        self.text_id_col = text_id_col

    def _fill(self, col):
        self.matrix = np.zeros( (self.N, self.K) )
        for _, row in self.df.iterrows():
            index = self.map[row[f"{self.text_id_col}"]]
            val = row[col]
            if val not in (0,1):
                raise ValueError("Bad value: {}".format(val))
            # val 0 or 1
            self.matrix[index, val] += 1

    def _fleiss(self):
        m = self.matrix

        N = np.sum(m)
        p = np.sum(m, axis=0) / N
        n = np.sum(m, axis=1)
        P = np.array([ (1/(n[i] * (n[i] - 1))) * np.sum([m[i,j]*(m[i,j] - 1) for j in range(m.shape[1])]) for i in range(m.shape[0])])
        P_bar = np.mean(P)
        P_bar_exp = np.sum([p[i] ** 2 for i in range(len(p))])
        kappa = (P_bar - P_bar_exp) / (1 - P_bar_exp)
        return kappa, P_bar, P_bar_exp

    def _pabak(self, observed):
        return (self.K / (self.K - 1)) * (observed - (1 / self.K))

    def kappas(self):
        agreements = dict()
        for col in self.cols:
            self._fill(col)
            fleiss, obs, exp = self._fleiss()
            pabak = self._pabak(obs)
            agreements[col] = {"Fleiss": fleiss,
                               "PABAK": pabak}
        return pd.DataFrame(agreements)

if __name__ == '__main__':

    fake_dataset = pd.DataFrame({
        'text_id':[2,1,1,2,2,2,3,3,3],
        'col_A': [0,0,1,0,1,1,1,1,1],
        'col_B': [1,1,0,0,1,0,1,0,0],
        'annotator_id': ['an1','an2','an3']*3
    })
    print(fake_dataset)
    agr_test = Agreement(fake_dataset, ['col_A', 'col_B']).kappas()
    print(agr_test.round(2))
