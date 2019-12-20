'''
A simple way to create marov chains from data frames
'''
import pandas as pd
import numpy as np
import logging
import networkx as nx
import pydot
import os

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# TODO : sessie optioneel maken

# TODO : setup.py opzetten,  docstrings,  pylint


class Markov():
    '''some docstring'''
    def __init__(self, df, state_col="state", date_col="date", user_col="user",
                 session_col="session", sort=True):
        self.number_of_states = df[state_col].nunique()

        self.transition_matrices = {}

        self.states = df[state_col].unique()

        if sort:      
            df = df.sort_values(by=[date_col])

        else:
            LOGGER.info('You specified that you do *not* want to let df2markov sort the '
                        'date column. We hope that you know what you are doing and made '
                        'sure that the column is in chronological order. '
                        'Otherwise, the results will be wrong.')
                
        for user, group in df.groupby(user_col):

            S = [[0]*self.number_of_states for _ in range(self.number_of_states)]
            S = np.matrix(S)
            data0 = df[df[user_col] == user]
            LOGGER.info("Currently creating a transition matrix for respondent: {}".format(user))
            for session, group in df.groupby(session_col):
                data00 = data0.loc[data0[session_col] == session]
                transitions = data00[state_col].tolist()
                def rank(c):
                    return ord(c) - ord('A')
                T = [rank(c) for c in transitions]
                M = [[0]*self.number_of_states for _ in range(self.number_of_states)]
                for (i, j) in zip(T, T[1:]):
                    M[i][j] += 1
                M = np.matrix(M)
                S = S+M
            self.transition_matrices[user] = S


    def plot(self, outputdirectory, user):
        q_df = pd.DataFrame(columns=self.states, index=self.states)

        count = 0

        while count < self.number_of_states:
            q_df.loc[self.states[count]] = self.prob_transition_matrices[user][count]
            count += 1


        q = q_df.values
        def _get_markov_edges(Q):
            edges = {}
            for col in Q.columns:
                for idx in Q.index:
                    edges[(idx, col)] = Q.loc[idx, col]
            return edges
        edges_wts = _get_markov_edges(q_df)
        G = nx.MultiDiGraph()
        G.add_nodes_from(self.states)
        for k, v in edges_wts.items():
            tmp_origin, tmp_destination = k[0], k[1]
            if v>0:
                G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos)
        edge_labels = {(n1, n2):d['label'] for n1, n2, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        if not os.path.exists(outputdirectory):
            os.mkdir(outputdirectory)
        filename = os.path.join(outputdirectory, '{}_probabilities.dot'.format(user))
        self.draw_markov_chain = nx.drawing.nx_pydot.write_dot(G, filename)

    def get_probability_matrices(self):
        self.prob_transition_matrices = {}
        for user, matrix in self.transition_matrices.items():
            LOGGER.info('Currently creating a probability transition matrix for respondent {}'.format(user))
            S = matrix.tolist()
            for row in S:
                n = sum(row)
                if n > 0:
                    row[:] = [f/sum(row) for f in row]
            #for row in S:
                #print(row)
            S = np.matrix(S)
            S = np.matrix.round(S, 3)
            self.prob_transition_matrices[user] = S


SAMPLEDATA = pd.DataFrame({'user': ["Anna", "Anna", "Anna", "Anna", "Anna", "Anna", "Anna", "Anna",
                                    "Anna", "Anna", "Anna", "Anna", "Anna", "Anna", "Anna", "Anna",
                                    "Anna", "Anna", "Anna", "Anna", "Bob", "Bob", "Bob", "Bob", "Bob",
                                    "Bob", "Bob", "Bob", "Paul", "Paul", "Paul", "Paul", "Paul", "Paul",
                                    "Paul", "Paul", "Paul", "Paul", "Paul", "Paul", "Paul", "Paul",
                                    "Paul", "Paul", "Paul", "Paul", "Paul", "Paul", "Karen", "Karen",
                                    "Karen", "Karen", "Karen", "Eric", "Eric", "Eric", "Eric", "Eric",
                                    "Eric", "Eric", "Eric", "Eric", "Eric", "Eric", "Eric", "Eric",
                                    "Eric", "Eric", "Eric", "Eric", "Eric", "Eric", "Eric", "Judith",
                                    "Judith", "Judith", "Judith", "Judith", "Judith", "Judith",
                                    "Judith", "Tim", "Tim", "Tim", "Tim", "Tim", "Tim", "Tim", "Tim", "Tim",
                                    "Tim", "Tim", "Tim", "Tim", "Tim", "Tim", "Tim", "Tim", "Tim", "Tim",
                                    "Tim", "Sandra", "Sandra", "Sandra", "Sandra", "Sandra"],
                           'session': [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                       2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                       3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2,
                                       1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                       2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                       3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2],
                           'state': ['A', 'F', 'B', 'C', 'C', 'C', 'A', 'A', 'B',
                                     'C', 'B', 'C', 'A', 'A', 'B', 'C', 'F', 'A',
                                     'B', 'F', 'C', 'B', 'D', 'F', 'A', 'A', 'C',
                                     'F', 'C', 'B', 'D', 'B', 'C', 'C', 'D', 'A',
                                     'F', 'A', 'B', 'A', 'C', 'F', 'E', 'D', 'D',
                                     'C', 'C', 'F', 'A', 'E', 'A', 'D', 'F', 'A',
                                     'F', 'B', 'C', 'C', 'D', 'A', 'A', 'B', 'C',
                                     'B', 'A', 'B', 'C', 'B', 'C', 'F', 'A', 'E',
                                     'F', 'C', 'B', 'C', 'F', 'A', 'A', 'C', 'F',
                                     'C', 'D', 'D', 'B', 'C', 'C', 'D', 'A', 'F',
                                     'A', 'B', 'A', 'C', 'F', 'E', 'E', 'E', 'E',
                                     'B', 'F', 'E', 'E', 'B', 'D', 'F'],
                           'date': ["2019-12-01 07:27:01", "2019-12-01 07:27:02", "2019-12-01 08:27:01", "2019-12-01 08:27:02",
                                    "2019-12-01 08:27:03", "2019-12-01 08:27:04", "2019-12-01 08:27:05", "2019-12-01 08:27:06",
                                    "2019-12-01 08:27:07", "2019-12-01 08:27:08", "2019-12-01 08:27:09", "2019-12-01 08:27:10", 
                                    "2019-12-01 08:27:11", "2019-12-01 08:27:12", "2019-12-01 08:27:13", "2019-12-01 08:27:14",
                                    "2019-12-01 08:29:15", "2019-12-01 10:29:13", "2019-12-01 10:29:14", "2019-12-01 10:29:15",
                                    "2019-12-01 08:27:01", "2019-12-01 08:27:02", "2019-12-01 08:27:03", "2019-12-01 08:27:04",
                                    "2019-12-01 10:29:13", "2019-12-01 10:29:14", "2019-12-01 10:29:15", "2019-12-01 10:29:16",
                                    "2019-12-01 08:27:01", "2019-12-01 08:27:02", "2019-12-01 08:27:03", "2019-12-01 08:27:04",
                                    "2019-12-01 08:27:05", "2019-12-01 08:27:06", "2019-12-01 08:27:07", "2019-12-01 08:27:08",
                                    "2019-12-01 08:27:09", "2019-12-03 08:27:06", "2019-12-03 08:27:07", "2019-12-03 08:27:08",
                                    "2019-12-03 08:27:09", "2019-12-03 08:27:11", "2019-12-04 08:27:06", "2019-12-04 08:27:07",
                                    "2019-12-04 08:27:08", "2019-12-04 08:27:09", "2019-12-04 08:27:11", "2019-12-04 08:27:11",
                                    "2019-12-01 07:27:01", "2019-12-01 07:27:02", "2019-12-01 08:27:01", "2019-12-01 08:27:02",
                                    "2019-12-01 08:27:03", "2019-12-01 07:27:01", "2019-12-01 07:27:02", "2019-12-01 08:27:01",
                                    "2019-12-01 08:27:02", "2019-12-01 08:27:03", "2019-12-01 08:27:04", "2019-12-01 08:27:05",
                                    "2019-12-01 08:27:06", "2019-12-01 08:27:07", "2019-12-01 08:27:08", "2019-12-01 08:27:09",
                                    "2019-12-01 08:27:10", "2019-12-01 08:27:11", "2019-12-01 08:27:12", "2019-12-01 08:27:13",
                                    "2019-12-01 08:27:14", "2019-12-01 08:29:15", "2019-12-01 10:29:13", "2019-12-01 10:29:14",
                                    "2019-12-01 10:29:15", "2019-12-01 08:27:01", "2019-12-01 08:27:02", "2019-12-01 08:27:03",
                                    "2019-12-01 08:27:04", "2019-12-01 10:29:13", "2019-12-01 10:29:14", "2019-12-01 10:29:15",
                                    "2019-12-01 10:29:16", "2019-12-01 08:27:01", "2019-12-01 08:27:02", "2019-12-01 08:27:03",
                                    "2019-12-01 08:27:04", "2019-12-01 08:27:05", "2019-12-01 08:27:06", "2019-12-01 08:27:07",
                                    "2019-12-01 08:27:08", "2019-12-01 08:27:09", "2019-12-03 08:27:06", "2019-12-03 08:27:07",
                                    "2019-12-03 08:27:08", "2019-12-03 08:27:09", "2019-12-03 08:27:11", "2019-12-04 08:27:06",
                                    "2019-12-04 08:27:07", "2019-12-04 08:27:08", "2019-12-04 08:27:09", "2019-12-04 08:27:11",
                                    "2019-12-04 08:27:11", "2019-12-01 07:27:01", "2019-12-01 07:27:02", "2019-12-01 08:27:01",
                                    "2019-12-01 08:27:02", "2019-12-01 08:27:03"]})
                                    
