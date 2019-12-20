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
                 session_col="session"):
        self.number_of_states = df[state_col].nunique()

        self.transition_matrices = {}

        self.states = df[state_col].unique()

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
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
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


SAMPLEDATA = pd.DataFrame({'user': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                    3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5,
                                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                    5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
                                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                                    7, 7, 7, 8, 8, 8, 8, 8],
                           'session': [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                       2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                                       9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10,
                                       10, 11, 11, 11, 11, 11, 11, 44, 44, 45, 45,
                                       45, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66,
                                       66, 66, 66, 66, 66, 66, 66, 813, 813, 813,
                                       6, 6, 6, 6, 8, 8, 8, 8, 13, 13, 13, 13, 13,
                                       13, 13, 13, 13, 19, 19, 19, 19, 19, 20, 20,
                                       20, 20, 20, 20, 66, 66, 68, 68, 68],
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
                           'date': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                    15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                    15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 1, 2, 3,
                                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                    18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4,
                                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    19, 20, 1, 2, 3, 4, 5]})
