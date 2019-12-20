'''
A simple way to create marov chains from data frames
'''
import logging
import os
import pandas as pd
import numpy as np
import networkx as nx
import pydot


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class Markov():
    '''
    A Markov chain based on a dataframe with users, timestamps, and states.

    Parameters:
    -----------
    df : pandas Dataframe
         Dataframe with input data
    state_col : str
         The column name of the column containing the state
    timestamp_col : str
         The column name of the column containing the timetamp.
         The column must be sortable in a cronological order.
    session_col : str
         The column name of the column containing a session ID.
    user_col : str
         The column name of the column containing a user ID.
    sort : Bool
         If set to False, sorting of the timestamp column is skipped.

    Attributes:
    -----------
    number_of_states : int
         The number of states in the state space.
    states : list
         The names of all states (i.e., the state space)
    transition_matrices: dict
         A dict with user ids as keys, and their individual transition
         matrices as values.
    '''
    def __init__(self, df, state_col="state", timestamp_col="timestamp", user_col="user",
                 session_col="session", sort=True):
        self.number_of_states = df[state_col].nunique()

        self.transition_matrices = {}

        self.states = df[state_col].unique()

        if sort:
            df = df.sort_values(by=[timestamp_col])

        else:
            LOGGER.info('You specified that you do *not* want to let df2markov sort the '
                        'timestamp column. We hope that you know what you are doing and made '
                        'sure that the column is in chronological order. '
                        'Otherwise, the results will be wrong.')

        for user, group in df.groupby(user_col):

            matrix_states = [[0]*self.number_of_states for _ in range(self.number_of_states)]
            matrix_states = np.matrix(matrix_states)
            data0 = df[df[user_col] == user]
            LOGGER.info("Currently creating a transition matrix for respondent: {}".format(user))
            for session, group in df.groupby(session_col):
                data00 = data0.loc[data0[session_col] == session]
                transitions = data00[state_col].tolist()
                def rank(state_value):
                    return ord(state_value) - ord('A')
                trans = [rank(state_value) for state_value in transitions]
                trans_matrix = [[0]*self.number_of_states for _ in range(self.number_of_states)]
                for (i, j) in zip(trans, trans[1:]):
                    trans_matrix[i][j] += 1
                trans_matrix = np.matrix(trans_matrix)
                matrix_states = matrix_states+trans_matrix
            self.transition_matrices[user] = matrix_states


    def plot(self, outputdirectory, user):
        '''
        Creates a dot file with a graphic representation of transition
        probabilities for a given user.

        Parameters
        ----------
        outputdir : str
             Directory in which to store the plot.
        user : str, int, etc.
             The user ID
        '''
        states_df = pd.DataFrame(columns=self.states, index=self.states)

        count = 0

        while count < self.number_of_states:
            states_df.loc[self.states[count]] = self.prob_transition_matrices[user][count]
            count += 1

        #q = states_df.values
        def _get_markov_edges(state_columns):
            edges = {}
            for col in state_columns.columns:
                for idx in state_columns.index:
                    edges[(idx, col)] = state_columns.loc[idx, col]
            return edges
        edges_wts = _get_markov_edges(states_df)
        graph_object = nx.MultiDiGraph()
        graph_object.add_nodes_from(self.states)
        for k, weight in edges_wts.items():
            tmp_origin, tmp_destination = k[0], k[1]
            if weight > 0:
                graph_object.add_edge(tmp_origin, tmp_destination, weight=weight, label=weight)
        pos = nx.drawing.nx_pydot.graphviz_layout(graph_object, prog='dot')
        nx.draw_networkx(graph_object, pos)
        edge_labels = {(n1, n2):d['label'] for n1, n2, d in graph_object.edges(data=True)}
        nx.draw_networkx_edge_labels(graph_object, pos, edge_labels=edge_labels)
        if not os.path.exists(outputdirectory):
            os.mkdir(outputdirectory)
        filename = os.path.join(outputdirectory, '{}_probabilities.dot'.format(user))
        self.draw_markov_chain = nx.drawing.nx_pydot.write_dot(graph_object, filename)

    def get_probability_matrices(self):
        '''
        Calculates probability matrices for all users based on their transition
        matrices. Results are stored in a new attribute, prob_transition_matrices.
        '''
        self.prob_transition_matrices = {}
        for user, matrix in self.transition_matrices.items():
            LOGGER.info("Currently creating a probability "
                        " transition matrix for respondent {}".format(user))
            matrix_states = matrix.tolist()
            for row in matrix_states:
                number_row = sum(row)
                if number_row > 0:
                    row[:] = [f/sum(row) for f in row]
            #for row in S:
                #print(row)
            matrix_states = np.matrix(matrix_states)
            matrix_states = np.matrix.round(matrix_states, 3)
            self.prob_transition_matrices[user] = matrix_states

    def aggregate(self, how='percentage'):
        '''
        Aggregates the user-specific Markov chains.

        Parameters
        ----------
        how : {'percentage' | 'probability' | 'frequency'}
            Specifies how to aggregate over all users.
            'probability' takes the mean of the individual probabilities.
            'percentage' does the same, but multiplies by 100.
            'frequency' sums the individual transitions in absolute numbers.

        Returns
        -------
        An aggregated matrix over all users.
        '''
        LOGGER.info('Currently aggregating the probability transition matrix for all respondents')
        aggregate_matrix = np.zeros((self.number_of_states, self.number_of_states))


        if how == 'percentage':
            for i, j in np.ndindex(aggregate_matrix.shape):
                aggregate_matrix[i, j] = np.mean([matrix[i, j] for matrix in self.prob_transition_matrices.values()]) * 100

        elif how == 'probability':
            for i, j in np.ndindex(aggregate_matrix.shape):
                aggregate_matrix[i, j] = np.mean([matrix[i, j] for matrix in self.prob_transition_matrices.values()])

        elif how == 'frequency':
            for i, j in np.ndindex(aggregate_matrix.shape):
                aggregate_matrix[i, j] = np.sum([matrix[i, j] for matrix in self.transition_matrices.values()])
        else:
            LOGGER.error('You need to specify the aggregation function as "percentage"'
                         ", 'probability', or 'frequency'")
        return aggregate_matrix

SAMPLEDATA = pd.DataFrame({'user': ["Anna", "Anna", "Anna", "Anna", "Anna", "Anna",
                                    "Anna", "Anna", "Anna", "Anna", "Anna", "Anna",
                                    "Anna", "Anna", "Anna", "Anna", "Anna", "Anna",
                                    "Anna", "Anna", "Bob", "Bob", "Bob", "Bob", "Bob",
                                    "Bob", "Bob", "Bob", "Paul", "Paul", "Paul",
                                    "Paul", "Paul", "Paul", "Paul", "Paul", "Paul",
                                    "Paul", "Paul", "Paul", "Paul", "Paul", "Paul",
                                    "Paul", "Paul", "Paul", "Paul", "Paul", "Karen",
                                    "Karen", "Karen", "Karen", "Karen", "Eric", "Eric",
                                    "Eric", "Eric", "Eric", "Eric", "Eric", "Eric",
                                    "Eric", "Eric", "Eric", "Eric", "Eric", "Eric",
                                    "Eric", "Eric", "Eric", "Eric", "Eric", "Eric",
                                    "Judith", "Judith", "Judith", "Judith", "Judith",
                                    "Judith", "Judith", "Judith", "Tim", "Tim", "Tim",
                                    "Tim", "Tim", "Tim", "Tim", "Tim", "Tim",
                                    "Tim", "Tim", "Tim", "Tim", "Tim", "Tim", "Tim",
                                    "Tim", "Tim", "Tim", "Tim", "Sandra", "Sandra",
                                    "Sandra", "Sandra", "Sandra"],
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
                           'timestamp': ["2019-12-01 07:27:01", "2019-12-01 07:27:02",
                                         "2019-12-01 08:27:01", "2019-12-01 08:27:02",
                                         "2019-12-01 08:27:03", "2019-12-01 08:27:04",
                                         "2019-12-01 08:27:05", "2019-12-01 08:27:06",
                                         "2019-12-01 08:27:07", "2019-12-01 08:27:08",
                                         "2019-12-01 08:27:09", "2019-12-01 08:27:10",
                                         "2019-12-01 08:27:11", "2019-12-01 08:27:12",
                                         "2019-12-01 08:27:13", "2019-12-01 08:27:14",
                                         "2019-12-01 08:29:15", "2019-12-01 10:29:13",
                                         "2019-12-01 10:29:14", "2019-12-01 10:29:15",
                                         "2019-12-01 08:27:01", "2019-12-01 08:27:02",
                                         "2019-12-01 08:27:03", "2019-12-01 08:27:04",
                                         "2019-12-01 10:29:13", "2019-12-01 10:29:14",
                                         "2019-12-01 10:29:15", "2019-12-01 10:29:16",
                                         "2019-12-01 08:27:01", "2019-12-01 08:27:02",
                                         "2019-12-01 08:27:03", "2019-12-01 08:27:04",
                                         "2019-12-01 08:27:05", "2019-12-01 08:27:06",
                                         "2019-12-01 08:27:07", "2019-12-01 08:27:08",
                                         "2019-12-01 08:27:09", "2019-12-03 08:27:06",
                                         "2019-12-03 08:27:07", "2019-12-03 08:27:08",
                                         "2019-12-03 08:27:09", "2019-12-03 08:27:11",
                                         "2019-12-04 08:27:06", "2019-12-04 08:27:07",
                                         "2019-12-04 08:27:08", "2019-12-04 08:27:09",
                                         "2019-12-04 08:27:11", "2019-12-04 08:27:11",
                                         "2019-12-01 07:27:01", "2019-12-01 07:27:02",
                                         "2019-12-01 08:27:01", "2019-12-01 08:27:02",
                                         "2019-12-01 08:27:03", "2019-12-01 07:27:01",
                                         "2019-12-01 07:27:02", "2019-12-01 08:27:01",
                                         "2019-12-01 08:27:02", "2019-12-01 08:27:03",
                                         "2019-12-01 08:27:04", "2019-12-01 08:27:05",
                                         "2019-12-01 08:27:06", "2019-12-01 08:27:07",
                                         "2019-12-01 08:27:08", "2019-12-01 08:27:09",
                                         "2019-12-01 08:27:10", "2019-12-01 08:27:11",
                                         "2019-12-01 08:27:12", "2019-12-01 08:27:13",
                                         "2019-12-01 08:27:14", "2019-12-01 08:29:15",
                                         "2019-12-01 10:29:13", "2019-12-01 10:29:14",
                                         "2019-12-01 10:29:15", "2019-12-01 08:27:01",
                                         "2019-12-01 08:27:02", "2019-12-01 08:27:03",
                                         "2019-12-01 08:27:04", "2019-12-01 10:29:13",
                                         "2019-12-01 10:29:14", "2019-12-01 10:29:15",
                                         "2019-12-01 10:29:16", "2019-12-01 08:27:01",
                                         "2019-12-01 08:27:02", "2019-12-01 08:27:03",
                                         "2019-12-01 08:27:04", "2019-12-01 08:27:05",
                                         "2019-12-01 08:27:06", "2019-12-01 08:27:07",
                                         "2019-12-01 08:27:08", "2019-12-01 08:27:09",
                                         "2019-12-03 08:27:06", "2019-12-03 08:27:07",
                                         "2019-12-03 08:27:08", "2019-12-03 08:27:09",
                                         "2019-12-03 08:27:11", "2019-12-04 08:27:06",
                                         "2019-12-04 08:27:07", "2019-12-04 08:27:08",
                                         "2019-12-04 08:27:09", "2019-12-04 08:27:11",
                                         "2019-12-04 08:27:11", "2019-12-01 07:27:01",
                                         "2019-12-01 07:27:02", "2019-12-01 08:27:01",
                                         "2019-12-01 08:27:02", "2019-12-01 08:27:03"]})
