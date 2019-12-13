'''
A simple way to create marov chains from data frames
'''
import pandas as pd
import numpy as np

class Markov():
    '''some docstring'''
    def __init__(self,df, state_col="state", date_col="date", user_col="user", session_col="session"):
        self.number_of_states = df[state_col].nunique()

        self.transition_matrices = {}

        for user, group in df.groupby(user_col):

            S = [[0]*self.number_of_states for _ in range(self.number_of_states)]
            S = np.matrix(S)
            data0 = df[df[user_col] == user]
            print("Currently creating a transition matrix for respondent: {}".format(user))
            for session, group in df.groupby(session_col): 
                data00 = data0.loc[data0[session_col] == session]
                transitions = data00[state_col].tolist()
                def rank(c):
                    return ord(c) - ord('A')
                T = [rank(c) for c in transitions]
                M = [[0]*self.number_of_states for _ in range(self.number_of_states)]
                for (i,j) in zip(T,T[1:]):
                    M[i][j] += 1
                M = np.matrix(M)
                S = S+M
            self.transition_matrices[user] = S


    def plot(self, path):
        q_df = pd.DataFrame(columns=state_col, index=state_col)
        for col in q_df.loc[state_col]:
            q_df.loc[state_col[col]] = S[col]
        q = q_df.values
        print('\n', q, q.shape, '\n')
        print(q_df.sum(axis=1))
        def _get_markov_edges(Q):
            edges = {}
            for col in Q.columns:
                for idx in Q.index:
                    edges[(idx,col)] = Q.loc[idx,col]
            return edges
        edges_wts = _get_markov_edges(q_df)
        pprint(edges_wts)
        G = nx.MultiDiGraph()
        G.add_nodes_from(states)
        for k, v in edges_wts.items():
            tmp_origin, tmp_destination = k[0], k[1]
            G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
        pprint(G.edges(data=True)) 
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos)
        edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)    
        nx.drawing.nx_pydot.write_dot(G, user +'_markov.dot')

    def get_probability_matrices(self):
        S = S.tolist()
        for row in S:
            n = sum(row)
            if n > 0:
                row[:] = [f/sum(row) for f in row]
        for row in S:
            print(row)
        S = np.matrix(S)
        S = np.matrix.round(S, 3)
        S.dump(user+"_prob_matrix.dat")


def create_chain(df, state_col="state", date_col="date", user_col="user"):
    '''
    Takes ... as input and returns ....
    '''


for respondent, group in data.groupby("visit.request_headers.X-robin-api-key"):
    S = [[0]*11 for _ in range(11)]
    S = np.matrix(S)
    data0 = data[data['visit.request_headers.X-robin-api-key'] == respondent]
    print("Currently creating a transition matrix for respondent: " + respondent)
    for session, group in data.groupby('session_id'): 
        data00 = data0.loc[data0['session_id'] == session]
        transitions = data00['state'].tolist()
        def rank(c):
            return ord(c) - ord('A')
        T = [rank(c) for c in transitions]
        M = [[0]*11 for _ in range(11)]
        for (i,j) in zip(T,T[1:]):
            M[i][j] += 1
        M = np.matrix(M)
        S = S+M
    S.dump(respondent+"_matrix.dat")
    print(S)
    S = S.tolist()
    for row in S:
        n = sum(row)
        if n > 0:
            row[:] = [f/sum(row) for f in row]
    for row in S:
        print(row)
    S = np.matrix(S)   
    S.dump(respondent+"_prob_matrix.dat")
    S = np.matrix.round(S, 3)
    S.dump(respondent+"_round_prob_matrix.dat")
    q_df = pd.DataFrame(columns=states, index=states)
    q_df.loc[states[0]] = S[0]
    q_df.loc[states[1]] = S[1]
    q_df.loc[states[2]] = S[2]
    q_df.loc[states[3]] = S[3]
    q_df.loc[states[4]] = S[4]
    q_df.loc[states[5]] = S[5]
    q_df.loc[states[6]] = S[6]
    q_df.loc[states[7]] = S[7]
    q_df.loc[states[8]] = S[8]
    q_df.loc[states[9]] = S[9]
    q_df.loc[states[10]] = S[10]
    print(q_df)
    q = q_df.values
    print('\n', q, q.shape, '\n')
    print(q_df.sum(axis=1))
    def _get_markov_edges(Q):
        edges = {}
        for col in Q.columns:
            for idx in Q.index:
                edges[(idx,col)] = Q.loc[idx,col]
        return edges
    edges_wts = _get_markov_edges(q_df)
    pprint(edges_wts)
    G = nx.MultiDiGraph()
    G.add_nodes_from(states)
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    pprint(G.edges(data=True)) 
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos)
    edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)    
    nx.drawing.nx_pydot.write_dot(G, respondent+'_markov.dot')
print('Done.')

sampledata = pd.DataFrame({'user':[1,1,1,2,2,2], 'session':[1,1,2,44,45,45], 'state': ['A','B','C','C','B','A'], 'date': [1,2,3,4,5,6]})
