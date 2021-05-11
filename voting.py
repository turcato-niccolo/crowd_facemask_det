import numpy as np
from collections import defaultdict
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt


def load_candidates(filename, separator):
    candidates_ = defaultdict()
    file = open(filename)

    file.readline()  # skip first line

    for line in file:
        [label, name] = line.split(separator)
        candidates_[int(label)] = name[:-2]

    return candidates_


def load_votes(filename, separator):
    votes = []
    file = open(filename)
    for line in file:
        entries = line.split(separator)
        votes.append([int(entries[i]) for i in range(len(entries))])
    return votes


def apply_voting(votes_, rule_score, candidates_tags):
    """"rule_score is a lambda that computes the score of a candidate i in a voting list of preferences
        rule_score(preferences, candidate_of_interest)
    """
    scores = defaultdict()
    for vote in votes_:
        for tag in candidates_tags:
            if tag in scores:
                scores[tag] += rule_score(vote, tag)
            else:
                scores[tag] = rule_score(vote, tag)
    return scores


def reduce_vote_pairwise(candidate_a, candidate_b, vote):
    reduction = []
    for v in vote:
        if v == candidate_b or v == candidate_a:
            reduction.append(v)
    return reduction


def reduce_pairwise_election(candidate_a, candidate_b, votes_):
    return [reduce_vote_pairwise(candidate_a, candidate_b, vote) for vote in votes_]


def create_majority_graph(votes_, candidates_tag):
    mj_graph = nx.DiGraph()
    mj_graph.add_nodes_from(candidates_tag)
    for pair in it.combinations(candidates_tag, 2):  # produces combinations without repetition
        candidate_a, candidate_b = pair
        pair_election = reduce_pairwise_election(candidate_a, candidate_b, votes_)
        scores = apply_voting(pair_election, plurality_rule, candidates_tag)
        if scores[candidate_a] > scores[candidate_b]:
            mj_graph.add_edge(candidate_a, candidate_b, weight=scores[candidate_a])
        else:
            mj_graph.add_edge(candidate_b, candidate_a, weight=scores[candidate_b])

    return mj_graph


def print_scores(candidates_tag_names, scores):
    for candidate_tag in candidates_tag_names.keys():
        print(candidates_tag_names[candidate_tag] + ": " + str(scores[candidate_tag]))


def draw_mj_graph(G):
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.3)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='r', width=10, label=labels)
    plt.show()


def copeland_voting(mj_graph):
    scores_from_graph = defaultdict()
    for node in list(mj_graph.nodes()):
        scores_from_graph[node] = len(mj_graph.out_edges(node)) - len(mj_graph.in_edges(node))
    return scores_from_graph

def one_hot2vote(one_hot_):
    """ It essentially returns the indices of the vector ordered in descend order
    :param      one_hot_: one_hot vector representing a vote (must have positive entries)
    :return:    the vote corresponding to the one_hot vector: [first-choice, second-choice, ..], where the choices are
                indices of the vector
    """
    one_hot = one_hot_
    vote = []
    for i in range(len(one_hot)):
        candidate_tag = np.argmax(one_hot)
        vote.append(candidate_tag)
        one_hot[candidate_tag] = -1
    return vote

if __name__ == '__main__':
    plurality_rule = lambda preferences, candidate_of_interest: 1 if preferences[0] == candidate_of_interest else 0
    print(one_hot2vote([0.3, 0.5, 0.2]))



