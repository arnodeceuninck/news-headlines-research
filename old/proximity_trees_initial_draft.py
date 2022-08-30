# Code from proximity trees, from which I'm going to remove a lot, so keep it here in case I would still need it later

import numpy as np


class Branch:
    def __init__(self, exemplar, subtree=None):
        self.exemplar = exemplar  # An object, if from a node it is closest to this exemplar, it goes to given tree
        self.subtree = subtree  # Should be an internal node or a leaf node


class TreeNode:
    pass


def get_random_sample_per_class(data):
    data_x, data_y = data

    unique_y = np.unique(data_y)  # The different classes

    samples = []
    for y in unique_y:
        # Get the elements with given label
        indices = np.where(data_y == y)[0]
        x = data_x[indices]
        # Randomly sample an element from x
        exemplar = x[np.random.randint(0, len(x))]
        samples.append(exemplar)

    return samples


def gen_candidate_splitter(data, measures):
    measure = np.random.choice(measures)
    exemplars = get_random_sample_per_class(data)
    return measure, exemplars


def calculate_gini_index(split):
    pass


def get_best_splitter(splits):
    # Return argmax of gini index of each of the splits
    gini_indices = [calculate_gini_index(split) for split in splits]
    return np.argmax(gini_indices)  # TODO: Should return mesure and exemplars


def get_closest_exemplars(exemplars, data, measure):
    # Return the index of the exemplar that is closest to the data point
    distances = [measure(data, exemplar) for exemplar in exemplars]
    closest_exemplar_indices = np.argmin(distances)
    assert len(closest_exemplar_indices) == len(data)
    return closest_exemplar_indices

def get_candidate_splits(data, r, measures):
    splits = []
    for i in range(r):
        # Generate a candidate splitter
        candidate_splitter = gen_candidate_splitter(data, measures)
        splits.append(candidate_splitter)
    return splits

class InternalNode(TreeNode):
    def __init__(self, data, r, measures):
        assert len(data) == 2  # first index contains features, second the labels

        # self.measure = measure if measure is not None else lambda x, y: np.linalg.norm(x - y)
        self.branches = []  # Contains internal nodes or leaf nodes

        # TODO: Splits rndomly from picking a random test, selecting the winner and uniformly picking a non-winner.
        splits = get_candidate_splits(data, r, measures)

        self.measure, exemplars = get_best_splitter(splits)

        closest_exemplars = get_closest_exemplars(exemplars, data, self.measure)

        for i, exemplar in enumerate(exemplars):
            # Find the data items that are closest to the exemplar (and thus having i in the closest_exemplars array)
            closest_data_items = data[closest_exemplars == i]
            subtree = get_node(closest_data_items, r, measures)
            self.branches.append(Branch(exemplar, subtree))  # TODO: Fix label somewhere here instead of i

    def predict(self, data):
        # Return the label of the closest subtree to the data point
        exemplar_distance = [self.measure(data, branch.exemplar) for branch in self.branches]
        closest_exemplars = get_closest_exemplars(exemplar_distance, [data], self.measure)
        branch = self.branches[closest_exemplars[0]]
        return branch.subtree.predict(data)


# if all data reaching a node has the same class (node is pure), create_leaf function creates a new leaf node and assigns this class lbel to its field class
class LeafNode(TreeNode):
    def __init__(self, class_label):
        self.class_label = class_label  # This label is assigned to all data reaching this node

    def predict(self, data):
        return self.class_label
#%%
def is_pure(data):
    # Check if all data has the same class label
    unique_y = np.unique(data[1])
    return len(unique_y) == 1


def get_node(data, r, measures):
    if is_pure(data):
        class_label = data[1][0]  # Label of first element in data
        return LeafNode(class_label)
    else:
        return InternalNode(data, r, measures)


# Splitting criteriaa
class ProximityTreeClassifier:

    def __init__(self, r, measures):
        self.root = None
        self.r = r
        self.measures = measures

    def fit(self, data):
        self.root = get_node(data, self.r, self.measures)
        return self

    def predict(self, data):
        return self.root.predict(data)