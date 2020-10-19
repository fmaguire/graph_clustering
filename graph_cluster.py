#!/usr/bin/env python

import networkx as nx
import argparse
import numpy as np
import pandas as pd
import sklearn
import sklearn.cluster


def parse_graph(node_distances_fp):
    """
    Parses TSV containing all vs all node distances into a networkx
    graph structure
    """

    dist_df = pd.read_csv(node_distances_fp, sep='\t', index_col=0)

    # check all nodes are represented as rows and columns
    if all(dist_df.columns != dist_df.index):
        raise ValueError(f"All vs all TSV must be square: {node_distances_fp}"
                          " columns and row names do not match")

    # check if all distances are floats
    if all(dist_df.dtypes != 'float64'):
        raise ValueError(f"Non-float values in TSV: {node_distances_fp} "
                          "please fix and choose an appropriate value for "
                          "NaNs")

    # check if distances are symmetric and deal with float epsilon
    if not np.all(np.abs(dist_df.values - dist_df.values.T) < 1e-8):
        raise ValueError(f"Distances are not symmetrical: {node_distances_fp}"
                          " please fix or modify code to create directed "
                          "graph")

    # get graph
    graph = nx.Graph(dist_df)

    return dist_df, graph


def cluster_graph(dist_df, graph):
    """
    Perform unsupervised clustering of networkx graph
    """
    # change and modify as per your data and needs (although remember
    # you are using a precomputed distance matrix and most sklearn assume
    # you are providing raw data)
    model = sklearn.cluster.SpectralClustering(affinity='precomputed',
                                               n_clusters=5)

    model.fit(dist_df.values)

    labels = {node: label for node, label in zip(dist_df.columns,
                                                 model.labels_)}

    nx.set_node_attributes(graph, labels, "cluster")

    return graph


def add_metadata(graph, metadata_fp, metadata_col):
    """
    Add metadata as attribute to graph nodes
    """
    metadata = pd.read_csv(metadata_fp, sep='\t')

    if 'node' not in metadata.columns:
        raise ValueError(f"Must be a column called 'node' in metadata with "
                         f"node labels: {metadata_fp}")

    if metadata_col not in metadata.columns:
        raise ValueError(f"Supplied metadata col ({metadata_col} not found in "
                         f" {metadata_fp}")

    if set(graph.nodes) != set(metadata['node'].values):
        raise ValueError(f"Metadata node column doesn't contain same values "
                          "as node names in the graph")

    metadata_labels = metadata.set_index('node')[metadata_col].to_dict()

    nx.set_node_attributes(graph, metadata_labels,
                           metadata_col.replace(' ', '_'))

    return graph


def write_graph(graph, output_fp):
    """
    Output graph as gexf formatted for visualising in gephi
    """
    output = output_fp + ".gexf"
    print(f"Graph written to {output}, visualise in gephi or similar")
    nx.write_gexf(graph, output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cluster and visualise an'
                                                 ' arbitrary graph from a'
                                                 ' matrix of all vs all '
                                                 'distances')
    parser.add_argument('--allvsall',
                        help='Path to TSV containing all node distances i.e. '
                             'all vs all square matrix',
                        required=True)
    parser.add_argument('--metadata',
                        help='Path to TSV containing node metadata for '
                             ' annotation (must cotain "node" and specified '
                             ' metadata column)',
                        required=True)
    parser.add_argument('--metadata_col',
                        help='Which column to use for node colouring',
                        required=True)
    parser.add_argument('--output', default="out",
                         help="Path to output gexf file")
    args = parser.parse_args()

    dist_df, graph = parse_graph(args.allvsall)

    cluster_graph = cluster_graph(dist_df, graph)

    graph = add_metadata(graph, args.metadata, args.metadata_col)

    write_graph(graph, args.output)
