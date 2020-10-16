# Graph Cluster and Annotate

Quick script that takes:
- an all vs all distance matrix (must be square, with
rows and columns labelled with node names, and distances must be floats),
- a metadata file (must contain all nodes in a column call "node") 
- a column name in that metadata file

Then does a quick spectral graph clustering (fixed components change as needed
for your data) and annotates the graph according to assigned cluster and
specified metadata.

Finally script exports graph as a `.gexf` file for opening and interactive
visualisation in gephi.

Alternatively, variously gephi plugins and options can do everything this 
script does with a lot more clicking!

## Dependencies

- networkx
- pandas
- numpy
- scikit-learn

## Usage

`python graph_cluster.py --allvsall test/test.tsv --metadata test/metadata.csv --metadata_col metadata1 --output test/out_new.gexf`
