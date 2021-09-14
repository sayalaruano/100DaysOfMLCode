# NotesDay4

## Machine Learning with Graphs course - Lecture 2.1: Traditional Feature-based Methods

### Machine Learning tasks 

The traditional ML pipeline is all about designing proper features.

In GML, there are two types of features: 

* **Importance-based features:** capture the importance of a node in a graph, and they are useful for predicting influential nodes (e.g. node degree, node centrality measures). 

* **Structure-based features:** capture topological properties of the local neighborhood around a given node of interest. In other words, these are features related to how a node is 
positioned in the rest of the network and what is its local structure (e.g. node degree, clustering coefficient, graphlet degree vector). These features are 
useful for predicting a particular role a node plays in a graph. 


In general, traditional ML pipelines consist of two steps:

1. Represent the nodes, edges, or the entire graph and represent them with a vector of features. Then, train traditional ML algorithms (RF, SVM, NN, etc) with the mentioned vector of features as input. 
2. Given a new node, link, or graph, obtain the vector of features and make predictions. 

### Feature design 

Using effective features over graphs is the key to achieving good test performance. 

Traditional ML pipeline uses hand-designed features 

**General goal:** make predictions for a set of objects

**Design choices:** 

* **Features:** d-dimensional vectors
* **Objects:** nodes, edges, set of nodes, graphs
* **Objective function:** labels that are intended to predict

Given a graph G = (V, E), the idea is obtaining a function that makes predictions.

### Node-level prediction 

This is a semi-supervised task because there are some labeled nodes, and the goal is to predict the corresponding category of unlabeled nodes. 

**Goal:** characterize the structure and position of a node in the network. 

Some measures to make this characterization: 

#### Node degree
A drawback of this measure is that treats all neighboring nodes equally without capturing their importance, positions in the network, or other particularities. 

#### Node centrality
Takes the node importance in a graph into account.

* Eigenvector centrality: 
	* A node is important if surrounded by important neighboring nodes. The importance is the normalized sum of the importance of the nodes that it links to. 
	* The formula of this measure can be represented as the eigenvalue-eigenvector equation. 
	* In undirected graphs, the Perron-Frobenius theorem demonstrates that the largest eigenvalue is always positive and unique. 
	* The leading eigenvector is used for centrality. 

* Betweenness centrality: 
	* A node is important if it lies on many shortest paths between other nodes.
	* A node is important if it serves as a bridge or transit hub. 

* Closeness centrality: 
	* A node is important if it has small shortest path lengths to all other nodes. 
	* The more center is a node, it has the shorter paths to everyone else. 

#### Clustering coefficient
It measures the local structure of a node, in particular how connected are the neighboring nodes of a given node. It counts the number of triangles in the ego-network (network induced by the node itself and its neighbors - degree 1 neighborhood network around a given node). 

Social networks naturally evolve by triangle closing, because it is very likely that people with friends in common also become friends. 

#### Graphlets
Graphlets are rooted non-isomorphic subgraphs. It is possible to use the graphlet degree vectors (GDV - graphlet-base features for nodes) to characterize 
the local neighborhood structure of a node and the node’s local network topology (more detailed than the degree or clustering coefficient). GDV count the 
number of times a given graphlet appears rooted at a given node.
