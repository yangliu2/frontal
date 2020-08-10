# Frontal
Frontal lobe is known to process reasoning in human, hence the package name 
frontal. This package is used to analysze causal inference graphs for 
determining causal relationships. It's based on Judea Pearl's book about 
Directed Acyclic Graphs. Using the book CounterFactuals and Causal Inference
by Stephen L Mogan and Christopher Winship. 

# Usage
`python frontal/graphy.py`

For generating graph picture in `graph` directory `graphy.png`. The numbered
graphs `1.png` is each backdoor path between start and end nodes. It is used
to check whether the backdoor criterion is satified and give a set of variables
to condition on, so causal relationship can be determined. 
