import argparse 
import tensorflow as tf
import numpy as np

UP_MEAN = 7.78846154
DOWN_MEAN = 12.33076923
UP_STD = 0.81350473
DOWN_STD = 1.03432223

# Linear regression Frozen path
frozen_path = "model/LR_frozen_model.pb"

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def normalize_size(x):
    #Values obtained from train.py output
    mean = np.asarray([[UP_MEAN, DOWN_MEAN]])
    std = np.asarray([[UP_STD, DOWN_STD]])
    x_normalized = (x - mean) / std
    return x_normalized

def convert_sizes(size):
	size = int(size)
	if size >= 400:
		return "Large"
	elif size <= 399 and size >= 200:
		return "medium"
	elif size < 199:
		return "small"

def predict_size(x_input):
    # input: [width, length, thickness]
    # output: [size, size_classification]
    x_input = normalize_size(x_input)
    graph = load_graph("frozen_models/LR_frozen_model.pb")

    #input and output node
    x = graph.get_tensor_by_name('prefix/x:0')
    y = graph.get_tensor_by_name('prefix/Wx_b/Add:0')

    with tf.Session(graph = graph) as sess:
        y_output = sess.run(y, feed_dict={x: x_input})

    return y_output, convert_sizes(y_output)