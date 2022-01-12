import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper
import sys

# func get_fork_nodes: get depth layer that forks branches
# input: list - onnx.shape_inference.infer_shapes.graph.node
# output: list - to be appended to metadata of layers
def get_fork_nodes(nodes):
    fork_list = []
    for i, node in enumerate(nodes):
        fork_count = 0
        output_name = node.output[0]
        for node_ds in nodes[i:]:
            if output_name in node_ds.input:
                fork_count += 1
        if fork_count > 1:
            fork_list.append([i, fork_count])
    return fork_list

# func get_join_nodes: get list and degree of concatention layers
def get_join_nodes(nodes):
    join_list = []
    input_name = []
    for i, node in enumerate(nodes):
        join_count = 0
        input_name = node.input
        for node_us in nodes[:i]:
            if node_us.output[0] in input_name:
                join_count += 1
        if join_count > 1:
            join_list.append([i, join_count])
    return join_list

# func get_edges_list: get list of nodes that are connected
# input: list
# output: list - [[a.b],[b,c],[b,d],...]
def get_edges_list(nodes, corr_list):
    edge_list = []
    for i, node in enumerate(nodes):
        output_name = node.output[0]
        node_edge = []
        for j, ds in enumerate(nodes[i:]):
            if output_name in ds.input:
                node_edge.append(i+j)
        edge_list.append(node_edge)

    reduced_edge_list = []
    first_node = 0
    for i, node in enumerate(edge_list):
        if i in corr_list:
            first_node = i
            for j, ed in enumerate(node):
                second_node = ed
                found = False
                while not found:
                    if second_node in corr_list:
                        found = True
                        reduced_edge_list.append([corr_list.index(first_node), corr_list.index(second_node)])
                    else:
                        if second_node < len(edge_list)-1:
                            second_node = (edge_list[second_node])
                            second_node = second_node[0]
                        else:
                            break
                        
    return reduced_edge_list
            
# func get_conv_layers: get list of layers that are conv
# input: list
# output: list 
def get_conv_layers(nodes):
    conv_layers = filter(lambda x: x.op_type == "Conv", nodes)
    return list(conv_layers)

# func get_fmap_out: get list of output feature map size after each layer
# input: list, Int (input data size)
# output: list 
def get_fmap_out(nodes, data_size):
    fmap_out_list = []
    fmap_in = data_size
    for node in nodes:
        #  [(W−K+2P)/S]+1
        kernel = []
        pad = []
        stride = []
        if node.op_type == "Conv" or node.op_type == "MaxPool":
            for i in node.attribute:
                if i.name == "kernel_shape":
                    kernel = i.ints
                if i.name == "pads":
                    pad = i.ints
                if i.name == "strides":
                    stride = i.ints
            fmap_out = int((fmap_in-kernel[0]+2*pad[0])/stride[0])+1
            fmap_in = fmap_out
        fmap_out_list.append(fmap_in)
    return fmap_out_list

# func get_weight_by_name: get weight from initializer
# input: model, String (layer name)
# output: list 
def get_weight_by_name(model, name):
        for weight in model.graph.initializer:
            if weight.name == name:
                return numpy_helper.to_array(weight)


if __name__ == "__main__":
    # Import Model

    # model_name = "resnet101-v1-7.onnx"
    model_name = "mobilenetv2-7.onnx"
    # model_name = "inception-v1-3.onnx"

    model = onnx.load(model_name)
    model = onnx.shape_inference.infer_shapes(model)

    # print(len(numpy_helper.to_array(model.graph.initializer[0])[0]))

    fw = open("test.in", "w")

    # Process inputs
    fmap_out_list = get_fmap_out(model.graph.node, 224)
    fork_nodes_degree = get_fork_nodes(model.graph.node)
    join_nodes_degree = get_join_nodes(model.graph.node)

    fork_nodes = []
    for i in fork_nodes_degree:
        fork_nodes.append(i[0])
    
    join_nodes = []
    for i in join_nodes_degree:
        join_nodes.append(i[0])

    simplified_count = 0
    fork_count = 0
    corr_list = []

    # Write info of conv, branch layers
    for i, node in enumerate(model.graph.node):
        if i in join_nodes:
            channel = 0
            for ds in (model.graph.node[i:]):
                if ds.op_type == "Conv":
                    input_name = ds.input[1]
                    channel = len(get_weight_by_name(model, input_name))
                    break
            fw.write("%d d join %d \n" %(simplified_count, len(node.input)))
            corr_list.append(i)
            simplified_count += 1
            fw.write("%d %d %d\n" %(fmap_out_list[i], fmap_out_list[i], channel))    
        if node.op_type == "Conv":
            fw.write("%d c \n" %(simplified_count))

            corr_list.append(i)

            simplified_count += 1
            kernel = []
            for j in node.attribute:
                if j.name == "kernel_shape":
                    kernel = j.ints
            # input_name = [n for n in node.input if "w" in n]
            input_name = node.input[1]

            fw.write("%d %d %d %d %d %d\n" %(fmap_out_list[i], 
            fmap_out_list[i], 
            kernel[0], 
            kernel[1],
            len(get_weight_by_name(model, input_name)[0]), 
            len(get_weight_by_name(model, input_name))
            ))    
        if i in fork_nodes:
            channel = 0
            for ds in (model.graph.node[i:]):
                if ds.op_type == "Conv":
                    # input_name = [n for n in ds.input if "w" in n]
                    input_name = ds.input[1]
                    channel = len(get_weight_by_name(model, input_name))
                    break
            fw.write("%d d fork %d \n" %(simplified_count, fork_nodes_degree[fork_count][1]))

            corr_list.append(i)

            simplified_count += 1
            fork_count += 1
            fw.write("%d %d %d\n" %(fmap_out_list[i],
            fmap_out_list[i], channel
            ))

    fw.write("\n")

    # Write edge connections for conv and branch layers
    corr_edge_list = get_edges_list(model.graph.node, corr_list)    
    for node in corr_edge_list:
        fw.write("%d %d\n" %(node[0], node[1]))

    fw.close()





