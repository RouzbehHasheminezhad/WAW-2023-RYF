import logging
import os
import shutil
from engine.config.config import *

# get_categories(data_dir) lists the network categories in 'data_dir'.
def get_categories(data_dir):
    base_path = data_dir
    result_list = []
    for category_name in os.listdir(base_path):
        if category_name == "__MACOSX" or category_name == "nets.pkl" or category_name == ".DS_Store":
            continue
        result_list.append(category_name)
    return result_list

# get_networks(data_dir, category) lists the networks in a category given the
# dataset's directory, 'data_dir' and the network category 'category'.
def get_networks(data_dir, category):
    result_list = []
    base_path = os.path.join(data_dir + category)
    for network_name in os.listdir(base_path):
        if network_name == ".DS_Store":
            continue
        result_list.append(network_name)
    return result_list

# get_subnetworks(data_dir, category, network) lists the subnetworks of a
# network given the dataset's directory, 'data_dir', the network category
# 'category', and the network name 'network'.
def get_subnetworks(data_dir, category, network):
    result_list = []
    network_path = os.path.join(data_dir + category, network)
    for subnetwork_name in os.listdir(network_path):
        if subnetwork_name == ".DS_Store":
            continue
        result_list.append(subnetwork_name)
    return result_list

# load_graph(args) loads the preprocessed version of a graph, given an argument
# list 'args' containing the: dataset's directory, the network's: category,
# network, and subnetwork. 
def load_graph(args):
    from graph_tool import load_graph
    data_dir, category, network, subnetwork = args[0], args[1], args[2], args[3]
    pre_processed_file = os.path.join(data_dir + category, network, subnetwork,
                                       "Graph-Data", "preprocessed", subnetwork + ".gt")
    return load_graph(pre_processed_file)

# pre_process([data_dir, category, network, subnetwork])
# preprocesses an empirical network given its descriptors 'args'.
def pre_process(args):
    from graph_tool import load_graph
    from graph_tool.generation import remove_self_loops, remove_parallel_edges
    from graph_tool.topology import extract_largest_component
    # As arguments of the function the directory of the datasets, the network's:
    # category, network, subnetwork information are mentioned. 
    data_dir, category, network, subnetwork = args[0], args[1], args[2], args[3]
    base = os.path.join(data_dir + category, network, subnetwork)
    file = os.path.join(base, "Graph-Data", subnetwork + ".gt")
    pre_processed_base = os.path.join(base, "Graph-Data", "preprocessed")
    pre_processed_file = os.path.join(pre_processed_base, subnetwork + ".gt")

    os.mkdir(pre_processed_base)
    # The preprocessing removes self-loops and parallel edges, finally
    # discarding anything not in the largest connected component.
    g = load_graph(file)
    remove_self_loops(g)
    remove_parallel_edges(g)
    extract_largest_component(g, prune=True).save(pre_processed_file, fmt="gt")
    return (0,) + args


# reset_logger() resets the logger currently at use, to be able to start a new
# logging procedure.  
def reset_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


# set_logger(file_name) re-configures the logger, sets the formatting style, and
# a new logging file, 'file_name'.
def set_logger(file_name):
    from engine.config.config import get_log_dir
    reset_logger()
    logging.basicConfig(filename=get_log_dir() + file_name,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filemode="w",
                        level=logging.INFO)


# log_initial_parameters() creates a logging directory, and logs the initial
# parameter of the analysis there.
def log_initial_parameters():
    if os.path.isdir(get_log_dir()):
        shutil.rmtree(get_log_dir())
    os.mkdir(get_log_dir())
    set_logger("initial_params.log")
    logging.info("num_engines: %s", get_num_engines())
    logging.info("data_dir: %s", get_data_dir())
    logging.info("log_dir: %s", get_log_dir())
    logging.info("num_sampled_random_graphs: %s", get_num_sampled_random_graphs())
    logging.info("vertex_cut_off (lower bound): %s", get_vertex_cut_off())
    logging.info("edge_cut_off (lower bound): %s", get_edge_cut_off())
    logging.info("seed: %s", get_seed())
    reset_logger()

# z_score(val, arr) computes the z-score to describe the relationship of the
# single value, 'val', to the mean of the group of values, 'arr'. 
def z_score(val, arr):
    import numpy as np
    return (np.sqrt(len(arr)) * (val - np.mean(arr))) / (np.sqrt(np.var(arr, ddof=0)))


# compute_z_score(beta) computes for each network from the collection stored in
# get_data_dir() three z-scores when the fraction 0 <= 'beta' <= 1 of the
# vertices are removed from the network. It returns a list of tuples where each
# tuple corresponds to a network from the collection. The first three elements
# of each tuple are z-scores comparing the robustness of an empirical network,
# identified uniquely by the last four elements of the tuple. Each of the three
# z-score values reflects how the robustness of an empirical network compares to
# size-matching random graphs under: static/adaptive targeted attack.
def compute_z_score(beta):
    import pickle
    index = int(beta * 100) - 1
    points = []
    data_dir = get_data_dir()
    for category in get_categories(data_dir):
        for network in get_networks(data_dir, category):
            for subnetwork in get_subnetworks(data_dir, category, network):
                file_dir = data_dir + category + "/" + network + "/" + subnetwork + "/Robustness-Score-Data/scores.pkl"
                point = ()
                with open(file_dir, "rb") as f:
                    scores = pickle.load(f)
                    for removal_strategy in ["static-targeted-attack", "adaptive-targeted-attack",
                                                "random-failure"]:
                        score_main = scores["main"][removal_strategy][index]
                        score_baseline = [scores["baseline"][removal_strategy][i][index] for i in
                                            range(len(scores["baseline"][removal_strategy]))]
                        point = point + (z_score(score_main, score_baseline),)
                points.append(point + (category, network, subnetwork))
    return points
