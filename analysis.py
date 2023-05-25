import argparse
import shutil
import warnings
import numpy as np
import pickle
import ipyparallel as ipp
from engine.utils.io import *
from engine.utils.network import *
from engine.config.config import *

# run_analysis() first preprocesses graphs stored in the datasets folder and
# generates random size-matching graphs. Then, the function generates scores
# using the latter two, and finally postprocesses these scores before storing
# them in the same locations as the analyzed networks. 
def run_analysis():
    # --------------------------------------------------------------------------
    # preprocessing
    # --------------------------------------------------------------------------
    # The array 'nets' is created and filled with tuples indicating the
    # network's: directory, category, network and subnetwork. 
    nets = []
    for category in get_categories(get_data_dir()):
        for network in get_networks(get_data_dir(), category):
            for subnetwork in get_subnetworks(get_data_dir(), category, network):
                nets.append((get_data_dir(), category, network, subnetwork))
    # We preprocess the networks in parallel, and log if any preprocessing step
    # failed. 
    n_engines = get_num_engines()
    cluster = ipp.Cluster(
        n=n_engines,
        controller_ip="*",
        engine_launcher_class="MPI",
        location="server.local",
    )
    cluster.start_cluster_sync()
    client = cluster.connect_client_sync()
    client.wait_for_engines(n=n_engines)
    engines = client.load_balanced_view()
    engines.block = True
    result = engines.map_async(pre_process, nets)
    result.wait_interactive()
    set_logger("preprocessing.log")
    logging.info(
        "The format is: "
        "Category, Network Dataset, Network")
    for args in result:
        if args[0] == 0:
            logging.info("Finished the preprocessing of: %s", args[2:])
        elif args[0] == 1:
            logging.error("Failed in the preprocessing of: %s", args[2:])
    reset_logger()
    # If after preprocessing, the number of vertices and edges are below the
    # pre-defined cut-off values, we exclude them from the analysis.
    updated_nets = []
    for net in nets:
        g = load_graph(net)
        n, m = g.num_vertices(), g.num_edges()
        if n >= get_vertex_cut_off() and m >= get_edge_cut_off():
            updated_nets.append(net + (n, m))
    nets = updated_nets

    # --------------------------------------------------------------------------
    # random graph generation
    # --------------------------------------------------------------------------
    # Setup the random number generator.
    rs = np.random.default_rng(get_seed())
    # We create the array 'args', and for each random network to be generated we
    # add an entry to 'args' containing: the directory to read the corresponding
    # empirical network, the directory to write the generated size-matching
    # random graph, the number of vertices (n) in the random graph, the number
    # of edges (m) in the random graph, and the random seed. 
    args = []
    for (data_dir, category, network, subnetwork, n, m) in nets:
        random_net_dir = os.path.join(data_dir + category, network, subnetwork, "Graph-Data", "random-nets/")
        os.mkdir(random_net_dir)
        args.extend(
            [(data_dir, random_net_dir, n, m, rs.integers(low=0, high=np.iinfo(np.int64).max)) for _ in
             range(get_num_sampled_random_graphs())])
    # We generate the random networks in parallel, and log if any random network
    # generation step failed. 
    result = engines.map_async(fast_gnm, args)
    result.wait_interactive()
    set_logger("random_network_generation.log")
    logging.info(
        "The format is: "
        "Category, Network Dataset, Network, "
        "Number of Vertices (of the random graph), Number of Edges (of the random graph), Seed (for randomization)")
    for args in result:
        if args[0] == 0:
            logging.info("Generated a random network with the following parameters: %s", args[1:])

        elif args[0] == 1:
            logging.error("Failed to generate a random network with the following parameters: %s", args[1:])
    reset_logger()
    # If for an empirical network the required number of size-matching random networks could not be generated, we discard it from analysis.
    updated_nets = []
    for net in nets:
        data_dir, category, network, subnetwork, n, m = net
        random_net_dir = os.path.join(get_data_dir() + category, network, subnetwork, "Graph-Data", "random-nets/")
        if len(os.listdir(random_net_dir)) == get_num_sampled_random_graphs():
            updated_nets.append(net)
    nets = updated_nets

    # --------------------------------------------------------------------------
    # score generation
    # --------------------------------------------------------------------------
    # Setup the random number generator.
    rs = np.random.default_rng(get_seed())
    # We create an array 'args' which contains for each empirical and randomly
    # generated network: the directory to read it from (the preprocessed
    # version), the directories to write the robustness scores against different
    # vertex removal strategies to, the random seed, the file name used for saving
    # the robustness score, and the directory where all the datasets are saved. 
    args = []
    for (data_dir, category, network, subnetwork, n, m) in nets:
        base = os.path.join(data_dir + category, network, subnetwork) + "/"
        robustness_score_dirs = [base + "Robustness-Score-Data/" + "static-targeted-attack/",
                                 base + "Robustness-Score-Data/" + "adaptive-targeted-attack/",
                                 base + "Robustness-Score-Data/" + "random-failure/"]
        for robustness_score_dir in robustness_score_dirs:
            os.mkdir(robustness_score_dir)
        random_net_dir = base + "Graph-Data/" + "random-nets/"
        args.extend([(os.path.join(random_net_dir, path), robustness_score_dirs,
                      rs.integers(low=0, high=np.iinfo(np.int64).max), i + 1, data_dir)
                     for i, path in
                     enumerate(os.listdir(random_net_dir))])
        pre_processed_file = base + "Graph-Data/preprocessed/" + subnetwork + ".gt"
        args.append((pre_processed_file, robustness_score_dirs, rs.integers(low=0, high=np.iinfo(np.int64).max), 0,
                     data_dir))
    # We compute the robustness score for all the empirical networks and
    # randomly generated networks, in parallel. If at any point the computation
    # of the robustness score produces and error we log it. 
    result = engines.map_async(compute_robustness_score, args)
    result.wait_interactive()
    set_logger("compute_robustness_score.log")
    logging.info(
        "The format is: "
        """Category, Network Dataset, Network, Seed (for
        randomization/tie-breaking), Index (0 corresponds to the original
        network, >0 corresponds to the index in the size-matching random graph baseline)""")
    for args in result:
        if args[0] == 0:
            logging.info("Computed the score with the following parameters: %s", args[1:])
        elif args[0] == 1:
            logging.error("Failed to compute the score with the following parameters: %s", args[1:])
    reset_logger()
    # For each empirical network we remove the directory containing the
    # corresponding size-matching graphs. We also remove for each empirical
    # network the corresponding directories where the robustness scores against
    # different vertex removal strategies are saved. However, we combine all the
    # information regarding robustness of the empirical networks and the
    # size-matching random graphs in a "scores.pkl" pickle file. 
    for (data_dir, category, network, subnetwork, n, m) in nets:
        base = os.path.join(data_dir + category, network, subnetwork) + "/"

        shutil.rmtree(base + "Graph-Data/" + "random-nets/")

        robustness_score_dirs = [base + "Robustness-Score-Data/" + "static-targeted-attack/",
                                 base + "Robustness-Score-Data/" + "adaptive-targeted-attack/",
                                 base + "Robustness-Score-Data/" + "random-failure/"]

        res = {"main": {},
               "baseline": {"static-targeted-attack": np.empty(shape=(get_num_sampled_random_graphs(), 100),
                                                               dtype=float),
                            "adaptive-targeted-attack": np.empty(shape=(get_num_sampled_random_graphs(), 100),
                                                                 dtype=float),
                            "random-failure": np.empty(shape=(get_num_sampled_random_graphs(), 100),
                                                       dtype=float)}}
        res["main"]["static-targeted-attack"] = np.load(robustness_score_dirs[0] + "0.npy")
        res["main"]["adaptive-targeted-attack"] = np.load(robustness_score_dirs[1] + "0.npy")
        res["main"]["random-failure"] = np.load(robustness_score_dirs[2] + "0.npy")
        for i in range(1, get_num_sampled_random_graphs() + 1):
            res["baseline"]["static-targeted-attack"][i - 1] = np.load(
                robustness_score_dirs[0] + str(i) + ".npy")
            res["baseline"]["adaptive-targeted-attack"][i - 1] = np.load(
                robustness_score_dirs[1] + str(i) + ".npy")
            res["baseline"]["random-failure"][i - 1] = np.load(robustness_score_dirs[2] + str(i) + ".npy")
        shutil.rmtree(robustness_score_dirs[0])
        shutil.rmtree(robustness_score_dirs[1])
        shutil.rmtree(robustness_score_dirs[2])
        with open(base + "Robustness-Score-Data/" + "scores.pkl", "wb") as scores_file:
            pickle.dump(res, scores_file)

    # --------------------------------------------------------------------------------
    # postprocessing
    # --------------------------------------------------------------------------------
    # We turn off logging.
    logging.shutdown()
    cluster.stop_cluster_sync()
    # This function recursively deletes empty directories, in a bottom-up fashion.
    def remove_empty_folders(path):
        # Function to remove empty folders.
        if not os.path.isdir(path):
            return
        # Remove empty sub-folders.
        files = os.listdir(path)
        if len(files):
            for f in files:
                fullpath = os.path.join(path, f)
                if os.path.isdir(fullpath):
                    remove_empty_folders(fullpath)
        # If the folder is empty, then we delete it.
        files = os.listdir(path)
        if len(files) == 0:
            os.rmdir(path)

    # If any empirical network is discarded during analysis we remove the
    # corresponding directory, and then remove empty parent directories
    # recursively. After the above step, the datasets folder contains only the
    # empirical networks for which the robustness and scale-freeness analysis is
    # complete and their scores.
    nets = set([(x[0], x[1], x[2], x[3]) for x in nets])
    for category in get_categories(get_data_dir()):
        for network in get_networks(get_data_dir(), category):
            for subnetwork in get_subnetworks(get_data_dir(), category, network):
                if not (get_data_dir(), category, network, subnetwork) in nets:
                    shutil.rmtree(
                        os.path.join(get_data_dir(), category, network, subnetwork) + "/")
    remove_empty_folders(get_data_dir())
    shutil.move(get_data_dir(), get_permanent_dir() + "datasets/")
    shutil.move(get_log_dir(), get_permanent_dir() + "logs/")


# argument_checker(x) verifies that the user input specifying the amount of
# cores to use for this script is valid and raises an error if it is not.
def argument_checker(x):
    num = int(x)
    if num < 0 and num != -1:
        raise argparse.ArgumentTypeError('invalid value!')
    else:
        return num



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=argument_checker, required=True)
    cli_input = parser.parse_args()
    set_num_engines(cli_input.cores)
    # set_seed(init_seed) sets the meta-seed for the randomness in the analysis.
    set_seed(0)
    # set_cut_off(min_num_vertices, min_num_edges) sets the minimum number of
    # vertices and edges required for each empirical network after preprocessing
    # to qualify for analysis.
    set_cut_off(1000, 1000)
    # set_num_sampled_random_graphs(number_of_sampled_random_graphs) sets the
    # number of size-matching random networks compared to each empirical network
    # to evaluate its relative robustness.
    set_num_sampled_random_graphs(10)
    # set_working_dir(working_dir_path) sets the working directory where the
    # analysis is performed, and the corresponding results are temporarily
    # saved. For example:
    # set_working_dir("/cluster/project/gess/coss/users/shashemi/")
    set_working_dir(os.getcwd() + "/")
    # set_permanent_dir(permanent_dir_path) sets the permanent directory where
    # the final results are saved.
    # For example: set_permanent_dir("/cluster/scratch/shashemi/results/")
    set_permanent_dir(os.getcwd() + "/")
    # log_initial_parameters() logs all the above initial parameters for the
    # analysis so that the results can be replicated.
    log_initial_parameters()
    # The following runs the analysis.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        run_analysis()
