import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import numpy as np


def make_microbe_table(microbe,spc_met_networks):

    """
    Creates a pandas dataframe with network information for a microbe in the set of networks.

    :param microbe: Name of the microbe
    :type microbe: str
    :param spc_met_networks: The set of species-metabolite networks as produced by MetConSIN
    :type spc_met_networks: dict

    :return: Table of network information for each network (each time interval). Contains the following:

    - TotalOutCoefficients: Sum of all weights of outgoing edges.
    - TotalProductionCoefficients: Sum of all weights of positive outgoing edges.
    - TotalConsumptionCoefficients: Sum of all weights of negative outgoing edges.
    - LimitingMetabolites: Metabolites that rate-limit the growth of the microbe, separated by ".".
    - GrowthCoefficient_Metabolite: The coeffecient (edge weight) in the growth equation term corresponding the rate limiting Metabolite.
    - IntrinsicGrowth: Constant (w.r.t. metabolites) term in microbe growth
    - Produces: Name of all metabolites with positive edge from microbe, separated by ".".
    - ProductionCofactors: Cofactors of all positive edges with microbe as source, separated by ".".
    - Consumes: Name of all metabolites with negative edge from microbe, separated by ".".
    - ConsumptionCofactors: Cofactors of all negative edges with microbe as source, separated by ".".
    - BipartiteClusteringCoefficient: Clustering coefficient of microbe node in the bipartite grapth.
    - TimeRange: Time range of network (same as column header)
    
    :rtype: pandas dataframe
    """

    network_keys = list(spc_met_networks.keys())
    try:
        network_keys.remove("Average")
        ky = "Average"
        combined_info = pd.Series()

        netx_graph = nx.from_pandas_edgelist(spc_met_networks[ky]['edges'],source = 'Source',target = 'Target',edge_attr = ["SourceType","Weight"],create_using=nx.DiGraph)
        edge_out_info = spc_met_networks[ky]['edges'][spc_met_networks[ky]['edges']["Source"] == microbe]
        edge_in_info = spc_met_networks[ky]['edges'][spc_met_networks[ky]['edges']["Target"] == microbe]
        node_info = spc_met_networks[ky]['nodes'].loc[microbe]
        combined_info.loc["TotalOutCoefficients"] = edge_out_info["Weight"].sum()
        combined_info.loc["TotalProductionCoefficients"] = edge_out_info[edge_out_info["Weight"]>0]["Weight"].sum()
        combined_info.loc["TotalConsumptionCoefficients"] = edge_out_info[edge_out_info["Weight"]<0]["Weight"].sum()
        combined_info.loc["LimitingMetabolites"] = ".".join(edge_in_info["Source"].values)
        for met in edge_in_info["Source"].values:
            #.iloc[0] is fine because there should only be one...
            combined_info.loc["GrowthCoefficient_{}".format(met)] = edge_in_info[edge_in_info["Source"] == met]["Weight"].iloc[0]
        combined_info.loc["Produces"] = ".".join(np.unique(edge_out_info[edge_out_info["Weight"] > 0]["Target"].values))
        combined_info.loc["ProductionCofactors"] = ".".join(np.unique(edge_out_info[edge_out_info["Weight"] > 0]["Cofactor"].values))
        combined_info.loc["Consumes"] = ".".join(np.unique(edge_out_info[edge_out_info["Weight"] < 0]["Target"].values))
        combined_info.loc["ConsumptionCofactors"] = ".".join(np.unique(edge_out_info[edge_out_info["Weight"] < 0]["Cofactor"].values))
        combined_info.loc["BipartiteClusteringCoefficient"] = bipartite.clustering(netx_graph).get(microbe,0)


    except:
        combined_info = None
        pass
    microbe_table = pd.DataFrame(columns = network_keys)
    for ky in network_keys:     
        if ky not in ["Average","Difference","Combined"]:
            netx_graph = nx.from_pandas_edgelist(spc_met_networks[ky]['edges'],source = 'Source',target = 'Target',edge_attr = ["SourceType","Weight"],create_using=nx.DiGraph)
            edge_out_info = spc_met_networks[ky]['edges'][spc_met_networks[ky]['edges']["Source"] == microbe]
            edge_in_info = spc_met_networks[ky]['edges'][spc_met_networks[ky]['edges']["Target"] == microbe]
            node_info = spc_met_networks[ky]['nodes'].loc[microbe]
            microbe_table.loc["TotalOutCoefficients",ky] = edge_out_info["Weight"].sum()
            microbe_table.loc["TotalProductionCoefficients",ky] = edge_out_info[edge_out_info["Weight"]>0]["Weight"].sum()
            microbe_table.loc["TotalConsumptionCoefficients",ky] = edge_out_info[edge_out_info["Weight"]<0]["Weight"].sum()
            microbe_table.loc["LimitingMetabolites",ky] = ".".join(edge_in_info["Source"].values)
            for met in edge_in_info["Source"].values:
                #.iloc[0] is fine because there should only be one...
                microbe_table.loc["GrowthCoefficient_{}".format(met),ky] = edge_in_info[edge_in_info["Source"] == met]["Weight"].iloc[0]
            microbe_table.loc["IntrinsicGrowth",ky] = node_info["IntrinsicGrowth"]
            microbe_table.loc["Produces",ky] = ".".join(np.unique(edge_out_info[edge_out_info["Weight"] > 0]["Target"].values))
            microbe_table.loc["ProductionCofactors",ky] = ".".join(np.unique(edge_out_info[edge_out_info["Weight"] > 0]["Cofactor"].values))
            microbe_table.loc["Consumes",ky] = ".".join(np.unique(edge_out_info[edge_out_info["Weight"] < 0]["Target"].values))
            microbe_table.loc["ConsumptionCofactors",ky] = ".".join(np.unique(edge_out_info[edge_out_info["Weight"] < 0]["Cofactor"].values))
            microbe_table.loc["BipartiteClusteringCoefficient",ky] = bipartite.clustering(netx_graph).get(microbe,0)
            microbe_table.loc["TimeRange",ky] = ky
    return microbe_table.fillna(0),combined_info

def make_microbe_growthlimiter(microbe,spc_met_networks):

    """Creates table of growth limiting metabolites for the microbe, suitable for plotting with seaborn.

    :param microbe: Name of microbe
    :type microbe: str
    :param spc_met_networks: The set of species-metabolite networks as produced by MetConSIN
    :type spc_met_networks: dict

    :return: Table of growth limiting metabolites for each time interval with coefficient and time-range
    :rtype: pandas dataframe
    """


    network_keys = list(spc_met_networks.keys())
    try:
        network_keys.remove("Combined")
    except:
        pass
    try:
        network_keys.remove("Average")
    except:
        pass
    try:
        network_keys.remove("Difference")
    except:
        pass
    microbe_limiter_table = pd.DataFrame(columns = ["Metabolite","Coefficient","TimeRange"])
    for tm in network_keys:
        edge_in_info = spc_met_networks[tm]['edges'][spc_met_networks[tm]['edges']["Target"] == microbe]
        mets = edge_in_info["Source"].values
        for m in mets:
            #.iloc[0] is fine because there should only be one...
            growth_coeff = edge_in_info[edge_in_info["Source"] == m]["Weight"].iloc[0]
            microbe_limiter_table.loc["{}.{}".format(tm,m)] = [m,growth_coeff,tm]
    return microbe_limiter_table

def make_limiter_table(met,spc_met_networks,models):

    """Generates a table of growth coefficients for each model for a given metabolite (will be 0 if this is never a growth limiting metabolite.)

    :param met: Name of metabolite
    :type met: str
    :param spc_met_networks: The set of species-metabolite networks as produced by MetConSIN
    :type spc_met_networks: dict
    :param models: Names of all the microbes in the set of networks
    :type models: list[str]

    :return: Table indexed by time range with models as column headers containing limiting growth coefficients.
    :rtype: pandas dataframe
    """


    network_keys = list(spc_met_networks.keys())
    try:
        network_keys.remove("Combined")
    except:
        pass
    try:
        network_keys.remove("Difference")
    except:
        pass
    try:
        avg_lim = pd.Series(index = models)
        comb_edges_out = spc_met_networks["Average"]["edges"][spc_met_networks["Average"]["edges"]["Source"] == met]
        for mod in comb_edges_out["Target"]:
            avg_lim.loc[mod] = out_edges[out_edges["Target"] == mod]["Weight"].iloc[0]
        network_keys.remove("Average")
    except:
        avg_lim = None
        pass
    limiter_table = pd.DataFrame(columns = models,index = network_keys)
    for ky in network_keys:
        out_edges = spc_met_networks[ky]["edges"][spc_met_networks[ky]["edges"]["Source"] == met]
        for mod in out_edges["Target"]:
            limiter_table.loc[ky,mod] = out_edges[out_edges["Target"] == mod]["Weight"].iloc[0]
    return limiter_table.fillna(0),avg_lim

def make_limiter_plot(met,spc_met_networks):

    """Creates a longform table with growth limiting coefficients for any model/timerange pair for which the metabolite is growth limiting.

    :param met: Name of metabolite
    :type met: str
    :param spc_met_networks: The set of species-metabolite networks as produced by MetConSIN
    :type spc_met_networks: dict

    :return: Table of microbes that the metabolite growth-limits for each time range, with coefficient and time range
    :rtype: pandas dataframe
    """

    network_keys = list(spc_met_networks.keys())
    try:
        network_keys.remove("Combined")
    except:
        pass
    try:
        network_keys.remove("Average")
    except:
        pass
    try:
        network_keys.remove("Difference")
    except:
        pass
    limiter_plot = pd.DataFrame(columns = ["Model","Coefficient","TimeRange"])
    for ky in network_keys:
        out_edges = spc_met_networks[ky]["edges"][spc_met_networks[ky]["edges"]["Source"] == met]
        for mod in out_edges["Target"]:
            coeff = out_edges[out_edges["Target"] == mod]["Weight"].iloc[0]
            limiter_plot.loc["{}.{}".format(mod,ky)] = [mod,coeff,ky]
    return limiter_plot


def get_mm_indegree(met_met,nd):

    """Function to inspect the **in** degree of a node in the metabolite-metabolite network.

    :param met_met: Metabolite-metabolite interaction network
    :type met_met: DataFrame
    :param nd: Name of the node (i.e. metabolite name)
    :type nd: str

    :return: number of edges into node, sum of edge weights into node, sum of abs. value of edge weights into node, sum of positive edge weights into node, sum of abs. value of negative edge weights into node
    :rtype: dict
    """

    edges_in = met_met[met_met["Target"] == nd]
    ret = {}
    ret["NumberEdges"] = len(edges_in)
    ret["SumWeight"] = edges_in["Weight"].sum()
    ret["SumAbsWeight"] = edges_in["ABS_Weight"].sum()
    ret["PositiveSumWeight"] = edges_in[edges_in["Weight"] > 0]["Weight"].sum()
    ret["NegativeSumWeight"] = -edges_in[edges_in["Weight"] < 0]["Weight"].sum()
    return ret


def get_mm_outdegree(met_met,nd):

    """Function to inspect the **out** degree of a node in the metabolite-metabolite network.

    :param met_met: Metabolite-metabolite interaction network
    :type met_met: DataFrame
    :param nd: Name of the node (i.e. metabolite name)
    :type nd: str

    :return: number of edges out of node, sum of edge weights out of node, sum of abs. value of edge weights out of node, sum of positive edge weights out of node, sum of abs. value of negative edge weights inout ofto node
    :rtype: dict
    """


    edges_out = met_met[met_met["Source"] == nd]
    ret = {}
    ret["NumberEdges"] = len(edges_out)
    ret["SumWeight"] = edges_out["Weight"].sum()
    ret["SumAbsWeight"] = edges_out["ABS_Weight"].sum()
    ret["PositiveSumWeight"] = edges_out[edges_out["Weight"] > 0]["Weight"].sum()
    ret["NegativeSumWeight"] = -edges_out[edges_out["Weight"] < 0]["Weight"].sum()
    return ret

def node_in_stat_distribution(metabolite_list,network_set):

    """Function to inspect average and variance of **in** degree for all nodes in the metabolite-metabolite networks.

    :param metabolite_list: list of all the node names in the networks
    :type metabolite_list: list[str]
    :param network_set: set of metabolite-metabolite networks
    :type network_set: dict[DataFrame]

    :return: tuple of tables: Average degrees, variance in degrees, and boolean table indicating nodes that are always unconnected in every network
    :rtype: tuple[DataFrame]
    """

    netkeys = list(network_set.keys())
    try:
        netkeys.remove("Combined")
    except:
        pass
    try:
        netkeys.remove("Average")
    except:
        pass
    try:
        netkeys.remove("Difference")
    except:
        pass
    avg_in_degrees = pd.DataFrame(index = metabolite_list, columns = ["NumberEdges","SumWeight","SumAbsWeight","PositiveSumWeight","NegativeSumWeight"])
    var_in_degrees = pd.DataFrame(index = metabolite_list, columns = ["NumberEdges","SumWeight","SumAbsWeight","PositiveSumWeight","NegativeSumWeight"])
    for met in metabolite_list:
        node_stats = np.empty((5,len(netkeys)))
        for i,ky in enumerate(netkeys):
            deg = get_mm_indegree(network_set[ky]['edges'],met)
            node_stats[0,i] = deg["NumberEdges"]
            node_stats[1,i] = deg["SumWeight"]
            node_stats[2,i] = deg["SumAbsWeight"]
            node_stats[3,i] = deg["PositiveSumWeight"]
            node_stats[4,i] = deg["NegativeSumWeight"]
        avg_in_degrees.loc[met] = np.mean(node_stats,axis = 1)
        var_in_degrees.loc[met] = np.var(node_stats,axis = 1)
    nonzero = avg_in_degrees["NumberEdges"] > 0
    zeros = avg_in_degrees["NumberEdges"] == 0
    avg_in_degrees = avg_in_degrees[nonzero]
    var_in_degrees = var_in_degrees[nonzero]

    return avg_in_degrees,var_in_degrees,zeros

def node_out_stat_distribution(metabolite_list,network_set):

    """Function to inspect average and variance of **out** degree for all nodes in the metabolite-metabolite networks.

    :param metabolite_list: list of all the node names in the networks
    :type metabolite_list: list[str]
    :param network_set: set of metabolite-metabolite networks
    :type network_set: dict[DataFrame]

    :return: tuple of tables: Average degrees, variance in degrees, and boolean table indicating nodes that are always unconnected in every network
    :rtype: tuple[DataFrame]
    """

    netkeys = list(network_set.keys())
    try:
        netkeys.remove("Combined")
    except:
        pass
    try:
        netkeys.remove("Average")
    except:
        pass
    try:
        netkeys.remove("Difference")
    except:
        pass
    avg_out_degrees = pd.DataFrame(index = metabolite_list, columns = ["NumberEdges","SumWeight","SumAbsWeight","PositiveSumWeight","NegativeSumWeight"])
    var_out_degrees = pd.DataFrame(index = metabolite_list, columns = ["NumberEdges","SumWeight","SumAbsWeight","PositiveSumWeight","NegativeSumWeight"])
    for met in metabolite_list:
        node_stats = np.empty((5,len(netkeys)))
        for i,ky in enumerate(netkeys):
            deg = get_mm_outdegree(network_set[ky]['edges'],met)
            node_stats[0,i] = deg["NumberEdges"]
            node_stats[1,i] = deg["SumWeight"]
            node_stats[2,i] = deg["SumAbsWeight"]
            node_stats[3,i] = deg["PositiveSumWeight"]
            node_stats[4,i] = deg["NegativeSumWeight"]
        avg_out_degrees.loc[met] = np.mean(node_stats,axis = 1)
        var_out_degrees.loc[met] = np.var(node_stats,axis = 1)
    nonzero = avg_out_degrees["NumberEdges"] > 0
    zeros = avg_out_degrees["NumberEdges"] == 0
    avg_out_degrees = avg_out_degrees[nonzero]
    var_out_degrees = var_out_degrees[nonzero]

    return avg_out_degrees,var_out_degrees,zeros