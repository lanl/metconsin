import numpy as np
# from scipy.sparse import dok_matrix
import pandas as pd
import time
import scipy as sp



###### Build a species-metabolite interaction network (that's not hard)
#### Then build a species-species network (that's fuzzier or harder)

def species_metabolite_network(metlist,metcons,community,report_activity = True,flobj = None):

    """
    Constructs microbe-metabolite and metabolite-metabolite networks from a set of SurfMod models (with forward-simulation bases). The microbe-metabolite network will have duplicate edges with
    different mediating metabolites (see :ref:`metconsin`)

    :param metlist: list of metabolites in the environment
    :type metlist: list[str]
    :param metcons: metabolite concentrations at the start of the interval on which the bases are valid
    :type metcons: array[float]
    :param community: SurfMod models in the microbial community. These should have bases defined.
    :type community: list[SurfMod] or dict[str,SurfMod]
    :param report_activity: Whether or not to log progress
    :type report_activity: bool
    :param flobj: File object to write logging to. If None, writes to stdout. Default None
    :type flobj: File

    :return: Node table for microbe-metabolite network, Edge table for microbe-metabolite network, Edge table for metabolite-metabolite network, Node table for metabolite-metabolite network
    :rtype: tuple[pandas dataframe]
    """


    #let community be dict or listlike - going to use as listlike
    if isinstance(community,dict):
        models = list(community.values())
    else:
        models = community

    start_time = time.time()
    node_table = pd.DataFrame(np.array([[model.Name for model in models] + list(metlist),["Microbe"]*len(models) + ["Metabolite"]*len(metlist)]).T,columns = ["Name","Type"],index = [model.Name for model in models] + list(metlist))

    met_med_net = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","Cofactor","Match","ABS_Weight","Sign_Weight","ABSRootWeight","SignedRootWeight"])

    met_met_edges = pd.DataFrame(columns = ["Source","Target","Microbe","Weight","ABS_Weight","Sign_Weight","ABSRootWeight","SignedRootWeight"])
    met_met_nodes = pd.DataFrame(columns = [model.Name for model in models],index=metlist)
    microbes_exchanging = dict([(met,[]) for met in metlist])

    if report_activity:
        try:
            flobj.write("[species_metabolite_network] Building network\n")
        except:
            print("[species_metabolite_network] Building network")

    intr_growths = {}

    for model in models:

        Q,R,beta = model.current_basis
        #beta is (basis rows,basis columns)
        # so v_i = (B^-1 y[beta[0]])_j:i=beta[1]_j for i in beta[1], 0 otherwise.
        # and a\cdot v = a[beta[1]]^T B^-1 y[beta[0]]
        # and Av = A[:,beta[1]]B^-1 y[beta[0]]

        if len(beta[0]):
            #compute (gamma.T B^(-1)).T = (gamma.T R^(-1)Q^(-1)).T = (gamma.T R^(-1)Q.T) = QR.T^(-1)gamma
            growth_vec = np.dot(Q,sp.linalg.solve_triangular(R.T,-model.objective[beta[1]],lower = True))

            #compute -GammaStarB^(-1) = -GammaStar R^(-1) Q^(-1) = -GammaStar R^(-1) Q.T
            usage_matrix = -np.dot(model.GammaStar[:,beta[1]],sp.linalg.solve_triangular(R,Q.T))
        
        else:
            growth_vec = np.empty( shape=(0) )
            usage_matrix = np.empty( shape=(len(model.exchanged_metabolites), 0) )

        num_exch_met = len(model.exchanged_metabolites)

        

        #form the bound vector the LP so we can compute the constant growth of the microbe
        metabolite_con = metcons[model.ExchangeOrder]
        exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])
        bound_rhs = np.concatenate([exchg_bds,model.internal_bounds])

        #if lower bounds are constant, assume that upper bounds are not and exclude from intrinsic growth
        if model.lower_exch_type.lower() == 'constant':
            internal_basic = np.where(beta[0] >= num_exch_met)
        elif model.upper_exch_type.lower() == 'constant': #if upper bounds are constant, then lower bounds cannot be (otherwise the problem is trivial). So exclude lower bounds from intrinsic growth
            internal_basic = np.where(~((num_exch_met<np.array(beta[0]))*(np.array(beta[0])<2*num_exch_met)))
        else: #we can also include both bounds on exchange if both are dynamic
            internal_basic = np.where(beta[0] >= 2*num_exch_met)

        intrinsic_growth = np.dot(growth_vec[internal_basic],bound_rhs[beta[0]][internal_basic])
        intr_growths[model.Name] = intrinsic_growth

        #Next we compute the edges y_j -> x_i. These correspond to non-zero entries l of growth_vec such that beta[0][l] = j (j = 1,...,number of metabolites) if model.upper_exch_type != Constant
        # PLUS beta[0][l] = j+ number of metabolites if model.lower_exch_type != Constant

        b0ubmsk = np.array(beta[0])<num_exch_met
        b0ub = np.array(beta[0])[b0ubmsk]
        b0lbmsk = (num_exch_met<np.array(beta[0]))*(np.array(beta[0])<2*num_exch_met)
        b0lb = np.array(beta[0])[b0lbmsk] - num_exch_met

        y_to_x_weights = np.zeros(num_exch_met)
        #
        if model.upper_exch_type.lower() != 'constant':#if upper bounds are dynamic, add in their effect
            y_to_x_weights[b0ub] += growth_vec[b0ubmsk]
        if model.lower_exch_type.lower() != 'constant':#same for lower bounds
            y_to_x_weights[b0lb] += growth_vec[b0lbmsk]
        
        met_met_nodes.loc[model.exchanged_metabolites,model.Name] = y_to_x_weights

        ##Now we can turn that into a list of edges

        impacting_mets = np.array(model.exchanged_metabolites)[y_to_x_weights.round(7).astype(bool)]#i think its actually unnecessary to convert to bools
        full_df_addition = pd.DataFrame(columns = met_med_net.columns)
        full_df_addition["Source"] = impacting_mets
        full_df_addition["Target"] = [model.Name]*len(impacting_mets)
        full_df_addition["SourceType"] = ["Metabolite"]*len(impacting_mets)
        full_df_addition["Weight"] = y_to_x_weights[y_to_x_weights.round(7).astype(bool)]
        full_df_addition["Cofactor"] = ["None"]*len(impacting_mets)
        full_df_addition["Match"] = [-1]*len(impacting_mets)
        full_df_addition["ABS_Weight"] = np.abs(y_to_x_weights[y_to_x_weights.round(7).astype(bool)])
        full_df_addition["Sign_Weight"] = np.sign(y_to_x_weights[y_to_x_weights.round(7).astype(bool)])
        full_df_addition["ABSRootWeight"] = np.sqrt(np.abs(y_to_x_weights[y_to_x_weights.round(7).astype(bool)]))
        full_df_addition["SignedRootWeight"] = np.sqrt(np.abs(y_to_x_weights[y_to_x_weights.round(7).astype(bool)]))*np.sign(y_to_x_weights[y_to_x_weights.round(7).astype(bool)])
        ##


        met_med_net = pd.concat([met_med_net,full_df_addition],ignore_index=True)

        #####
        #####
        ##      Next we compute the x->y edges and the y->y edges.
        ####

        for j in range(num_exch_met):



            
            metnm = model.exchanged_metabolites[j]

            if j in beta[0]:
                microbes_exchanging[metnm] += [model.Name]


            ode_row = usage_matrix[j]

            #if lower bounds are constant, assume that upper bounds are not and exclude from constant effect
            if model.lower_exch_type.lower() == 'constant':
                internal_basic = np.where(beta[0] >= num_exch_met)
            elif model.upper_exch_type.lower() == 'constant': #if upper bounds are constant, then lower bounds cannot be (otherwise the problem is trivial). So exclude lower bounds from constant effect
                internal_basic = np.where(~((num_exch_met<np.array(beta[0]))*(np.array(beta[0])<2*num_exch_met)))
            else: #we can also include both bounds on exchange if both are dynamic
                internal_basic = np.where(beta[0] >= 2*num_exch_met)

            constant_effect = np.dot(ode_row[internal_basic],bound_rhs[beta[0]][internal_basic])
            tmpdf1 = pd.DataFrame(columns = met_med_net.columns)
            tmpdf1.loc[0] = [model.Name,metnm,"Microbe",constant_effect,"None",-1,np.abs(constant_effect),np.sign(constant_effect),np.sqrt(np.abs(constant_effect)),np.sign(constant_effect)*np.sqrt(np.abs(constant_effect))]
            # tmpdf2 = pd.DataFrame(columns = met_med_net_summary.columns)
            # tmpdf2.loc[0] = [model.Name,metnm,"Microbe",constant_effect,np.abs(constant_effect),np.sign(constant_effect),np.sqrt(np.abs(constant_effect)),np.sign(constant_effect)*np.sqrt(np.abs(constant_effect))]

            b0ubmsk = np.array(beta[0])<num_exch_met
            b0ub = np.array(beta[0])[b0ubmsk]
            b0lbmsk = (num_exch_met<np.array(beta[0]))*(np.array(beta[0])<2*num_exch_met)
            b0lb = np.array(beta[0])[b0lbmsk] - num_exch_met

            y_to_y_weights = np.zeros(num_exch_met)
            #
            if model.upper_exch_type.lower() != 'constant':#if upper bounds are dynamic, add in their effect
                y_to_y_weights[b0ub] += ode_row[b0ubmsk]
            if model.lower_exch_type.lower() != 'constant':#same for lower bounds
                y_to_y_weights[b0lb] += ode_row[b0lbmsk]

            ##Now we can turn that into a list of edges

            impacting_mets = np.array(model.exchanged_metabolites)[y_to_y_weights.round(7).astype(bool)]#i think its actually unnecessary to convert to bools
            full_df_addition = pd.DataFrame(columns = met_med_net.columns)
            # summ_df_addition = pd.DataFrame(columns = met_med_net_summary.columns)
            metmet_df_addition = pd.DataFrame(columns = met_met_edges.columns)
            
            full_df_addition["Source"] = [model.Name]*len(impacting_mets)
            full_df_addition["Target"] = [metnm]*len(impacting_mets)
            full_df_addition["SourceType"] = ["Microbe"]*len(impacting_mets)
            full_df_addition["Weight"] = y_to_y_weights[y_to_y_weights.round(7).astype(bool)]
            full_df_addition["Cofactor"] = impacting_mets
            full_df_addition["Match"] = (impacting_mets == metnm).astype(int)
            full_df_addition["ABS_Weight"] = np.abs(y_to_y_weights[y_to_y_weights.round(7).astype(bool)])
            full_df_addition["Sign_Weight"] = np.sign(y_to_y_weights[y_to_y_weights.round(7).astype(bool)])
            full_df_addition["ABSRootWeight"] = np.sqrt(np.abs(y_to_y_weights[y_to_y_weights.round(7).astype(bool)]))
            full_df_addition["SignedRootWeight"] = np.sqrt(np.abs(y_to_y_weights[y_to_y_weights.round(7).astype(bool)]))*np.sign(y_to_y_weights[y_to_y_weights.round(7).astype(bool)])
            ##

            metmet_df_addition["Source"] = impacting_mets
            metmet_df_addition["Target"] = [metnm]*len(impacting_mets)
            metmet_df_addition["Microbe"] = [model.Name]*len(impacting_mets)
            metmet_df_addition["Weight"] = y_to_y_weights[y_to_y_weights.round(7).astype(bool)]
            metmet_df_addition["ABS_Weight"] = np.abs(y_to_y_weights[y_to_y_weights.round(7).astype(bool)])
            metmet_df_addition["Sign_Weight"] = np.sign(y_to_y_weights[y_to_y_weights.round(7).astype(bool)])
            metmet_df_addition["ABSRootWeight"] = np.sqrt(np.abs(y_to_y_weights[y_to_y_weights.round(7).astype(bool)]))
            metmet_df_addition["SignedRootWeight"] = np.sqrt(np.abs(y_to_y_weights[y_to_y_weights.round(7).astype(bool)]))*np.sign(y_to_y_weights[y_to_y_weights.round(7).astype(bool)])

            met_med_net = pd.concat([met_med_net,full_df_addition],ignore_index=True)
            met_met_edges = pd.concat([met_met_edges,metmet_df_addition],ignore_index = True)


    associated = pd.DataFrame(index = node_table.index,columns = ["In","Out","All"])
    for ndinx in node_table.index:
        nd = node_table.loc[ndinx,"Name"]
        associated.loc[ndinx,"Out"] = ".".join(np.unique(list(met_med_net.loc[met_med_net["Source"] == nd]["Target"])))
        associated.loc[ndinx,"In"] = ".".join(np.unique(list(met_med_net.loc[met_med_net["Target"] == nd]["Source"])))
        associated.loc[ndinx,"All"] = associated.loc[ndinx,"Out"] + associated.loc[ndinx,"In"]#np.concatenate([associated.loc[ndinx,"Out"],associated.loc[ndinx,"In"]])

    associated.fillna("",inplace = True)

    node_table = pd.concat([node_table,associated],axis = 1)
    node_table["IntrinsicGrowth"] = np.zeros(len(node_table))
    for model in models:
        node_table.loc[model.Name,"IntrinsicGrowth"] = intr_growths[model.Name]

    minuts,sec = divmod(time.time() - start_time, 60)

    met_met_nodes.fillna(0,inplace = True)
    met_met_nodes = pd.concat([met_met_nodes,associated.loc[met_met_nodes.index]],axis = 1)

    if report_activity:
        try:
            flobj.write("[species_metabolite_network] Network built in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
        except:
            print("[species_metabolite_network] Network built in ",int(minuts)," minutes, ",sec," seconds.")

    return node_table,met_med_net,met_met_edges,met_met_nodes

def trim_network(edges,nodes,dynamics):

    """

    Trims out nodes in a microbe-metabolite or metabolite-metabolite network by removing nodes that are not present (according to the given dynamics). Removes edges associated with those nodes.

    :param edges: List of edges of the network
    :type edges: pandas dataframe
    :param nodes: List of nodes of the network
    :type nodes: pandas dataframe
    :param dynamics: Dynamic simulation result containing values of node variables
    :type dynamics: pandas dataframe

    :return: Edges of trimmed network, Nodes of trimmed network
    :rtype: tuple[pandas dataframe]

    """


    newnodes = nodes.copy()
    newedges = edges.copy()
    dropped = 0
    messedup = 0
    for nd in dynamics.index:
        if nd in newnodes.index:
            if max(dynamics.loc[nd]) < 10**-6:
                newnodes.drop(index = nd, inplace = True)
                newedges = newedges.loc[(newedges["Source"] != nd) & (newedges["Target"] != nd)]
                if "Cofactor" in newedges.columns:
                    newedges = newedges.loc[newedges["Cofactor"] != nd]
                dropped += 1

    return newedges,newnodes

def make_medmet_summ(medmet_full):

    """
    Makes microbe-metabolite summary network by collapsing all edges between two metabolites (i.e. edges with seperate mediating metabolites) into a single edge.

    :param medmet_full: The full list of network edges
    :type medmet_full: pandas dataframe

    :return: The new list of network edges, with only unique (source,target) pairs
    :rtype: pandas dataframe
    """    
    
    medmet_summ = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","ABS_Weight","Sign_Weight","ABSRootWeight","SignedRootWeight"])
    
    unique_edges = [list(x) for x in set(tuple(x) for x in medmet_full[["Source","Target"]].values)]
    
    for i,v in enumerate(unique_edges):
        subdf = medmet_full[(medmet_full[["Source","Target"]] == v).all(axis = 1)]
        stype = subdf["SourceType"].iloc[0]
        we = subdf["Weight"].sum()
        medmet_summ.loc[i] = v + [stype,we,np.abs(we),np.sign(we),np.sqrt(np.abs(we)),np.sign(we)*np.sqrt(np.abs(we))]
    
    return medmet_summ

def heuristic_ss(metmed,nodes,report_activity = False,flobj = None):
    
    '''

    Creates a microbe-microbe interaction network heuristically using any length-2 paths in the microbe-metabolite network

    :param metmed: list of edges in the microbe-metabolite network
    :type metmed: pandas dataframe
    :param nodes: list of nodes of the microbe-metabolite network
    :type nodes: pandas dataframe
    :param report_activity: Whether or not to log progress
    :type report_activity: bool
    :return: 
    :rtype: tuple[pandas dataframe]
    
    ``metmed`` should have the following columns: 

    - Source
    - Target
    - SourceType
    - Weight
    - ABS_Weight
    - Sign_Weight
    - ABSRootWeight
    - SignedRootWeight
    
    '''

    nodetable = nodes[nodes["Type"] == "Microbe"]
    edge_table = pd.DataFrame(columns = ["Source","Target","Weight","Metabolites","ABSWeight","SignWeight","ABSRootWeight","SignedRootWeight"],dtype = object)
    adjacency = pd.DataFrame(columns = nodetable["Name"],index = nodetable["Name"])
    if len(metmed):
        metmed_woself = metmed[metmed["Source"] != metmed["Target"]]
        selfloops = metmed[metmed["Source"]==metmed["Target"]]
        for tgnd in adjacency.index:
            if report_activity:
                try:
                    flobj.write("[heuristic_ss] Target: {}\n".format(tgnd))
                except:
                    print("[heuristic_ss] Target: {}".format(tgnd))

            #rows will be targets
            allmediators = metmed_woself[metmed_woself["Target"]==tgnd]#list of metabolites that effect it.
            allsrces = metmed_woself[[(tg in allmediators["Source"].values) for tg in metmed_woself["Target"]]]#list of sources for those metabolites
            for srcnd in np.unique(allsrces["Source"].values):
                if srcnd != tgnd:
                    step1s = allsrces[allsrces["Source"]==srcnd]
                    mediators = step1s["Target"].values
                    weight = 0
                    for rw in step1s.index:
                        stp2 = allmediators[allmediators["Source"] == step1s.loc[rw,"Target"]]["Weight"].values
                        weight += step1s.loc[rw,"Weight"]*sum(stp2)
                    adjacency.loc[tgnd,srcnd] = weight
                    if abs(weight)>0:
                        edge_table.loc["{}->{}".format(srcnd,tgnd)] = [srcnd,tgnd,weight,".".join(list(mediators)),abs(weight),np.sign(weight),np.sqrt(abs(weight)),np.sign(weight)*np.sqrt(abs(weight))]
            relslp = selfloops[selfloops["Target"] == tgnd]
            if len(relslp):
                slfweight = np.mean(relslp["Weight"].values)
                adjacency.loc[tgnd,tgnd] = slfweight
                edge_table.loc["{}->{}".format(tgnd,tgnd)] = [tgnd,tgnd,slfweight,"None",abs(slfweight),np.sign(slfweight),np.sqrt(abs(slfweight)),np.sign(slfweight)*np.sqrt(abs(slfweight))]
            else:
                adjacency.loc[tgnd,tgnd] = 0
    return edge_table,nodetable,adjacency


def average_network(networks,interval_times,network_type):

    """
    Creates time-averaged network from sequence of networks and time-interval lengths.

    :param networks: Set of networks to average, given as dicts with "edges" and "nodes" as keys, in a dict keyed by some label that matches the time interval labels
    :type networks: dict[dict[pandas dataframe]]
    :param interval_times: Length of the time intervals corresponding to the networks. Keyed by time interval labels.
    :type interval_times: dict[float]
    :param network_type: type of network (micmet, metmet, or spc)
    :type network_type: str

    :return: Time-averaged network as tuple of edges, nodes, and a flag that indicates if the method failed
    :rtype: tuple[pandas dataframe,pandas dataframe, bool]

    options for ``network_type`` are:

    - micmet: microbe-metabolite network
    - metmet: metabolite-metabolite network
    - spc: microbe-microbe network

    """


    if network_type == "micmet":
        return average_network_micmet(networks,interval_times)
    elif network_type == "metmet":
        return average_network_metmet(networks,interval_times)
    elif network_type == "spc":
        return average_network_spc(networks,interval_times)
    else:
        return None,None,False

def average_network_micmet(networks,interval_times):

    """
    Creates time-averaged network from sequence of microbe-metabolite networks and time-interval lengths.

    :param networks: Set of networks to average, given as dicts with "edges" and "nodes" as keys, in a dict keyed by some label that matches the time interval labels
    :type networks: dict[dict[pandas dataframe]]
    :param interval_times: Length of the time intervals corresponding to the networks. Keyed by time interval labels.
    :type interval_times: dict[float]

    :return: Time-averaged network as tuple of edges, nodes, and a flag that indicates if the method failed
    :rtype: tuple[pandas dataframe,pandas dataframe, bool]
    """

    all_networks = pd.DataFrame(dtype = float)
    for ky in networks.keys():
        edges = networks[ky]["edges"]
        weights = edges["Weight"]
        if "Cofactor" in edges.columns:
            weights.index = ["++".join(edges.loc[rw,["Source","Target","SourceType","Cofactor"]]) for rw in edges.index]
        else:
            weights.index = ["++".join(edges.loc[rw,["Source","Target","SourceType"]]) for rw in edges.index]
        all_networks = pd.concat([all_networks,weights],axis = 1).rename({"Weight":ky},axis = 1)
    all_networks = all_networks.fillna(value = 0)
    avg_network_rw = all_networks.dot([interval_times[col] for col in all_networks.columns])
    var_network_rw_vals = np.dot(((all_networks.values.T - avg_network_rw.values)**2).T,[interval_times[col] for col in all_networks.columns])
    var_network_rw = pd.Series(var_network_rw_vals,index = all_networks.index)
    if "Cofactor" in edges.columns:
        avg_network = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","Variance","Cofactor","ABS_Weight","Sign_Weight","ABSRootWeight","SignedRootWeight"])
    else:
        avg_network = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","Variance","ABS_Weight","Sign_Weight","ABSRootWeight","SignedRootWeight"])
    for rw in avg_network_rw.index:
        we = avg_network_rw[rw]
        varwe = var_network_rw.loc[rw]
        if "Cofactor" in edges.columns:
            source,target,ty,cof = rw.split("++")
            avg_network.loc[rw] = [source,target,ty,we,varwe,cof,np.abs(we),np.sign(we),np.sqrt(np.abs(we)),np.sign(we)*np.sqrt(np.abs(we))]
        else:
            source,target,ty = rw.split("++")
            avg_network.loc[rw] = [source,target,ty,we,varwe,np.abs(we),np.sign(we),np.sqrt(np.abs(we)),np.sign(we)*np.sqrt(np.abs(we))]
    avg_network.index = np.arange(len(avg_network))

    node_table = make_avg_micmet_node_table(avg_network)
    return avg_network,node_table,True

def make_avg_micmet_node_table(avg_edges):

    """
    Creates a node table for the time-averaged microbe-metabolite network

    :param avg_edges: the time-averaged network edges
    :type avg_edges: pandas dataframe

    :return: node table
    :rtype: pandas dataframe
    """

    all_nodes = np.unique(list(avg_edges["Source"]) + list(avg_edges["Target"]))
    node_table = pd.DataFrame(index =all_nodes, columns = ["In","Out","All","Type"])
    for nd in all_nodes:
        ##As Source:
        assrc = avg_edges[avg_edges["Source"] == nd]
        issrc = False
        if len(assrc):
            ndtype = assrc["SourceType"].iloc[0]
            outs = list(assrc["Target"])
            issrc = True
        astrgt = avg_edges[avg_edges["Target"] == nd]
        if len(astrgt):
            if not issrc:
                if astrgt["SourceType"].iloc[0] == "Microbe":
                    ndtype = "Metabolite"
                else:
                    ndtype = "Microbe"
                outs = ''
            ins  = list(astrgt["Source"])
        node_table.loc[nd] = [".".join(ins),".".join(outs),".".join(ins)+".".join(outs),ndtype]
    return node_table

def average_network_metmet(networks,interval_times):

    """
    Creates time-averaged network from sequence of metabolite-metabolite networks and time-interval lengths.

    :param networks: Set of networks to average, given as dicts with "edges" and "nodes" as keys, in a dict keyed by some label that matches the time interval labels
    :type networks: dict[dict[pandas dataframe]]
    :param interval_times: Length of the time intervals corresponding to the networks. Keyed by time interval labels.
    :type interval_times: dict[float]

    :return: Time-averaged network as tuple of edges, nodes, and a flag that indicates if the method failed
    :rtype: tuple[pandas dataframe,pandas dataframe, bool]
    """

    all_networks = pd.DataFrame(dtype = float)
    for ky in networks.keys():
        edges = networks[ky]["edges"]
        weights = edges["Weight"]
        weights.index = ["++".join(edges.loc[rw,["Source","Target","Microbe"]]) for rw in edges.index]
        all_networks = pd.concat([all_networks,weights],axis = 1).rename({"Weight":ky},axis = 1)
    all_networks = all_networks.fillna(value = 0)
    avg_network_rw = all_networks.dot([interval_times[col] for col in all_networks.columns])
    var_network_rw_vals = np.dot(((all_networks.values.T - avg_network_rw.values)**2).T,[interval_times[col] for col in all_networks.columns])
    var_network_rw = pd.Series(var_network_rw_vals,index = all_networks.index)
    avg_network = pd.DataFrame(columns = ["Source","Target","Microbe","Weight","Variance","ABS_Weight","Sign_Weight","ABSRootWeight","SignedRootWeight"])
    for rw in avg_network_rw.index:
        we = avg_network_rw[rw]
        varwe = var_network_rw.loc[rw]
        source,target,mic = rw.split("++")
        avg_network.loc[rw] = [source,target,mic,we,varwe,np.abs(we),np.sign(we),np.sqrt(np.abs(we)),np.sign(we)*np.sqrt(np.abs(we))]
    avg_network.index = np.arange(len(avg_network))

    node_table = make_avg_metmet_node_table(networks,interval_times)
    return avg_network,node_table,True

def make_avg_metmet_node_table(networks,interval_times):

    """
    Creates time-averaged metabolite-metabolite node table from sequence of metabolite-metabolite node tables and time-interval lengths.

    :param networks: Set of networks to average, given as dicts with "edges" and "nodes" as keys, in a dict keyed by some label that matches the time interval labels
    :type networks: dict[dict[pandas dataframe]]
    :param interval_times: Length of the time intervals corresponding to the networks. Keyed by time interval labels.
    :type interval_times: dict[float]

    :return: time-averaged node table
    :rtype: pandas dataframe
    """

    node_tab = pd.DataFrame()

    model_list = [col for col in list(networks.values())[0]['nodes'].columns if col not in ["In","Out","All"]]

    for mod in model_list:
        moddf = pd.DataFrame()
        for ky,val in networks.items():
            tab = val["nodes"]
            moddf = pd.concat([moddf,tab[mod]],axis = 1).rename({mod:ky},axis = 1)
        resdf = pd.DataFrame()
        moddf.fillna(0,inplace = True)
        mavg = moddf.dot([interval_times[col] for col in moddf])
        resdf["Avg_{}".format(mod)] = mavg
        varis = np.dot(((moddf.values.T - mavg.values)**2).T,[interval_times[col] for col in moddf.columns])
        resdf["Var_{}".format(mod)] = varis
        node_tab = pd.concat([node_tab,resdf],axis = 1)

    i_node_tab = pd.DataFrame()
    o_node_tab = pd.DataFrame()
    a_node_tab = pd.DataFrame()

    for ky,val in networks.items():
        tab = val["nodes"]
        i_node_tab = pd.concat([i_node_tab,tab["In"]],axis = 1).rename({"In":ky},axis = 1)
        o_node_tab = pd.concat([o_node_tab,tab["Out"]],axis = 1).rename({"Out":ky},axis = 1)
        a_node_tab = pd.concat([a_node_tab,tab["All"]],axis = 1).rename({"All":ky},axis = 1)

    i_node_tab.fillna("",inplace = True)
    o_node_tab.fillna("",inplace = True)
    a_node_tab.fillna("",inplace = True)

    node_tab["In"] = [".".join(pd.unique(i_node_tab.loc[rw])) for rw in i_node_tab.index]
    node_tab["Out"] = [".".join(pd.unique(o_node_tab.loc[rw])) for rw in o_node_tab.index]
    node_tab["All"] = [".".join(pd.unique(a_node_tab.loc[rw])) for rw in a_node_tab.index]

    return node_tab

def average_network_spc(networks,interval_times):

    """
    Creates time-averaged network from sequence of microbe-microbe networks and time-interval lengths.

    :param networks: Set of networks to average, given as dicts with "edges" and "nodes" as keys, in a dict keyed by some label that matches the time interval labels
    :type networks: dict[dict[pandas dataframe]]
    :param interval_times: Length of the time intervals corresponding to the networks. Keyed by time interval labels.
    :type interval_times: dict[float]

    :return: Time-averaged network as tuple of edges, nodes, and a flag that indicates if the method failed
    :rtype: tuple[pandas dataframe,pandas dataframe, bool]
    """

    all_networks = pd.DataFrame(dtype = float)
    # metabs = pd.DataFrame()
    for ky in networks.keys():
        edges = networks[ky]["edges"]
        weights = edges["Weight"]
        weights.index = ["++".join(edges.loc[rw,["Source","Target","Metabolites"]]) for rw in edges.index]
        # mets = edges["Metabolites"]
        # mets.index = ["++".join(edges.loc[rw,["Source","Target","Metabolites"]]) for rw in edges.index]
        # metabs = pd.concat([metabs,mets],axis = 1).rename({"Metabolites":ky},axis = 1)
        all_networks = pd.concat([all_networks,weights],axis = 1).rename({"Weight":ky},axis = 1)
    all_networks = all_networks.fillna(value = 0)
    # metabs.fillna("",inplace = True)
    avg_network_rw = all_networks.dot([interval_times[col] for col in all_networks.columns])
    var_network_rw_vals = np.dot(((all_networks.values.T - avg_network_rw.values)**2).T,[interval_times[col] for col in all_networks.columns])
    var_network_rw = pd.Series(var_network_rw_vals,index = all_networks.index)
    avg_network = pd.DataFrame(columns = ["Source","Target","Metabolites","Weight","Variance","ABS_Weight","Sign_Weight","ABSRootWeight","SignedRootWeight"])
    for rw in avg_network_rw.index:
        we = avg_network_rw[rw]
        varwe = var_network_rw.loc[rw]
        source,target,metabolites = rw.split("++")
        avg_network.loc[rw] = [source,target,metabolites,we,varwe,np.abs(we),np.sign(we),np.sqrt(np.abs(we)),np.sign(we)*np.sqrt(np.abs(we))]
    avg_network.index = np.arange(len(avg_network))

    node_table = make_avg_spc_node_table(networks,interval_times)

    return avg_network,node_table,True

def make_avg_spc_node_table(networks,interval_times):


    """
    Creates time-averaged microbe-microbe node table from sequence of microbe-microbe node tables and time-interval lengths.

    :param networks: Set of networks to average, given as dicts with "edges" and "nodes" as keys, in a dict keyed by some label that matches the time interval labels
    :type networks: dict[dict[pandas dataframe]]
    :param interval_times: Length of the time intervals corresponding to the networks. Keyed by time interval labels.
    :type interval_times: dict[float]

    :return: time-averaged node table
    :rtype: pandas dataframe
    """

    node_tab = pd.DataFrame()

    igrth = pd.DataFrame()
    for ky,val in networks.items():
        tab = val["nodes"]
        igrth = pd.concat([igrth,tab["IntrinsicGrowth"]],axis = 1).rename({"IntrinsicGrowth":ky},axis = 1)
    resdf = pd.DataFrame()
    igrth.fillna(0,inplace = True)
    mavg = igrth.dot([interval_times[col] for col in igrth])
    resdf["Avg_IntrinsicGrowth"] = mavg
    varis = np.dot(((igrth.values.T - mavg.values)**2).T,[interval_times[col] for col in igrth.columns])
    resdf["Var_IntrinsicGrowth"] = varis
    node_tab = pd.concat([node_tab,resdf],axis = 1)

    i_node_tab = pd.DataFrame()
    o_node_tab = pd.DataFrame()
    a_node_tab = pd.DataFrame()

    for ky,val in networks.items():
        tab = val["nodes"]
        i_node_tab = pd.concat([i_node_tab,tab["In"]],axis = 1).rename({"In":ky},axis = 1)
        o_node_tab = pd.concat([o_node_tab,tab["Out"]],axis = 1).rename({"Out":ky},axis = 1)
        a_node_tab = pd.concat([a_node_tab,tab["All"]],axis = 1).rename({"All":ky},axis = 1)

    i_node_tab.fillna("",inplace = True)
    o_node_tab.fillna("",inplace = True)
    a_node_tab.fillna("",inplace = True)

    node_tab["In"] = [".".join(pd.unique(i_node_tab.loc[rw])) for rw in i_node_tab.index]
    node_tab["Out"] = [".".join(pd.unique(o_node_tab.loc[rw])) for rw in o_node_tab.index]
    node_tab["All"] = [".".join(pd.unique(a_node_tab.loc[rw])) for rw in a_node_tab.index]

    return node_tab

# Consider a dynamical system that can be written as
#
#     dx_i/dt = x_i (g^T M_i h_i(y))
#     dy/dt = -sum(x_i N_i h_i(y))
#
# Then a sufficient condition for the existence of a set of linear conservation laws
# A1 dx/dt + A2 dy/dt = 0
# A1_ji g^T M_i - (A2 N_i)_j = 0
#
# for all i,j

# For us, M_i is the inverse of the basis for LP_i and N_i is Gamma^*_i*M_i
# Rather than compute the inverse, we store Q_i,R_i, the QR decompostion of basis i


# def get_linear_system(QRsdict,QRkys,GammaStars,all_gs):
#     QRs = [QRsdict[ky] for ky in QRkys]
#     #we need to find A1,A2. All are lists over i.
#     num_xi = len(QRs) #m
#     num_vs = [Mi.shape[1] for Mi in GammaStars]#li
#     num_y = GammaStars[0].shape[0]#n
#     num_eqs = num_y*sum(num_vs)
#     eqs = dok_matrix((num_eqs, num_y**2 + num_y*num_xi), dtype=np.float64)#{}
#     for i in range(num_xi):
#         #To find y = g^T M_i, we note that R_i^TQ_i^Ty = g
#         # so let w = solve(R_i^T,g), and then y = Q_iw (b/c Q is unitary)
#         w = np.linalg.solve(QRs[i][1].T,all_gs[i])
#         gTM = np.dot(QRs[i][0],w)
#         ####
#         for k in range(num_vs[i]):
#             kthBinv = np.linalg.solve(QRs[i][1],QRs[i][0][k])
#             KthN = np.dot(GammaStars[i],kthBinv)
#             for j in range(num_y):
#                 eqnum = num_y*sum(num_vs[:i]) + j*num_vs[i] + k
#                 #save the coefficients index so that the first n*m variables are the
#                 #entries of A1, flattened, and next n*n are entries of A2, flattened.
#                 coeffs = {}
#                 #only one entry of A1
#                 if round(gTM[k],9):
#                     eqs[eqnum,j*num_xi + i] = gTM[k]
#                     # coeffs[j*num_xi + i] = gTM[k]
#                 #jth row of A2 has coefficients kth column of GammaStar*B^(-1)
#                 #To find this, we compute the kth column of B^(-1) (above)
#                 for l in np.where(KthN.round(9))[0]:
#                     # coeffs[num_xi*num_y + num_y*j + l] = KthN[l]
#                     eqs[eqnum,num_xi*num_y + num_y*j + l] = KthN[l]
#                 # eqs[eqnum] = coeffs
#     #return a dictionary keyed by equation number (row in linear system), each entry a dict
#     # keyed by variable number (column in linear system) with value equal to coefficient
#     # This should be easy to use with various sparse solvers.
#     return eqs
