from importlib.machinery import all_suffixes
import numpy as np
# from scipy.sparse import dok_matrix
import pandas as pd
import time
import scipy as sp



###### Build a species-metabolite interaction network (that's not hard)
#### Then build a species-species network (that's fuzzier or harder)

def species_metabolite_network(metlist,metcons,community,report_activity = True,flobj = None):

    #let community be dict or listlike - going to use as listlike
    if isinstance(community,dict):
        models = list(community.values())
    else:
        models = community

    start_time = time.time()
    node_table = pd.DataFrame(np.array([[model.Name for model in models] + list(metlist),["Microbe"]*len(models) + ["Metabolite"]*len(metlist)]).T,columns = ["Name","Type"],index = [model.Name for model in models] + list(metlist))

    met_med_net = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","Cofactor","Match","ABS_Weight","Sign_Weight","Distance","ABSRootWeight","SignedRootWeight"])
    met_med_net_summary = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","ABS_Weight","Sign_Weight","Distance","ABSRootWeight","SignedRootWeight"])

    met_met_edges = pd.DataFrame(columns = ["Source","Target","Microbe","Weight","ABS_Weight","Sign_Weight","Distance","ABSRootWeight","SignedRootWeight"])
    met_met_nodes = pd.DataFrame(columns = ["Microbes"]+[model.Name for model in models],index=metlist)
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



        met_met_nodes.loc[:,model.Name] = np.zeros(len(met_met_nodes))

        #form the bound vector the LP
        metabolite_con = metcons[model.ExchangeOrder]
        exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])
        bound_rhs = np.concatenate([exchg_bds,model.internal_bounds])

        internal_basic = np.where(beta[0] >= len(model.exchanged_metabolites))

        intrinsic_growth = np.dot(growth_vec[internal_basic],bound_rhs[beta[0]][internal_basic])
        intr_growths[model.Name] = intrinsic_growth

        for j in range(len(model.exchanged_metabolites)):


            metab = model.exchanged_metabolites[j]

            if report_activity:
                try:
                    flobj.write("[species_metabolite_network] computing {0} connections to {1}\n".format(model.Name,metab))
                except:
                    print("[species_metabolite_network] computing {0} connections to {1}".format(model.Name,metab))



            if j in beta[0]:

                met_on_mic = growth_vec[np.where(beta[0]==j)[0][0]] 
                met_met_nodes.loc[metab,model.Name] = round(met_on_mic,7)
                microbes_exchanging[metab] += [model.Name]



                if round(met_on_mic,7):
                    tmp1 = pd.DataFrame([[metab,model.Name,"Metabolite",met_on_mic,"None",0,abs(met_on_mic),np.sign(met_on_mic),1/np.abs(met_on_mic),np.sqrt(np.abs(met_on_mic)),np.sign(met_on_mic)*np.sqrt(np.abs(met_on_mic))]],columns = met_med_net.columns)
                    tmp2 = pd.DataFrame([[metab,model.Name,"Metabolite",met_on_mic,abs(met_on_mic),np.sign(met_on_mic),1/np.abs(met_on_mic),np.sqrt(np.abs(met_on_mic)),np.sign(met_on_mic)*np.sqrt(np.abs(met_on_mic))]],columns = met_med_net_summary.columns)
                    met_med_net = met_med_net.append(tmp1,ignore_index = True)
                    met_med_net_summary = met_med_net_summary.append(tmp2,ignore_index = True)

            interactions = usage_matrix[j] #row that Impacts j

            # Separate Exchange/ExchangeLB/Internal/Positivity constaints/Equilibrium
            eq = []
            exlb = []
            exub = []
            internal = []
            for cnst in beta[0]:#range(len(bound_rhs)):#
                ii = np.where(beta[0]==cnst)[0][0]
                if cnst < (model.num_exch_rxns):
                    exub += [{"Metabolite":model.exchanged_metabolites[cnst],"Coefficient":interactions[ii],"Constraint_Value":bound_rhs[cnst],"Instant_Impact":interactions[ii]*bound_rhs[cnst]}]
                    if round(interactions[ii],7):
                        tmpmm = pd.DataFrame([[model.exchanged_metabolites[cnst],metab,model.Name,interactions[ii],abs(interactions[ii]),np.sign(interactions[ii]),1/abs(interactions[ii]),np.sqrt(np.abs(interactions[ii])),np.sign(interactions[ii])*np.sqrt(np.abs(interactions[ii]))]],columns = met_met_edges.columns)
                        met_met_edges = met_met_edges.append(tmpmm,ignore_index = True)
                elif cnst <  (2*(model.num_exch_rxns)):
                    exlb += [{"Metabolite":model.exchanged_metabolites[cnst-(model.num_exch_rxns)],"Coefficient":interactions[ii],"Constraint_Value":bound_rhs[cnst],"Instant_Impact":interactions[ii]*bound_rhs[cnst]}]
                    if round(interactions[cnst],7):
                        tmpmm = pd.DataFrame([[model.exchanged_metabolites[cnst-(model.num_exch_rxns)],metab,model.Name,interactions[ii],abs(interactions[ii]),np.sign(interactions[ii]),1/abs(interactions[ii]),np.sqrt(np.abs(interactions[ii])),np.sign(interactions[ii])*np.sqrt(np.abs(interactions[ii]))]],columns = met_met_edges.columns)
                        met_met_edges = met_met_edges.append(tmpmm,ignore_index = True)
                elif cnst < (2*(model.num_exch_rxns)+2*model.num_fluxes):
                    internal += [{"Index":cnst,"Constraint_Value":bound_rhs[cnst],"Coefficient":interactions[ii],"Instant_Impact":interactions[ii]*bound_rhs[cnst]}]
                else:
                    eq += [{"Index":cnst,"Constraint_Value":bound_rhs[cnst],"Coefficient":interactions[ii],"Instant_Impact":interactions[ii]*bound_rhs[cnst]}]

            const_impact = sum([v['Instant_Impact'] for v in eq]) + sum([v['Instant_Impact'] for v in internal])
            if const_impact:
                impact_summary = [{"Cofactor":di["Metabolite"],"Coefficient":di["Coefficient"]} for di in exub] + [{"Cofactor":di["Metabolite"],"Coefficient":di["Coefficient"]} for di in exlb] + [{"Cofactor":"Constant","Coefficient":const_impact}]
            else:
                impact_summary = [{"Cofactor":di["Metabolite"],"Coefficient":di["Coefficient"]} for di in exub] + [{"Cofactor":di["Metabolite"],"Coefficient":di["Coefficient"]} for di in exlb]


            total_impact = 0

            for di in impact_summary:
                if round(di["Coefficient"],7):
                    total_impact += di["Coefficient"]
                    if di["Cofactor"] == metab:
                        mtch = 1
                        cof = di["Cofactor"]
                    elif di["Cofactor"] == "Constant":
                        mtch = 0
                        cof = "None"
                    else:
                        mtch = -1
                        cof = di["Cofactor"]
                    tmp1 = pd.DataFrame([[model.Name,metab,"Microbe",di["Coefficient"],cof,mtch,abs(di["Coefficient"]),np.sign(di["Coefficient"]),1/abs(di["Coefficient"]),np.sqrt(np.abs(di["Coefficient"])),np.sign(di["Coefficient"])*np.sqrt(np.abs(di["Coefficient"]))]],columns = met_med_net.columns)
                    met_med_net = met_med_net.append(tmp1,ignore_index = True)
            if round(total_impact,7):
                tmp2 = pd.DataFrame([[model.Name,metab,"Microbe",total_impact,abs(total_impact),np.sign(total_impact),1/abs(total_impact),np.sqrt(np.abs(total_impact)),np.sign(total_impact)*np.sqrt(np.abs(total_impact))]],columns = met_med_net_summary.columns)
                met_med_net_summary = met_med_net_summary.append(tmp2,ignore_index = True)


    associated = pd.DataFrame(index = node_table.index,columns = ["In","Out","All"])
    for ndinx in node_table.index:
        nd = node_table.loc[ndinx,"Name"]
        associated.loc[ndinx,"Out"] = ".".join(np.unique(list(met_med_net.loc[met_med_net["Source"] == nd]["Target"])))
        associated.loc[ndinx,"In"] = ".".join(np.unique(list(met_med_net.loc[met_med_net["Target"] == nd]["Source"])))
        associated.loc[ndinx,"All"] = associated.loc[ndinx,"Out"] + associated.loc[ndinx,"In"]#np.concatenate([associated.loc[ndinx,"Out"],associated.loc[ndinx,"In"]])

    node_table = pd.concat([node_table,associated],axis = 1)
    node_table["IntrinsicGrowth"] = np.ones(len(node_table))
    for model in models:
        node_table.loc[model.Name,"IntrinsicGrowth"] = intr_growths[model.Name]

    minuts,sec = divmod(time.time() - start_time, 60)

    for met,mics in microbes_exchanging.items():
        met_met_nodes.loc[met,"Microbes"] = ".".join(mics)

    if report_activity:
        try:
            flobj.write("[species_metabolite_network] Network built in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
        except:
            print("[species_metabolite_network] Network built in ",int(minuts)," minutes, ",sec," seconds.")

    return node_table,met_med_net,met_med_net_summary,met_met_edges,met_met_nodes



def trim_network(edges,nodes,dynamics):
    newnodes = nodes.copy()
    newedges = edges.copy()
    dropped = 0
    messedup = 0
    for nd in dynamics.index:
        if nd in newnodes.index:
            # try:
            if max(dynamics.loc[nd]) < 10**-6:
                newnodes.drop(index = nd, inplace = True)
                newedges = newedges.loc[(newedges["Source"] != nd) & (newedges["Target"] != nd)]
                dropped += 1
            # except:
            #     newnodes.drop(index = nd, inplace = True)
            #     newedges = newedges.loc[(newedges["Source"] != nd) & (newedges["Target"] != nd)]
            #     messedup += 1
    # print("Dropped {}, Messed up {}".format(dropped,messedup))
    return newedges,newnodes



def heuristic_ss(metmed,nodes,report_activity = False):
    '''
    metmed should have columns ["Source","Target","SourceType","Weight","ABS_Weight","Sign_Weight","Distance","ABSRootWeight","SignedRootWeight"]
    '''
    nodetable = nodes[nodes["Type"] == "Microbe"]
    edge_table = pd.DataFrame(columns = ["Source","Target","Weight","Metabolites","ABSWeight","SignWeight","Distance","ABSRootWeight","SignedRootWeight"],dtype = object)
    adjacency = pd.DataFrame(columns = nodetable["Name"],index = nodetable["Name"])
    if len(metmed):
        metmed_woself = metmed[metmed["Source"] != metmed["Target"]]
        selfloops = metmed[metmed["Source"]==metmed["Target"]]
        for tgnd in adjacency.index:
            if report_activity:
                print("Target: {}".format(tgnd))
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
                        edge_table.loc["{}->{}".format(srcnd,tgnd)] = [srcnd,tgnd,weight,list(mediators),abs(weight),np.sign(weight),1/abs(weight),np.sqrt(abs(weight)),np.sign(weight)*np.sqrt(abs(weight))]
            relslp = selfloops[selfloops["Target"] == tgnd]
            if len(relslp):
                slfweight = np.mean(relslp["Weight"].values)
                adjacency.loc[tgnd,tgnd] = slfweight
                edge_table.loc["{}->{}".format(tgnd,tgnd)] = [tgnd,tgnd,slfweight,"None",abs(slfweight),np.sign(slfweight),1/abs(slfweight),np.sqrt(abs(slfweight)),np.sign(slfweight)*np.sqrt(abs(slfweight))]
            else:
                adjacency.loc[tgnd,tgnd] = 0
    return edge_table,nodetable,adjacency







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
