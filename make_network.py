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

    met_med_net = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","Cofactor","Match","ABS_Weight","Sign_Weight","Distance"])
    met_med_net_summary = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","ABS_Weight","Sign_Weight","Distance"])

    met_met_edges = pd.DataFrame(columns = ["Source","Target","Microbe","Weight","ABS_Weight","Sign_Weight","Distance"])
    met_met_nodes = pd.DataFrame(columns = ["Microbes"]+[model.Name for model in models],index=metlist)
    microbes_exchanging = dict([(met,[]) for met in metlist])

    if report_activity:
        try:
            flobj.write("[species_metabolite_network] Building network\n")
        except:
            print("[species_metabolite_network] Building network")

    for model in models:

        Q,R,beta = model.current_basis

        #compute (gamma.T B^(-1)).T = (gamma.T R^(-1)Q^(-1)).T = (gamma.T R^(-1)Q.T) = QR.T^(-1)gamma
        growth_vec = np.dot(Q,sp.linalg.solve_triangular(R.T,-model.objective[beta],lower = True))

        #compute -GammaStarB^(-1) = -GammaStar R^(-1) Q^(-1) = -GammaStar R^(-1) Q.T
        # expandgamstar = model.expandGammaStar#np.concatenate([-model.GammaStar,model.GammaStar,np.zeros((model.num_exch_rxns,model.total_var - 2*model.num_fluxes))],axis = 1)
        usage_matrix = -np.dot(model.expandGammaStar[:,beta],sp.linalg.solve_triangular(R,Q.T))

        met_met_nodes.loc[:,model.Name] = np.zeros(len(met_met_nodes))

        #form the bound vector the LP
        metabolite_con = metcons[model.ExchangeOrder]
        exchg_bds = np.array([bd(metabolite_con) for bd in model.exchange_bounds])
        bound_rhs = np.concatenate([exchg_bds,model.internal_bounds])

        for j in range(len(model.exchanged_metabolites)):

            metab = model.exchanged_metabolites[j]

            if report_activity:
                try:
                    flobj.write("[species_metabolite_network] computing {0} connections to {1}\n".format(model.Name,metab))
                except:
                    print("[species_metabolite_network] computing {0} connections to {1}".format(model.Name,metab))




            met_on_mic = growth_vec[j] + growth_vec[j+model.num_exch_rxns]
            met_met_nodes.loc[metab,model.Name] = round(met_on_mic,7)
            microbes_exchanging[metab] += [model.Name]



            if round(met_on_mic,7):
                tmp1 = pd.DataFrame([[metab,model.Name,"Metabolite",met_on_mic,"None",0,abs(met_on_mic),np.sign(met_on_mic),1/np.abs(met_on_mic)]],columns = met_med_net.columns)
                tmp2 = pd.DataFrame([[metab,model.Name,"Metabolite",met_on_mic,abs(met_on_mic),np.sign(met_on_mic),1/np.abs(met_on_mic)]],columns = met_med_net_summary.columns)
                met_med_net = met_med_net.append(tmp1,ignore_index = True)
                met_med_net_summary = met_med_net_summary.append(tmp2,ignore_index = True)

            interactions = usage_matrix[j]

            # Separate Exchange/ExchangeLB/Internal/Positivity constaints/Equilibrium
            eq = []
            exlb = []
            exub = []
            internal = []
            for cnst in range(len(bound_rhs)):#
                if cnst < (model.num_exch_rxns):
                    exub += [{"Metabolite":model.exchanged_metabolites[cnst],"Coefficient":interactions[cnst],"Constraint_Value":bound_rhs[cnst],"Instant_Impact":interactions[cnst]*bound_rhs[cnst]}]
                    if round(interactions[cnst],7):
                        tmpmm = pd.DataFrame([[model.exchanged_metabolites[cnst],metab,model.Name,interactions[cnst],abs(interactions[cnst]),np.sign(interactions[cnst]),1/abs(interactions[cnst])]],columns = met_met_edges.columns)
                        met_met_edges = met_met_edges.append(tmpmm,ignore_index = True)
                elif cnst <  (2*(model.num_exch_rxns)):
                    exlb += [{"Metabolite":model.exchanged_metabolites[cnst-(model.num_exch_rxns)],"Coefficient":interactions[cnst],"Constraint_Value":bound_rhs[cnst],"Instant_Impact":interactions[cnst]*bound_rhs[cnst]}]
                    if round(interactions[cnst],7):
                        tmpmm = pd.DataFrame([[model.exchanged_metabolites[cnst-(model.num_exch_rxns)],metab,model.Name,interactions[cnst],abs(interactions[cnst]),np.sign(interactions[cnst]),1/abs(interactions[cnst])]],columns = met_met_edges.columns)
                        met_met_edges = met_met_edges.append(tmpmm,ignore_index = True)
                elif cnst < (2*(model.num_exch_rxns)+2*model.num_fluxes):
                    internal += [{"Index":cnst,"Constraint_Value":bound_rhs[cnst],"Coefficient":interactions[cnst],"Instant_Impact":interactions[cnst]*bound_rhs[cnst]}]
                else:
                    eq += [{"Index":cnst,"Constraint_Value":bound_rhs[cnst],"Coefficient":interactions[cnst],"Instant_Impact":interactions[cnst]*bound_rhs[cnst]}]

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
                    tmp1 = pd.DataFrame([[model.Name,metab,"Microbe",di["Coefficient"],cof,mtch,abs(di["Coefficient"]),np.sign(di["Coefficient"]),1/abs(di["Coefficient"])]],columns = met_med_net.columns)
                    met_med_net = met_med_net.append(tmp1,ignore_index = True)
            if round(total_impact,7):
                tmp2 = pd.DataFrame([[model.Name,metab,"Microbe",total_impact,abs(total_impact),np.sign(total_impact),1/abs(total_impact)]],columns = met_med_net_summary.columns)
                met_med_net_summary = met_med_net_summary.append(tmp2,ignore_index = True)


    associated = pd.DataFrame(index = node_table.index,columns = ["In","Out","All"])
    for ndinx in node_table.index:
        nd = node_table.loc[ndinx,"Name"]
        associated.loc[ndinx,"Out"] = ".".join(np.unique(list(met_med_net.loc[met_med_net["Source"] == nd]["Target"])))
        associated.loc[ndinx,"In"] = ".".join(np.unique(list(met_med_net.loc[met_med_net["Target"] == nd]["Source"])))
        associated.loc[ndinx,"All"] = associated.loc[ndinx,"Out"] + associated.loc[ndinx,"In"]#np.concatenate([associated.loc[ndinx,"Out"],associated.loc[ndinx,"In"]])

    node_table = pd.concat([node_table,associated],axis = 1)

    minuts,sec = divmod(time.time() - start_time, 60)

    for met,mics in microbes_exchanging.items():
        met_met_nodes.loc[met,"Microbes"] = ".".join(mics)

    if report_activity:
        try:
            flobj.write("[species_metabolite_network] Network built in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
        except:
            print("[species_metabolite_network] Network built in ",int(minuts)," minutes, ",sec," seconds.")

    return node_table,met_med_net,met_med_net_summary,met_met_edges,met_met_nodes

















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
