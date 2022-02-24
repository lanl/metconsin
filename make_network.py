import numpy as np
from scipy.sparse import dok_matrix
import pandas as pd
import time
import scipy as sp



###### Build a species-metabolite interaction network (that's not hard)
#### Then build a species-species network (that's fuzzier or harder)

def species_metabolite_network(bases,metlist,metcons,models):

    start_time = time.time()
    node_table = pd.DataFrame(np.array([list(bases.keys()) + list(metlist),["Microbe"]*len(bases) + ["Metabolite"]*len(metlist)]).T,columns = ["Name","Type"])

    met_med_net = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","Cofactor","Match","ABS_Weight","Sign_Weight","Distance"])
    met_med_net_summary = pd.DataFrame(columns = ["Source","Target","SourceType","Weight","ABS_Weight","Sign_Weight","Distance"])

    print("[species_metabolite_network] Building network")
    for ky in bases.keys():

        Q = bases[ky][0]
        R = bases[ky][1]

        # print(Q.shape,R.shape)
        #compute (gamma.T B^(-1)).T = (gamma.T R^(-1)Q^(-1)).T = (gamma.T R^(-1)Q.T) = QR.T^(-1)gamma
        growth_vec = np.dot(Q,sp.linalg.solve_triangular(R.T,models[ky].objective,lower = True))

        #compute -Gamma1B^(-1) = -Gamma1 R^(-1) Q^(-1) = -Gamma1 R^(-1) Q.T
        usage_matrix = -np.dot(models[ky].Gamma1,sp.linalg.solve_triangular(R,Q.T))

        for j in range(len(metlist)):

            print("[species_metabolite_network] computing {0} connections to {1}".format(ky,metlist[j]))

            metab = metlist[j]

            exch = [metcons[metlist[i]]*models[ky].uptakes[i] for i in range(len(metlist))]
            all_constr = np.concatenate([exch,models[ky].statbds,np.zeros(len(models[ky].Gamma2))])

            l = np.where(bases[ky][2] == j)[0]

            if len(l):
                # d/dy_j h(y) = alpha_j 1_{h_l=y_j}
                l = l[0]
                met_on_mic = exch[j]*growth_vec[l]
            else:
                met_on_mic = 0


            if met_on_mic:
                tmp1 = pd.DataFrame([[metab,ky,"Metabolite",met_on_mic,"None",0,abs(met_on_mic),np.sign(met_on_mic),1/np.sign(met_on_mic)]],columns = met_med_net.columns)
                tmp2 = pd.DataFrame([[metab,ky,"Metabolite",met_on_mic,abs(met_on_mic),np.sign(met_on_mic),1/np.sign(met_on_mic)]],columns = met_med_net_summary.columns)
                met_med_net = met_med_net.append(tmp1,ignore_index = True)
                met_med_net_summary = met_med_net_summary.append(tmp2,ignore_index = True)

            interactions = usage_matrix[j]

            # Separate Exchange/ExchangeLB/Internal/Positivity constaints/Equilibrium
            eq = []
            exlb = []
            ex = []
            internal = []
            posi = []
            for i in range(len(bases[ky][2])):#
                cnst = bases[ky][2][i]
                if cnst < len(exch):
                    ex += [{"Metabolite":metlist[cnst],"Coefficient":interactions[i],"Constrain_Value":all_constr[cnst],"Impact":interactions[i]*all_constr[cnst]}]
                elif cnst <  (len(exch) + len(models[ky].exchgLB)):
                    exlb += [{"Metabolite":metlist[cnst-len(exch)],"Coefficient":interactions[i],"Constrain_Value":all_constr[cnst],"Impact":interactions[i]*all_constr[cnst]}]
                elif cnst < (len(exch) + len(models[ky].exchgLB)+2*len(models[ky].intUB)):
                    internal += [{"Index":cnst,"Constraint_Value":all_constr[cnst],"Coefficient":interactions[i],"Impact":interactions[i]*all_constr[cnst]}]
                elif cnst < (len(exch) + len(models[ky].exchgLB)+2*len(models[ky].intUB) + models[ky].MatrixA.shape[1]):
                    posi += [{"Index":cnst,"Constraint_Value":all_constr[cnst],"Coefficient":interactions[i],"Impact":interactions[i]*all_constr[cnst]}]
                else:
                    eq += [{"Index":cnst,"Constraint_Value":all_constr[cnst],"Coefficient":interactions[i],"Impact":interactions[i]*all_constr[cnst]}]

            const_impact = sum([v['Impact'] for v in eq]) + sum([v['Impact'] for v in exlb]) + sum([v['Impact'] for v in internal]) + sum([v['Impact'] for v in posi])
            if const_impact:
                impact_summary = [{"Cofactor":di["Metabolite"],"Coefficient":di["Coefficient"]} for di in ex] + [{"Cofactor":"Constant","Coefficient":const_impact}]
            else:
                impact_summary = [{"Cofactor":di["Metabolite"],"Coefficient":di["Coefficient"]} for di in ex]

            total_impact = 0

            for di in impact_summary:
                if round(di["Coefficient"],8):
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
                    tmp1 = pd.DataFrame([[ky,metab,"Microbe",di["Coefficient"],cof,mtch,abs(di["Coefficient"]),np.sign(di["Coefficient"]),1/(di["Coefficient"])]],columns = met_med_net.columns)
                    met_med_net = met_med_net.append(tmp1,ignore_index = True)
            if round(total_impact,8):
                tmp2 = pd.DataFrame([[ky,metab,"Microbe",total_impact,abs(total_impact),np.sign(total_impact),1/(total_impact)]],columns = met_med_net_summary.columns)
                met_med_net_summary = met_med_net_summary.append(tmp2,ignore_index = True)


    associated = pd.DataFrame(index = node_table.index,columns = ["In","Out","All"])
    for ndinx in node_table.index:
        nd = node_table.loc[ndinx,"Name"]
        associated.loc[ndinx,"Out"] = np.unique(list(met_med_net.loc[met_med_net["Source"] == nd]["Target"]))
        associated.loc[ndinx,"In"] = np.unique(list(met_med_net.loc[met_med_net["Target"] == nd]["Source"]))
        associated.loc[ndinx,"All"] = np.concatenate([associated.loc[ndinx,"Out"],associated.loc[ndinx,"In"]])

    node_table = pd.concat([node_table,associated],axis = 1)

    minuts,sec = divmod(time.time() - start_time, 60)
    # try:
    #     flobj.write("prep_indv_model", self.Name,": Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
    # except:
    print("[species_metabolite_network] Network built in ",int(minuts)," minutes, ",sec," seconds.")

    return node_table,met_med_net,met_med_net_summary

















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


def get_linear_system(QRsdict,QRkys,GammaStars,all_gs):
    QRs = [QRsdict[ky] for ky in QRkys]
    #we need to find A1,A2. All are lists over i.
    num_xi = len(QRs) #m
    num_vs = [Mi.shape[1] for Mi in GammaStars]#li
    num_y = GammaStars[0].shape[0]#n
    num_eqs = num_y*sum(num_vs)
    eqs = dok_matrix((num_eqs, num_y**2 + num_y*num_xi), dtype=np.float64)#{}
    for i in range(num_xi):
        #To find y = g^T M_i, we note that R_i^TQ_i^Ty = g
        # so let w = solve(R_i^T,g), and then y = Q_iw (b/c Q is unitary)
        w = np.linalg.solve(QRs[i][1].T,all_gs[i])
        gTM = np.dot(QRs[i][0],w)
        ####
        for k in range(num_vs[i]):
            kthBinv = np.linalg.solve(QRs[i][1],QRs[i][0][k])
            KthN = np.dot(GammaStars[i],kthBinv)
            for j in range(num_y):
                eqnum = num_y*sum(num_vs[:i]) + j*num_vs[i] + k
                #save the coefficients index so that the first n*m variables are the
                #entries of A1, flattened, and next n*n are entries of A2, flattened.
                coeffs = {}
                #only one entry of A1
                if round(gTM[k],9):
                    eqs[eqnum,j*num_xi + i] = gTM[k]
                    # coeffs[j*num_xi + i] = gTM[k]
                #jth row of A2 has coefficients kth column of GammaStar*B^(-1)
                #To find this, we compute the kth column of B^(-1) (above)
                for l in np.where(KthN.round(9))[0]:
                    # coeffs[num_xi*num_y + num_y*j + l] = KthN[l]
                    eqs[eqnum,num_xi*num_y + num_y*j + l] = KthN[l]
                # eqs[eqnum] = coeffs
    #return a dictionary keyed by equation number (row in linear system), each entry a dict
    # keyed by variable number (column in linear system) with value equal to coefficient
    # This should be easy to use with various sparse solvers.
    return eqs
