import numpy as np
from scipy.sparse import dok_matrix

# Consider a dynamical system that can be written as
#
#     dx_i/dt = x_i (g^T M_i h_i(y))
#     dy/dt = -sum(x_i N_i h_i(y))
#
# Then a sufficient condition for the existence of a set of linear conservation laws
# A1 dx/dt + A2 dy/dt = 0
# is
#
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
