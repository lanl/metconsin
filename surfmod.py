import numpy as np
import scipy.linalg as la

try:
    import gurobipy as gb
except ImportError:
    print("Gurobipy import failed.")


import pandas as pd
import time
import cobra as cb

class SurfMod:
    def __init__(self,exchanged_metabolites,gamStar,gamDag,objective,intrn_order,interior_lbs,interior_ubs,exterior_lbfuns,exterior_ubfuns,exterior_lbfuns_derivative,exterior_ubfuns_derivatives,lbfuntype = "",ubfuntype = "",Name = None,deathrate = 0,gamma_star_indices = "Auto"):

        #put the model in standard form by forming the system
        # [GamStar -GamStar I 0 0 0]         [f(y)]
        # [-GamStar GamStar 0 I 0 0] [v_f]   [g(y)]
        # [I         0      0 0 I 0] [v_r] = [interior_ubs]
        # [0         I      0 0 0 I] [s]     [interior_lbs]
        # [GamDag  -GamDag  0 0 0 0]          [0]

        # where f,g are the upper/lower exchange bounds, and are functions of the
        #environmental metabolites.

        #All that matters is Ker(Gamma2), so we can replace Gamma2 with a new
        #matrix with orthonormal rows that has the same kernal.

        Z = la.null_space(gamDag.astype(float))
        gamDagtT = la.null_space(Z.T)
        gamDag = gamDagtT.T

        num_exch,num_v = gamStar.shape
        num_internal = gamDag.shape[0]

        self.GammaStar = gamStar.astype(float)

        if isinstance(gamma_star_indices,str):
            if gamma_star_indices == "Auto":
                self.ExchangeOrder = np.arange(gamStar.shape[0]).astype(int)
            else:
                print("Cannot understand exchange mapping. Assuming 1:1 and in order, using default (Auto)")
                self.ExchangeOrder = np.arange(gamStar.shape[0]).astype(int)
        elif isinstance(gamma_star_indices,np.ndarray):#should add a check here.
            self.ExchangeOrder = gamma_star_indices
        else:
            print("Cannot understand exchange mapping. Assuming 1:1 and in order, using default (Auto)")
            self.ExchangeOrder = np.arange(gamStar.shape[0]).astype(int)


        self.GammaDagger = gamDag.astype(float)
        self.num_fluxes = num_v
        self.num_exch_rxns = num_exch
        self.num_internal_metabolites = num_internal

        self.total_var = 4*num_v + 2*num_exch
        rw1 = np.concatenate([gamStar.astype(float), -gamStar.astype(float), np.eye(num_exch),np.zeros((num_exch,num_exch+2*num_v))], axis = 1)
        rw2 = np.concatenate([-gamStar.astype(float), gamStar.astype(float), np.zeros((num_exch,num_exch)),np.eye(num_exch),np.zeros((num_exch,2*num_v))], axis = 1)
        rw3 = np.concatenate([np.eye(num_v),np.zeros((num_v,num_v+2*num_exch)),np.eye(num_v),np.zeros((num_v,num_v))], axis = 1)
        rw4 = np.concatenate([np.zeros((num_v,num_v)), np.eye(num_v), np.zeros((num_v,2*num_exch+num_v)),np.eye(num_v)], axis = 1)
        rw5 = np.concatenate([gamDag, -gamDag, np.zeros((num_internal,2*num_exch+2*num_v))], axis = 1)
        std_form_mat = np.concatenate([rw1,rw2,rw3,rw4,rw5], axis = 0)

        self.expandGammaStar = np.concatenate([gamStar.astype(float),-gamStar.astype(float),np.zeros((num_exch,2*num_v + 2*num_exch))],axis = 1)

        self.objective = np.concatenate([-np.array(objective).astype(float),np.array(objective).astype(float),np.zeros(2*num_exch+2*num_v).astype(float)])

        self.flux_order = intrn_order

        self.exchange_bounds = np.concatenate([exterior_ubfuns,exterior_lbfuns])
        self.lower_exch_type = lbfuntype
        self.upper_exch_type = ubfuntype


        interior_ubs_min0 = np.array([max(bd,0) for bd in interior_ubs])
        interior_lbs_min0 = np.array([max(bd,0) for bd in interior_lbs])
        all_internal = np.concatenate([interior_ubs_min0.astype(float),interior_lbs_min0.astype(float),np.zeros(num_internal).astype(float)])

        #We need to add constraints for negative bounds.
        ##A negative lower bound means the FORWARD reaction must be above -bd
        ##A negative upper bound means the REVERSE reaction must be above -bd

        neg_lower = np.where(interior_lbs < 0) #need to add a constraint for the FORWARD reaction
        num_lower = len(neg_lower[0])
        if num_lower:
            std_form_mat = np.concatenate([std_form_mat,np.zeros((std_form_mat.shape[0],num_lower))],axis = 1)
            new_rows = np.concatenate([-np.eye(num_v)[neg_lower],np.zeros((num_lower,3*num_v+2*num_exch)),np.eye(num_lower)],axis = 1)
            std_form_mat = np.concatenate([std_form_mat,new_rows],axis = 0)
            #for the constraint a<x<b, need a<b...make sure a = min(a,b)
            all_internal = np.concatenate([all_internal,-np.minimum(-interior_lbs[neg_lower],interior_ubs_min0[neg_lower])])
            self.total_var += num_lower
            self.objective = np.concatenate([self.objective,np.zeros(num_lower)])
            self.expandGammaStar = np.concatenate([self.expandGammaStar,np.zeros((self.expandGammaStar.shape[0],num_lower))],axis = 1)


        neg_upper = np.where(interior_ubs < 0) #need to add a constraint for the REVERSE reaction
        num_upper = len(neg_upper[0])
        if num_upper:
            std_form_mat = np.concatenate([std_form_mat,np.zeros((std_form_mat.shape[0],num_upper))],axis = 1)
            new_rows = np.concatenate([np.zeros((num_upper,num_v)),-np.eye(num_v)[neg_upper],np.zeros((num_upper,2*num_v+2*num_exch+num_lower)),np.eye(num_upper)],axis = 1)
            std_form_mat = np.concatenate([std_form_mat,new_rows],axis = 0)
            all_internal = np.concatenate([all_internal,-np.minimum(-interior_ubs[neg_upper],interior_lbs_min0[neg_upper])])
            self.total_var += num_upper
            self.objective = np.concatenate([self.objective,np.zeros(num_upper)])
            self.expandGammaStar = np.concatenate([self.expandGammaStar,np.zeros((self.expandGammaStar.shape[0],num_upper))],axis = 1)


        self.standard_form_constraint_matrix = std_form_mat
        self.internal_bounds = all_internal

        self.exchange_bounds_dt = np.concatenate([exterior_ubfuns_derivatives,exterior_lbfuns_derivative])

        if Name == None:
            self.Name = ''.join([str(np.random.choice(list('abcdefghijklmnopqrstuvwxyz123456789'))) for n in range(5)])
        else:
            self.Name = Name


        self.deathrate = deathrate
        self.exchanged_metabolites = exchanged_metabolites

        self.fba_model = None

        self.current_basis = None

        self.fba_basic_index = None

    def fba_gb(self,master_metabolite_con,secondobj = "total",report_activity = True,flobj = None):

        '''

        Perform FBA and minimize total flux, or use different secondary objective if given.

        '''


        t1 = time.time()

        metabolite_con = master_metabolite_con[self.ExchangeOrder]

        std_form_mat = self.standard_form_constraint_matrix
        obje = self.objective

        # get the exchange bounds for the current metabolite environment
        exchg_bds = np.array([bd(metabolite_con) for bd in self.exchange_bounds])

        upe = exchg_bds[self.num_exch_rxns:]
        lowe = exchg_bds[:self.num_exch_rxns]
        upneg = np.where(upe<0)
        # print(upe[upneg],lowe[upneg])
        lowneg = np.where(lowe<0)
        # print(upe[lowneg],lowe[lowneg])

        bound_rhs = np.concatenate([exchg_bds,self.internal_bounds])

        #Now we use Gurobi to solve min(x'obje) subject to Ax = b, x \geq 0
        #the actual fluxes calculated are the first 2*num_v entries of x
        #(with forward and reverse reaction fluxes separated)
        #


        if report_activity:
            try:
                flobj.write(self.Name," fba_gb: initializing LP\n")
            except:
                print(self.Name," fba_gb: initializing LP")
        growth = gb.Model("growth")
        growth.setParam( 'OutputFlag', False )
        growth.setParam( 'Presolve', 0)


        allvars = growth.addMVar(self.total_var,lb = 0.0)
        growth.update()
        growth.setMObjective(None,obje,0,xc = allvars,sense = gb.GRB.MINIMIZE)
        growth.update()
        if report_activity:
            try:
                flobj.write(self.Name," fba_gb: Adding constraints\n")
            except:
                print(self.Name," fba_gb: Adding constraints")


    #
        growth.addMConstr(std_form_mat,allvars,"=",bound_rhs)

        growth.update()

        if report_activity:
            try:
                flobj.write(self.Name," fba_gb: optimizing LP\n")
                flobj.write(self.Name," fba_gb: optimizing with " + str(len(growth.getConstrs())) + " constraints\n" )
            except:
                print(self.Name," fba_gb: optimizing LP")
                print(self.Name," fba_gb: optimizing with ",len(growth.getConstrs()) ," constraints" )
        growth.optimize()


        status = growth.status

        statusdic = {1:"LOADED",2:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",5:"UNBOUNDED"}
        if status in statusdic.keys():
            if report_activity:
                try:
                    flobj.write(self.Name," fba_gb: LP Status: " +  statusdic[status] + '\n')
                except:
                    print(self.Name," fba_gb: LP Status: ", statusdic[status])
        else:
            if report_activity:
                try:
                    flobj.write(self.Name," fba_gb: LP Status: Other\n")
                except:
                    print(self.Name," fba_gb: LP Status: Other")

        if status == 2:


            val = growth.getObjective().getValue()
            dosec = False

            if secondobj == "total":
                newobj = np.concatenate([np.ones(2*self.num_fluxes),np.zeros(self.total_var - 2*self.num_fluxes)])
                dosec = True

            elif (type(secondobj) != str) and hasattr(secondobj, "__len__"):
                if len(secondobj) == self.num_fluxes:
                    newobj = np.concatenate([secondobj,-secondobj,np.zeros(self.total_var - 2*self.num_fluxes)])
                    dosec = True

                elif len(secondobj) == 2*self.num_fluxes:
                    newobj = np.concatenate([secondobj,np.zeros(self.total_var - 2*self.num_fluxes)])
                    dosec = True

                elif len(secondobj) == self.total_var:
                    newobj = secondobj
                    dosec = True

                else:
                    print(self.Name," FBAWarning: Don't know what to make of given second objective")

            if dosec:
                growth.addMConstr(np.array([obje]),allvars,"=",np.array([val]))
                growth.update()
                growth.setMObjective(None,newobj,0,xc = allvars,sense = gb.GRB.MINIMIZE)
                growth.update()
                growth.optimize()

            if report_activity:
                minuts,sec = divmod(time.time() - t1, 60)
                try:
                    flobj.write(self.Name," fba_gb: Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
                except:
                    print(self.Name," fba_gb: Done in ",int(minuts)," minutes, ",sec," seconds.")


            self.fba_model = growth
            self.fba_basic_index = np.where(allvars.getAttr(gb.GRB.Attr.VBasis) == 0)

            return allvars.getAttr(gb.GRB.Attr.X),-val
        else:
            return np.array(["failed to prep"])

    def find_waves_gb(self,flux,master_metabolite_con,master_metabolite_con_dt,report_activity = True, flobj = None, careful = False):

        '''
        Gets basis for LP at given solution that can be used to simulate forward, choosing from degenerate bases
        '''

        flux = flux.round(7)

        t1 = time.time()
        metabolite_con = master_metabolite_con[self.ExchangeOrder]
        metabolite_con_dt = master_metabolite_con_dt[self.ExchangeOrder]

        ncon,nvar = self.standard_form_constraint_matrix.shape

        # print(np.linalg.matrix_rank(self.standard_form_constraint_matrix))


        if report_activity:
            try:
                flobj.write(self.Name," find_waves_gb: Projecting\n")
            except:
                print(self.Name," find_waves_gb: Projecting LP")

        beta_hat = np.where(flux > 0.0)[0]

        original_basis = self.standard_form_constraint_matrix[:,beta_hat]
        # print((original_basis.round(7) == 0).all(axis = 0).any())
        # print("Starting vectors shape: ",original_basis.shape)
        # print("Starting vectors rank: ",np.linalg.matrix_rank(original_basis))


        # print("beta hat size: ",len(beta_hat))


        if len(beta_hat) == ncon:
            if report_activity:
                try:
                    flobj.write(self.Name," find_waves_gb: Only One Choice\n")
                except:
                    print(self.Name," find_waves_gb: Only One Choice LP")

            beta_index = beta_hat

            wave_to_ride = self.standard_form_constraint_matrix[:,beta_index]

            Q,R = np.linalg.qr(wave_to_ride)

            if report_activity:
                minuts,sec = divmod(time.time() - t1, 60)
                try:
                    flobj.write(self.Name," find_waves_gb: Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
                except:
                    print(self.Name," find_waves_gb: Done in ",int(minuts)," minutes, ",sec," seconds.")


            self.current_basis =  (Q,R,beta_index)

        else:

            s_hat =  self.standard_form_constraint_matrix[:,beta_hat]

            # print(np.linalg.matrix_rank(s_hat))

            exchg_bds_dt = np.array([bd(metabolite_con,metabolite_con_dt) for bd in self.exchange_bounds_dt])
            bound_rhs_dt = np.concatenate([exchg_bds_dt,np.zeros(len(self.internal_bounds))]).round(8)
            # print(bound_rhs_dt[np.where(bound_rhs_dt != 0)])

            beta_hat_comp = np.where(flux == 0.0)[0]
            s_hat_comp = self.standard_form_constraint_matrix[:,beta_hat_comp]

            # print(np.linalg.matrix_rank(s_hat_comp))

            A_tilde = proj_orth(s_hat_comp,s_hat)
            #Apparently we have to remove the 0 columns because Gurobi might
            # Choose those for a "basis" which...WTF.
            nz_columns = np.invert((A_tilde.round(7) == 0).all(axis = 0))
            beta_hat_comp = beta_hat_comp[nz_columns]
            A_tilde = A_tilde[:,nz_columns]
            # print("A tilde shape: ",A_tilde.shape)
            # print("A tilde rank: ",np.linalg.matrix_rank(A_tilde))

            b_tilde = proj_orth(bound_rhs_dt,s_hat).round(8)


            # ls = np.linalg.lstsq(A_tilde,b_tilde,rcond = None)
            # print(sum((b_tilde - np.dot(A_tilde,ls[0]))**2))


            if report_activity:
                try:
                    flobj.write(self.Name," find_waves_gb: initializing LP\n")
                except:
                    print(self.Name," find_waves_gb: initializing LP")
            waves = gb.Model("waves")
            waves.setParam( 'OutputFlag', False )
            waves.setParam( 'Presolve', 0)


            allvars = waves.addMVar(A_tilde.shape[1],lb = 0)#,ub = 1)#
            waves.update()
            waves.setMObjective(None,-np.ones(A_tilde.shape[1]),0,xc = allvars,sense = gb.GRB.MINIMIZE)
            waves.update()
            if report_activity:
                try:
                    flobj.write(self.Name," find_waves_gb: Adding constraints\n")
                except:
                    print(self.Name," find_waves_gb: Adding constraints")


        #
            waves.addMConstr(A_tilde,allvars,"=",b_tilde)

            waves.update()

            if report_activity:
                try:
                    flobj.write(self.Name," find_waves_gb: optimizing LP\n")
                    flobj.write(self.Name," find_waves_gb: optimizing with " + str(len(waves.getConstrs())) + " constraints\n" )
                except:
                    print(self.Name," find_waves_gb: optimizing LP")
                    print(self.Name," find_waves_gb: optimizing with ",len(waves.getConstrs()) ," constraints" )
            waves.optimize()

            status = waves.status

            statusdic = {1:"LOADED",2:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",5:"UNBOUNDED"}
            if status in statusdic.keys():
                if report_activity:
                    try:
                        flobj.write(self.Name," find_waves_gb: LP Status: " +  statusdic[status] + '\n')
                    except:
                        print(self.Name," find_waves_gb: LP Status: ", statusdic[status])
            else:
                if report_activity:
                    try:
                        flobj.write(self.Name," find_waves_gb: LP Status: Other\n")
                    except:
                        print(self.Name," find_waves_gb: LP Status: Other")

            if status == 2:

                self.current_basis =  getBetaTilde(waves,beta_hat,beta_hat_comp,A_tilde,self,careful = careful)


                if report_activity:
                    minuts,sec = divmod(time.time() - t1, 60)
                    try:
                        flobj.write(self.Name," find_waves_gb: Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
                    except:
                        print(self.Name," find_waves_gb: Done in ",int(minuts)," minutes, ",sec," seconds.")

                return None





            elif status == 4:

                waves.setParam("DualReductions", 0)
                waves.update()
                waves.optimize()
                status = waves.status

                statusdic = {1:"LOADED",2:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",5:"UNBOUNDED"}
                if status in statusdic.keys():
                    if report_activity:
                        try:
                            flobj.write(self.Name," find_waves_gb: LP Status: " +  statusdic[status] + '\n')
                        except:
                            print(self.Name," find_waves_gb: LP Status: ", statusdic[status])
                else:
                    if report_activity:
                        try:
                            flobj.write(self.Name," find_waves_gb: LP Status: Other\n")
                        except:
                            print(self.Name," find_waves_gb: LP Status: Other")

            if status == 5:
                print("Changing to 0 objective.")
                waves.setMObjective(None,np.zeros(A_tilde.shape[1]),0,xc = allvars,sense = gb.GRB.MINIMIZE)
                waves.update()
                waves.optimize()

                status = waves.status

                statusdic = {1:"LOADED",2:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",5:"UNBOUNDED"}
                if status in statusdic.keys():
                    if report_activity:
                        try:
                            flobj.write(self.Name," find_waves_gb: LP Status: " +  statusdic[status] + '\n')
                        except:
                            print(self.Name," find_waves_gb: LP Status: ", statusdic[status])
                else:
                    if report_activity:
                        try:
                            flobj.write(self.Name," find_waves_gb: LP Status: Other\n")
                        except:
                            print(self.Name," find_waves_gb: LP Status: Other")

                if status == 2:

                    #get the index of the basic variables of waves

                    self.current_basis =  getBetaTilde(waves,beta_hat,beta_hat_comp,A_tilde,self,careful = careful)

                    if report_activity:
                        minuts,sec = divmod(time.time() - t1, 60)
                        try:
                            flobj.write(self.Name," find_waves_gb: Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
                        except:
                            print(self.Name," find_waves_gb: Done in ",int(minuts)," minutes, ",sec," seconds.")


                else:

                    self.current_basis = None

                return None

            else:

                self.current_basis = None
                return None

    def compute_internal_flux(self,master_metabolite_con):

        '''
        Compute current fluxes (including slacks) from current basis & metabolite concentration
        '''
        metabolite_con = master_metabolite_con[self.ExchangeOrder]
        exchg_bds = np.array([bd(metabolite_con) for bd in self.exchange_bounds])
        bound_rhs = np.concatenate([exchg_bds,self.internal_bounds])

        Q,R,beta = self.current_basis

        fl_beta = la.solve_triangular(R,np.dot(Q.T,bound_rhs))

        all_vars = np.zeros(self.total_var)
        all_vars[beta] = fl_beta

        return all_vars




def getBetaTilde(waves,beta_hat,beta_hat_comp,A_tilde,smod, careful = False):

    # if careful:
    #     print(smod.standard_form_constraint_matrix[:,beta_hat].shape, " ",np.linalg.matrix_rank(smod.standard_form_constraint_matrix[:,beta_hat]))
    #     print(A_tilde.shape," ",np.linalg.matrix_rank(A_tilde))

    basic_vars = np.where(np.array(waves.vbasis) == 0)[0]

    if len(basic_vars) < A_tilde.shape[1]:
        #we need a linearly independent collection of the columns of A_tilde
        #first project away the ones we already have
        basic_vars_comp = np.where(np.array(waves.vbasis) != 0)[0]
        spc = A_tilde[:,basic_vars]
        bas = np.linalg.qr(spc,mode="complete")[0]
        A_t2 = proj_orth(A_tilde[:,basic_vars_comp],A_tilde[:,basic_vars])
        #then get the linearly independent columns of the remaining
        lin_indep_columns = getLIcols(A_t2)
        basic_vars = np.sort(np.concatenate([basic_vars,basic_vars_comp[lin_indep_columns]]))


    beta_tilde = beta_hat_comp[basic_vars]

    beta_index = np.sort(np.concatenate([beta_hat,beta_tilde]))

    wave_to_ride = smod.standard_form_constraint_matrix[:,beta_index]

    if careful:
        # print("Full Mat Rank ",np.linalg.matrix_rank(smod.standard_form_constraint_matrix))
        rk = np.linalg.matrix_rank(wave_to_ride)
        shp = wave_to_ride.shape
        print("Basis Shape: ",shp)
        print("Basis Rank: ", rk)
        if (shp[0] != shp[1]) or (shp[0] != rk):
            return None

    Q,R = np.linalg.qr(wave_to_ride)

    return Q,R,beta_index

def getLIcols(mat):
    #assume mat is mxn, m<=n, rank(mat) = m
    rows,cols = mat.shape
    licols = np.array([])
    col_inds = np.arange(cols)
    while(len(licols)<rows):
        mat = mat/abs(mat).max()
        U = la.lu(mat)[2].round(7)
        lin_indep_columns = np.unique(np.array([np.flatnonzero(U[i, :])[0] for i in range(U.shape[0]) if len(np.flatnonzero(U[i,:]))]))
        if len(lin_indep_columns):
            licols = np.concatenate([licols,col_inds[lin_indep_columns]])
            col_inds = np.delete(col_inds,lin_indep_columns)
            mat = proj_orth(np.delete(mat,lin_indep_columns,axis = 1),mat[:,lin_indep_columns])
        else:
            print("Failed to find LI columns")
            print("remaining shape ",mat.shape)
            print("remaining rank ", np.linalg.matrix_rank(mat))
            break
    return licols.astype(int)

def proj_orth(mat,spc):
    '''
    given a basis for a subspace, computes an (orthogonal) basis for the orthogonal space and
    returns coordinates for a given vector in that basis.
    '''
    bas = np.linalg.qr(spc,mode = "complete")[0]
    return np.linalg.solve(bas,mat)[spc.shape[1]:]
