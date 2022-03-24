import numpy as np
import scipy as sp

try:
    import gurobipy as gb
except ImportError:
    print("Gurobipy import failed.")


import pandas as pd
import time
import cobra as cb

class SurfMod:
    def __init__(self,exchanged_metabolites,gamStar,gamDag,objective,interior_lbs,interior_ubs,exterior_lbfuns,exterior_ubfuns,exterior_lbfuns_derivative = [],exterior_ubfuns_derivatives = [],Name = None,deathrate = 0):

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

        Z = sp.linalg.null_space(gamDag.astype(float))
        gamDagtT = sp.linalg.null_space(Z.T)
        gamDag = gamDagtT.T

        num_exch,num_v = gamStar.shape
        num_internal = gamDag.shape[0]

        self.GammaStar = gamStar.astype(float)
        self.GammaDagger = gamDag.astype(float)
        self.num_fluxes = num_v
        self.num_exch_rxns = num_exch
        self.num_internal_rxns = num_internal

        self.total_var = 4*num_v + 2*num_exch

        rw1 = np.concatenate([gamStar, -gamStar, np.eye(num_exch),np.zeros((num_exch,num_exch+2*num_v))], axis = 1)
        rw2 = np.concatenate([-gamStar, gamStar, np.zeros((num_exch,num_exch)),np.eye(num_exch),np.zeros((num_exch,2*num_v))], axis = 1)
        rw3 = np.concatenate([np.eye(num_v),np.zeros((num_v,num_v+2*num_exch)),np.eye(num_v),np.zeros((num_v,num_v))], axis = 1)
        rw4 = np.concatenate([np.zeros((num_v,num_v)), np.eye(num_v), np.zeros((num_v,2*num_exch+num_v)),np.eye(num_v)], axis = 1)
        rw5 = np.concatenate([gamDag, -gamDag, np.zeros((num_internal,2*num_exch+2*num_v))], axis = 1)
        std_form_mat = np.concatenate([rw1,rw2,rw3,rw4,rw5], axis = 0)
        self.standard_form_constraint_matrix = std_form_mat

        self.objective = np.concatenate([-np.array(objective).astype(float),np.array(objective).astype(float),np.zeros(2*num_exch+2*num_v).astype(float)])

        self.exchange_bounds = np.concatenate([exterior_ubfuns,exterior_lbfuns])
        self.internal_bounds = np.concatenate([interior_ubs.astype(float),interior_lbs.astype(float),np.zeros(num_internal).astype(float)])

        self.exchange_bounds_dt = np.concatenate([exterior_ubfuns_derivatives,exterior_lbfuns_derivative])

        if Name == None:
            self.Name = ''.join([str(np.random.choice(list('abcdefghijklmnopqrstuvwxyz123456789'))) for n in range(5)])
        else:
            self.Name = Name


        self.deathrate = deathrate
        self.exchanged_metabolites = exchanged_metabolites

        self.fba_model = None

        self.current_bases = None

    def fba_gb(self,metabolite_con,secondobj = "total",report_activity = True,flobj = None):

        '''

        Perform FBA and minimize total flux, or use different secondary objective if given.

        '''


        t1 = time.time()

        std_form_mat = self.standard_form_constraint_matrix
        obje = self.objective

        # get the exchange bounds for the current metabolite environment
        exchg_bds = np.array([bd(metabolite_con) for bd in self.exchange_bounds])
        bound_rhs = np.concatenate([exchg_bds,self.internal_bounds])

        #Now we use Gurobi to solve min(x'obje) subject to Ax = b, x \geq 0
        #the actual fluxes calculated are the first 2*num_v entries of x
        #(with forward and reverse reaction fluxes separated)
        #


        if report_activity:
            try:
                flobj.write("fba_gb: initializing LP\n")
            except:
                print("fba_gb: initializing LP")
        growth = gb.Model("growth")
        growth.setParam( 'OutputFlag', False )


        allvars = growth.addMVar(self.total_var,lb = 0.0)
        growth.update()
        # obje = gb.quicksum([a[0]*a[1] for a in zip(obje,allvars)])
        growth.setMObjective(None,obje,None,xc = allvars,sense = gb.GRB.MINIMIZE)

        bds_vec = np.concatenate([upbds_exch,statbds])
        if report_activity:
            try:
                flobj.write("fba_gb: Adding constraints\n")
            except:
                print("fba_gb: Adding constraints")


        # growth.addConstrs((gb.quicksum([MatrixA[i][l]*sparms[l] for l in range(len(sparms))]) <= bds_vec[i] for i in range(len(MatrixA))), name = 'LE')
        # growth.addConstrs((gb.quicksum([Gamma2[i][l]*sparms[l] for l in range(len(sparms))]) == 0 for i in range(len(Gamma2))), name = 'Kernal')
        #
        growth.addMConstr(std_form_mat,allvars,"=",bds_vec)

        growth.update()

        if report_activity:
            try:
                flobj.write("fba_gb: optimizing LP\n")
                flobj.write("fba_gb: optimizing with " + str(len(growth.getConstrs())) + " constraints\n" )
            except:
                print("fba_gb: optimizing LP")
                print("fba_gb: optimizing with ",len(growth.getConstrs()) ," constraints" )
        growth.optimize()


        status = growth.status

        statusdic = {1:"LOADED",2:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",5:"UNBOUNDED"}
        if status in statusdic.keys():
            if report_activity:
                try:
                    flobj.write("fba_gb: LP Status: " +  statusdic[status] + '\n')
                except:
                    print("fba_gb: LP Status: ", statusdic[status])
        else:
            if report_activity:
                try:
                    flobj.write("fba_gb: LP Status: Other\n")
                except:
                    print("fba_gb: LP Status: Other")

        if status == 2:


            val = growth.objVal
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
                    print("FBAWarning: Don't know what to make of given second objective")

            if dosec:
                growth.addConstr(obje == val)
                growth.update()
                growth.setMObjective(None,newobj,None,xc = allvars,sense = gb.GRB.MINIMIZE)
                growth.update()
                growth.optimize()

            if report_activity:
                minuts,sec = divmod(time.time() - t1, 60)
                try:
                    flobj.write("fba_gb: Done in " + str(int(minuts)) + " minutes, " + str(sec) + " seconds.\n")
                except:
                    print("fba_gb: Done in ",int(minuts)," minutes, ",sec," seconds.")


            self.fba_model = growth

            return allvars.getAttr(GRB.Attr.X)
        else:
            return np.array(["failed to prep"])

    # def find_waves_gb(self,flux,metabolite_con,metabolite_con_dt,report_activity = True, flobj = None):
    #
    #     ncon,nvar = self.standard_form_constraint_matrix.shape
    #
    #     betahat = np.where(flux > 0.0)[0]
    #
