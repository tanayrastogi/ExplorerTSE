import numpy as np


class FD:
        '''
        Fundamental diagram models for FLOW-DENSITY relation.
        '''
        @classmethod
        def parameters(cls, vf=None, rho_j=None, rho_c=None)-> None:
            cls.vf = vf
            cls.rho_j = rho_j
            cls.rho_c = rho_c
            cls.Qmax  = vf*rho_c
            cls.wc    = cls.Qmax/(rho_j - rho_c)
            cls.flow  = cls.__triangular__
            
        @classmethod
        def __triangular__(cls, rho)-> float:
            return min(cls.vf*rho, cls.Qmax, cls.wc*(cls.rho_j - rho))



class CTM():
    def __init__(self, grid:dict, verbose=False) -> None: 
        self.grid=grid
        self.verbose=verbose 

        # Verbose
        if verbose:
            print("[CTM] CTM Simuation Parameters")
            print("\tSpace-time grid: ", self.densities.shape)
            print("\tDelta X: ", self.grid["deltaX"], " [m]")
            print("\tDelta T: ", self.grid["deltaT"], " [s]")
       
           
    def __check_constriants(self, rho_prv=None, rho_cur=None, rho_nxt=None) -> None:
        if rho_prv is not None:
            assert round(rho_prv, 2) <= round(self.rho_j, 2), "Pre Cell density:{} is greater than Jam density:{}".format(round(rho_prv, 2), self.rho_j)
        if rho_cur is not None:
            assert round(rho_cur, 2) <= round(self.rho_j, 2), "Cur Cell density:{} is greater than Jam density:{}".format(round(rho_cur, 2), self.rho_j)
        if rho_nxt is not None:
            assert round(rho_nxt, 2) <= round(self.rho_j, 2), "Nxt Cell density:{} is greater than Jam density:{}".format(round(rho_nxt, 2), self.rho_j)
        
        assert self.wc < self.vf, "The backward wave speed:{} cannot be greater than free flow speed:{} [rho_j:{} rho_c:{}]".format(self.wc, self.vf, self.rho_j, self.rho_c)
        assert self.wc > 0, "The backward wave speed:{} should be geater than zero".format(self.wc)


    def simulate_single(self, rho_prv:float, rho_cur:float, rho_nxt:float, fdParams:dict)->float:
        # FD Parameters
        self.vf    = fdParams['vf']
        self.rho_c = fdParams['rho_c']
        self.rho_j = fdParams['rho_j']
        self.Qmax  = self.vf*self.rho_c
        self.wc    = self.Qmax/(self.rho_j - self.rho_c)
        self.__check_constriants(rho_prv=rho_prv, rho_cur=rho_cur, rho_nxt=rho_nxt)

        flux_in  = min(self.vf*rho_prv, self.Qmax, self.wc*(self.rho_j - rho_cur))
        flux_out = min(self.vf*rho_cur, self.Qmax, self.wc*(self.rho_j - rho_nxt))
        return rho_cur + (self.grid["deltaT"]/self.grid["deltaX"])*(flux_in - flux_out)
    


    def simulate(self, initial_densities:np.array, inflow_densities:np.array, outflow_densities:np.array, fdParams:dict, trafficLight=None)->np.array:
     
        # FD Parameters
        self.vf    = fdParams['vf']
        self.rho_j = fdParams['rho_j']
        self.rho_c = fdParams['rho_c']
        self.Qmax  = self.vf*self.rho_c
        self.wc    = self.Qmax/(self.rho_j - self.rho_c)

        # Initialize the space-time densities
        self.densities=np.empty(shape=(self.grid["num_space"], self.grid["num_time"]), dtype=float)
        self.flux_in  =np.empty(shape=(self.grid["num_space"], self.grid["num_time"]), dtype=float)
        self.flux_out =np.empty(shape=(self.grid["num_space"], self.grid["num_time"]), dtype=float)

        # Initial cell values at time t=0
        self.densities[:, 0]= initial_densities

        # Simulating for the rest of the time-periods
        for t in range(self.grid["num_time"]-1):
            for x in range(self.grid["num_space"]):
                if self.verbose: print("[CTM] Simuting for X/T: {}/{}".format(x, t))
                if x == 0:
                    rho_cur = self.densities[x, t]
                    rho_nxt = self.densities[x+1, t]
                    self.__check_constriants(rho_cur=rho_cur, rho_nxt=rho_nxt)
                    if trafficLight is None: 
                        flux_in  = min(self.vf*inflow_densities[t], self.Qmax, self.wc*(self.rho_j - rho_cur))
                    else:
                        flux_in  = min(self.vf*inflow_densities[t], self.Qmax, self.wc*(self.rho_j - rho_cur)) * trafficLight[0, t]
                    flux_out = min(self.vf*rho_cur, self.Qmax, self.wc*(self.rho_j - rho_nxt))

                elif x == self.grid["num_space"] - 1:
                    rho_prv = self.densities[x-1, t]
                    rho_cur = self.densities[x, t]
                    self.__check_constriants(rho_prv=rho_prv, rho_cur=rho_cur)
                    flux_in  = min(self.vf*rho_prv, self.Qmax, self.wc*(self.rho_j - rho_cur))
                    if trafficLight is None: 
                        flux_out = min(self.vf*rho_cur, self.Qmax, self.wc*(self.rho_j - outflow_densities[t]))
                    else:
                        flux_out = min(self.vf*rho_cur, self.Qmax, self.wc*(self.rho_j - outflow_densities[t])) * trafficLight[1, t]

                else:
                    rho_prv = self.densities[x-1, t]
                    rho_cur = self.densities[x, t]
                    rho_nxt = self.densities[x+1, t]
                    self.__check_constriants(rho_prv=rho_prv, rho_cur=rho_cur, rho_nxt=rho_nxt)
                    flux_in  = min(self.vf*rho_prv, self.Qmax, self.wc*(self.rho_j - rho_cur))
                    flux_out = min(self.vf*rho_cur, self.Qmax, self.wc*(self.rho_j - rho_nxt))

                self.flux_in[x, t+1]   = flux_in
                self.flux_out[x, t+1]  = flux_out
                self.densities[x, t+1] = self.densities[x, t] + (self.grid["deltaT"]/self.grid["deltaX"])*(flux_in - flux_out)

