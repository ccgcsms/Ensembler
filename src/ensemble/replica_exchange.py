

import numpy as np
import pandas as pd
from collections import Iterable
import scipy.constants as const
from typing import List, Dict, Tuple
import copy
import itertools as it

from src import system, potentials as pot, integrator
from src.ensemble._replica_graph import ReplicaExchange


class TemperatureReplicaExchange(ReplicaExchange):

    _parameter_name:str = "temperature"
    coordinate_dimensions:int = 1
    replica_graph_dimensions:int = 1

    nSteps_between_trials:int

    def __init__(self, system, temperature_Range:Iterable=range(298,320, 10), exchange_criterium=None, steps_between_trials=10,
                 exchange_trajs:bool=True):
        super().__init__(system=system, exchange_dimensions={self._parameter_name:temperature_Range},
                         exchange_criterium=exchange_criterium, steps_between_trials=steps_between_trials)

        if(exchange_trajs):
            self.exchange_param = "trajectory"
        else:
            self.exchange_param = self.parameter_names[0]

    def exchange(self, verbose:bool=False):
        """
        .. autofunction:: Exchange the Trajectory of T-replicas in pairwise fashion
        :param verbose:
        :return:
        """
        self._currentTrial += 1

        #Get Potential Energies
        original_T, original_totPots, swapped_T, swapped_totPots = self._collect_replica_energies(verbose)

        if(verbose):
            print("origTotE ", [original_totPots[key] for key in sorted(original_T)])
            print("SWPTotE ", [swapped_totPots[key] for key in sorted(swapped_T)])

        #decide exchange
        exchanges_to_make={}
        for partner1, partner2 in zip(original_T[self.exchange_offset::2], original_T[1+self.exchange_offset::2]):
            originalEnergies = np.add(original_totPots.get(partner1), original_totPots.get(partner2))
            swapEnergies = np.add(swapped_totPots.get(partner1), swapped_totPots.get(partner2))
            exchanges_to_make.update({(partner1, partner2): self.exchange_criterium(originalEnergies, swapEnergies)})

            #print("partners: "+str(partner1)+"/"+str(partner2)+" \t originalEnergies "+str(originalEnergies)+" / Swap "+str(swapEnergies))
            #print("randomness part: "+str(self._defaultRandomness(originalEnergies, swapEnergies)))

        #Acutal Exchange of params (actually trajs here
        if(verbose):
            print("Exchange: ", exchanges_to_make)
            print("exchaning param: ", self.exchange_param)

        #execute exchange
        self._do_exchange(exchanges_to_make, original_T, original_totPots, swapped_totPots, verbose)
        self._current_exchanges = exchanges_to_make
        #update the offset
        self.exchange_offset = (self.exchange_offset+1)%2

    def _swap_coordinates(self, original_exCoord):
        ##take care of offset situations and border replicas
        swapped_exCoord = [] if self.exchange_offset == 0 else [original_exCoord[0]]
        ##generate sequence with swapped params
        for partner1, partner2 in zip(original_exCoord[self.exchange_offset::2],
                                      original_exCoord[1 + self.exchange_offset::2]):
            swapped_exCoord.extend([partner2, partner1])
        ##last replica on the border?
        if (self.exchange_offset == 0):
            swapped_exCoord.append(original_exCoord[-1])
        return swapped_exCoord

    def _adapt_system_to_exchange_coordinate(self, swapped_exCoord, original_exCoord):
        self._scale_velocities_fitting_to_temperature(swapped_exCoord, original_exCoord)

    def _scale_velocities_fitting_to_temperature(self, original_T, swapped_T):
        if (not any([getattr(self.replicas[replica], "_currentVelocities") == None for replica in self.replicas])): # are there velocities?
            [setattr(self.replicas[replica], "_currentVelocities",
                     np.multiply(self.replicas[replica]._currentVelocities, np.divide(original_T[i], swapped_T[i]))) for
             i, replica in enumerate(self.replicas)]



class HamiltonianReplicaExchange(ReplicaExchange):
    pass


