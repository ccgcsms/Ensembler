from collections.abc import Iterable
from scipy import constants as c
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from itertools import combinations


# eds_zwanzig
def calc_eds_zwanzig(V_is: List[Iterable], V_R: Iterable, undersampling_tresh: int = 0, temperature: float = 298, verbose: bool = False) -> (
Dict[Tuple[int, int], float], List[Tuple[float, float]]):
    """
        .. autofunction:: calculate_EDS_Zwanzig
        implementation from gromos++
        This function is calculating the relative Free Energy of multiple
        punish non sampling with "sampled mean"
        :param V_is:
    """
    # Todo: check same length! and shape
    # print(V_is[0].shape)
    # Todo: parse Iterables to np arrays or pandas.series

    if (verbose): print("params:")
    if (verbose): print("\ttemperature: ", temperature)
    if (verbose): print("\tundersampling_tresh: ", undersampling_tresh)

    # calculate:
    ##first exponent
    ##rowise -(V_i[t]-V_R[t]/(kb*temperature))
    exponent_vals = [np.multiply((-1 / (c.k * temperature)), (np.subtract(V_i, V_R))) for V_i in V_is]
    exponent_vals = [V_i.apply(lambda x: x if (x != 0) else 0.00000000000000000000000001) for V_i in V_is]

    ##calculate dF_i = avgerage(ln(exp(exponent_vals)))
    state_is_sampled = [[True if (t < undersampling_tresh) else False for t in V_i] for V_i in V_is]
    check_sampling = lambda sampled, state: float(np.std([V for (sam, V) in zip(sampled, state) if (sam)]))
    std_sampled = []
    for sampled, state in zip(state_is_sampled, V_is):
        std_sampled.append(check_sampling(sampled, state))

    sampling = [np.divide(np.sum(sam), len(sam)) for sam in state_is_sampled]
    extrema = [(np.min(np.log(np.exp(column))), np.max(np.log(np.exp(column)))) for column in exponent_vals]

    dF_t = [np.log(np.exp(column)) for column in exponent_vals]
    dF = {column.name: {"mean": np.mean(column), "std": np.std(column), "sampling": sampled_state, "std_sampled": std_sampl} for
          column, sampled_state, std_sampl in zip(dF_t, sampling, std_sampled)}

    if (verbose): print("\nDf[stateI]: (Mean\tSTD\tsampling)")
    if (verbose): print(
        "\n".join([str(stateI) + "\t" + str(dFI["mean"]) + "\t" + str(dFI["std"]) + "\t" + str(dFI["sampling"]) for stateI, dFI in dF.items()]))

    ##calculate ddF_ij and estimate error with Gauss
    ddF = {tuple(sorted([stateI, stateJ])): {"ddF": np.subtract(dF[stateI]["mean"], dF[stateJ]["mean"]), "gaussErrEstmSampled": np.sqrt(
        np.square(dF[stateI]["std_sampled"]) + np.square(dF[stateJ]["std_sampled"])),
                                             "gaussErrEstm": np.sqrt(np.square(dF[stateI]["std"]) + np.square(dF[stateJ]["std"]))} for stateI, stateJ
           in combinations(dF, r=2)}
    if (verbose): print("\nDDF[stateI][stateJ]\t diff \t gaussErrEstm")
    if (verbose): print(
        "\n\t".join([str(stateIJ) + ": " + str(ddFIJ["ddF"]) + " +- " + str(ddFIJ["gaussErrEstm"]) for stateIJ, ddFIJ in ddF.items()]))
    if (verbose): print()
    return ddF, dF