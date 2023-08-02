import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from tqdm import tqdm


class Pricer:
    def __init__(self, S0, drift, vol, strike, maturity):
        self.S0 = S0
        self.volatility = vol
        self.drift = drift
        self.K = strike
        self.T = maturity

    def BarrierOptionPricer(self, option_type="C", barrier=1.2, barrier_type="I", direction="U", N=10000, NTS=1000):
        """

        :param barrier: Moneyness of the barrier (expressed as a percentage of the current asset price)
        :param option_type: Type of option. C: Call, P: Put
        :param barrier_type: Type of barrier. I: In, O: Out
        :param direction: Trigger Direction. U: Up, D: Down
        :param N: Number of simulations
        :param NTS: Number of time-steps

        :return: Fair Value of Barrier option
        """
        # Step 1: Simulate asset paths.
        # Realization array will be a matrix, every column will be an asset path.
        realization_array = np.zeros((NTS, N))
        payoffs = [0 for i in range(N)]
        barrier_value = barrier * self.S0
        # We will start at S0 for every simulation
        realization_array[0, :] = [self.S0 for simulation in range(N)]
        dt = self.T / NTS
        for simulation in tqdm(range(N)):
            random_variable = np.random.normal(0, 1, NTS)
            trespassed = False
            for timestep in range(1, NTS):
                realization_array[timestep, simulation] = realization_array[timestep - 1, simulation] \
                                                          * math.exp((self.drift - 1 / 2 * self.volatility ** 2) * dt
                                                                     + self.volatility * math.sqrt(dt) *
                                                                     random_variable[
                                                                         timestep])
                # If it is an out option, let's check barrier is not trespassed (if trespassed we can stop the
                # simulation)
                if np.logical_and(barrier_type == "O", np.logical_and(direction == "U",
                                                                      realization_array[
                                                                          timestep, simulation] >= barrier)):
                    # If it is an up-and-out option, and we trespass the barrier: we can stop the simulation.
                    trespassed = True
                    break
                elif np.logical_and(barrier_type == "O", np.logical_and(direction == "D",
                                                                        realization_array[
                                                                            timestep, simulation] <= barrier)):
                    # If it is a down-and-out option, and we trespass the barrier: we can stop the simulation.
                    trespassed = True
                    break
                if np.logical_and(barrier_type == "I", np.logical_and(direction == "U",
                                                                      realization_array[
                                                                          timestep, simulation] >= barrier)):
                    # If it is an up-and-in option, and we trespass the barrier: keep track of this.
                    trespassed = True
                elif np.logical_and(barrier_type == "I", np.logical_and(direction == "D",
                                                                          realization_array[
                                                                              timestep, simulation] <= barrier)):
                    # If it is a down-and-in option, and we trespass the barrier: keep track of this.
                    trespassed = True

        return realization_array

        pass


def simulate_asset_path(S0, risk_free_rate, volatility, NTS, T, N=10000):
    """
    Simulating realizations of log-normal risk-neutral random walk

    :param S0: Initial asset price
    :param risk_free_rate: risk-free spot rate
    :param volatility: historical vol
    :param NTS: number of time steps
    :param T: Derivatives expiration
    :param N: Number of realizations

    :return:  2D array (NTS x N) having asset prices for every simulation.
    """
