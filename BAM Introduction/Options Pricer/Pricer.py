import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from tqdm import tqdm


class Pricer:
    def __init__(self, risk_free_rate, S0, drift, vol, strike, maturity):
        self.S0 = S0
        self.volatility = vol
        self.risk_free_rate = risk_free_rate
        self.drift = drift
        self.K = strike
        self.T = maturity

    def BarrierOptionPricer(self, option_type="C", barrier=1.2, barrier_type="I", direction="U", N=10, NTS=1000):
        """

        :param barrier: Moneyness of the barrier (expressed as a percentage of the current asset price)
        :param option_type: Type of option. C: Call, P: Put
        :param barrier_type: Type of barrier. I: In, O: Out
        :param direction: Trigger Direction. U: Up, D: Down
        :param N: Number of simulations
        :param NTS: Number of time-steps

        TODO: This method compute the whole path, we can do the same thing but only saving last realization
        of stock price.

        :return: Fair Value of Barrier option
        """
        # Step 1: Simulate asset paths.
        # Realization array will be a matrix, every column will be an asset path.
        realization_array = np.zeros((NTS, N))
        cumulated_payoffs = 0
        barrier_value = barrier * self.S0
        # We will start at S0 for every simulation
        realization_array[0, :] = [self.S0 for simulation in range(N)]
        dt = self.T / NTS
        for simulation in tqdm(range(N)):
            random_variable = np.random.normal(0, 1, NTS)

            # If the barrier is an out type de option will be executed until we face the limit.
            if barrier_type == "O":
                executed = True
            # ...but if the option is an in-type the option will not be executed until we face the barrier.
            else:
                executed = False
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
                                                                          timestep, simulation] >= barrier_value)):
                    # If it is an up-and-out option, and we trespass the barrier: we can stop the simulation.
                    executed = False
                    break
                elif np.logical_and(barrier_type == "O", np.logical_and(direction == "D",
                                                                        realization_array[
                                                                            timestep, simulation] <= barrier_value)):
                    # If it is a down-and-out option, and we trespass the barrier: we can stop the simulation.
                    executed = False
                    break
                if np.logical_and(barrier_type == "I", np.logical_and(direction == "U",
                                                                      realization_array[
                                                                          timestep, simulation] >= barrier_value)):
                    # If it is an up-and-in option, and we trespass the barrier: keep track of this.
                    executed = True
                elif np.logical_and(barrier_type == "I", np.logical_and(direction == "D",
                                                                        realization_array[
                                                                            timestep, simulation] <= barrier_value)):
                    # If it is a down-and-in option, and we trespass the barrier: keep track of this.
                    executed = True
                if timestep == NTS - 1:
                    # If it is an IN type option and the barrier has been trespassed.
                    if np.logical_and(option_type == "C", executed):
                        cumulated_payoffs += max(0, realization_array[timestep, simulation] - self.K)
                    elif np.logical_and(option_type == "P", executed):
                        cumulated_payoffs += max(0, self.K - realization_array[timestep, simulation])
        value = math.exp(-self.risk_free_rate * self.T) * 1 / N * cumulated_payoffs
        return value


if __name__ == "__main__":
    p = Pricer(0.05, 100, 0.05, 0.2, 100, 1)
    v = p.BarrierOptionPricer(option_type="P", barrier=0.8, barrier_type="O", direction="D", NTS=1000)
    print(v)
