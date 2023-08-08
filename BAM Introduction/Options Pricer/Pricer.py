import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm


class Pricer:
    def __init__(self, risk_free_rate, S0, drift, vol, strike, maturity, dividend_yield=None):
        self.S0 = S0
        self.volatility = vol
        self.risk_free_rate = risk_free_rate
        self.drift = drift
        self.K = strike
        self.T = maturity
        if not dividend_yield:
            self.dividend_yield = 0
        else:
            self.dividend_yield = dividend_yield

    def BarrierOptionPricer(self, option_type="C", barrier=1.2, barrier_type="I", direction="U", N=1000, NTS=1000):
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
        prices = np.zeros(N)
        cumulated_payoffs = 0
        barrier_value = barrier * self.S0
        # We will start at S0 for every simulation
        realization_array[0, :] = [self.S0 for simulation in range(N)]
        prices = [self.S0 for simulation in range(N)]
        dt = self.T / NTS
        for simulation in tqdm(range(N)):
            random_variable = np.random.normal(0, 1, NTS)

            # If the barrier is an out type de option will be executed until we face the limit.
            if barrier_type == "O":
                executed = True
            # ...but if the option is an in-type the option will not be executed until we face the barrier.
            else:
                executed = False
            for timestep in range(NTS):
                # We computed new realized price
                if timestep >= 1:
                    prices[simulation] = prices[simulation] \
                                         * math.exp((self.drift - 1 / 2 * self.volatility ** 2) * dt
                                                    + self.volatility * math.sqrt(dt) *
                                                    random_variable[
                                                        timestep])
                # If it is an out option, let's check barrier is not trespassed (if trespassed we can stop the
                # simulation)
                if np.logical_and(barrier_type == "O",
                                  np.logical_and(direction == "U",
                                                 prices[simulation] >= barrier_value)):
                    # If it is an up-and-out option, and we trespass the barrier: we can stop the simulation.
                    executed = False
                    break
                elif np.logical_and(barrier_type == "O",
                                    np.logical_and(direction == "D",
                                                   prices[simulation] <= barrier_value)):
                    # If it is a down-and-out option, and we trespass the barrier: we can stop the simulation.
                    executed = False
                    break
                if np.logical_and(barrier_type == "I",
                                  np.logical_and(direction == "U",
                                                 prices[simulation] >= barrier_value)):
                    # If it is an up-and-in option, and we trespass the barrier: keep track of this.
                    executed = True
                elif np.logical_and(barrier_type == "I",
                                    np.logical_and(direction == "D",
                                                   prices[simulation] <= barrier_value)):
                    # If it is a down-and-in option, and we trespass the barrier: keep track of this.
                    executed = True
                if timestep == NTS - 1:
                    # If it is an IN type option and the barrier has been trespassed.
                    if np.logical_and(option_type == "C", executed):
                        cumulated_payoffs += max(0, prices[simulation] - self.K)
                    elif np.logical_and(option_type == "P", executed):
                        cumulated_payoffs += max(0, self.K - prices[simulation])
        value = math.exp(-self.risk_free_rate * self.T) * 1 / N * cumulated_payoffs
        return value

    def DigitalOptionPricer(self, option_type="C"):
        """
        The aim of this code is to price a digital option with the pricer characteristics.

        :param option_type: C or P for call and put.

        :return: Price of the digital option with the pricer characteristics.
        """
        d2 = (math.log(self.S0 / self.K) + (
                    self.risk_free_rate - self.dividend_yield - 0.5 * self.volatility ** 2) * self.T) / (
                         self.volatility * math.sqrt(self.T))
        if option_type == "C":
            price = math.exp(-self.risk_free_rate * self.T) * norm.cdf(d2)
        elif option_type == "P":
            price = math.exp(-self.risk_free_rate * self.T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'C' or 'P' for call and put respectively.")

        return price


if __name__ == "__main__":
    # Printing a barrier price for different spots and same barrier level
    multiple_spots = np.linspace(80, 120, 50)
    barrier_prices = []
    digital_prices = []
    for spot in multiple_spots:
        print("Spot: ", spot)
        p = Pricer(0.05, spot, 0.05, 0.2, 100, 1)
        barrier_prices.append(
            p.BarrierOptionPricer(option_type="P", barrier=(100 / spot) * 0.8,
                                                                      barrier_type="O",
                                                                      direction="D", NTS=1000))
        digital_prices.append(p.DigitalOptionPricer(option_type = "P"))
    plt.plot(multiple_spots, barrier_prices)
    plt.title("1Y Down-and-Out Put (Barrier = 80, K= 100) prices")
    plt.show()
    plt.plot(multiple_spots, digital_prices)
    plt.title("1Y Digital Put (K = 100) prices")
    plt.show()
    print(barrier_prices)
    print(digital_prices)
