import argparse
import numpy as np
from pathlib import Path

# Import functions from the existing modules
from funding2 import main as funding_main
from impact2 import analyze_data as impact_analyze
from liquidations2 import main as liquidations_main
from pricedrift2 import main as pricedrift_main
from caps2 import main as caps_main

class RiskParameters:
    """
    Class to handle the running of all risk parameter calculations.

    Attributes:
        market_name (str): The name of the market being analyzed.
        data_file (str): Path to the data file.
        alpha (float): Confidence level for the risk calculations.
        t (int): Periodicity in seconds (default: 86400 seconds = 1 day).
        cp (int): Payoff cap.
        st (int): Shorter TWAP in seconds.
    """

    market_name = "Super"
    data_file = r"C:\Users\HP\bytes.csv"
    alpha = 0.01
    t = 86400  # Periodicity in seconds (default: 86400 seconds = 1 day)
    cp = 5  # Payoff cap
    st = 600  # Shorter TWAP in seconds

    @classmethod
    def run_all(cls):
        """
        Run all modules and generate a summary of the risk parameters.

        Returns:
            dict: A summary report of all the risk parameters.
        """
        summary = {}

        # Run Funding module
        print("\nRunning Funding Module:")
        funding_results = funding_main(cls.data_file, cls.alpha)
        if funding_results is not None:
            n_7_value = funding_results.loc["n=7.0 days"].item()  # Convert to plain number
            summary["Funding K: (alpha=0.05, n=7 days)"] = n_7_value

        # Run Impact module
        print("\nRunning Impact Module:")
        impact_deltas, impact_lambdas = impact_analyze(cls.data_file, cls.t, cls.cp, cls.st, cls.alpha)
        if impact_deltas is not None and impact_lambdas is not None:
            delta_value = impact_deltas.loc[impact_deltas['alpha'] == cls.alpha, 'delta'].values[0]
            lambda_value = impact_lambdas.loc[f"alpha={cls.alpha}", f"q0={cls.alpha}"]
            summary["Spread: (alpha=0.05, Delta)"] = delta_value
            summary["Impact: (q0=0.05, alpha=0.05, Lambda)"] = lambda_value

        # Run Liquidations module
        print("\nRunning Liquidations Module:")
        liquidations_results = liquidations_main(cls.data_file, cls.alpha)
        if liquidations_results is not None:
            mm_value = liquidations_results.loc[liquidations_results['time_frame_seconds'] == 14400, 'mm_alpha=0.05_beta=1.0_mu=-1.237435875750953e-07_sigma=0.00018023744339323657'].values[0]
            beta_value = liquidations_results.loc[liquidations_results['time_frame_seconds'] == 14400, 'beta_alpha=0.05_beta=1.0_mu=-1.237435875750953e-07_sigma=0.00018023744339323657'].values[0]
            summary["Maintenance margin: (time=14400s, mm)"] = mm_value
            summary["Beta MM: (time=14400s, Beta)"] = beta_value

        # Run Pricedrift module
        print("\nRunning Pricedrift Module:")
        pricedrift_results = pricedrift_main(cls.data_file, cls.alpha)
        if pricedrift_results is not None:
            mu_max_value = pricedrift_results.loc[0, 'mu_max']
            summary["Price Drift Upper Limit: (alpha=0.05, Mu Max)"] = mu_max_value

        return summary

def main():
    print("Running the main function...")

    # Run all the risk parameter calculations and get the summary
    summary = RiskParameters.run_all()

    # Print the summary report
    print("\n=== Summary Report ===")
    print(f"Market: {RiskParameters.market_name}")
    print("TWAP: Shorter TWAP = 10 minutes, Longer TWAP = 1 hour\n")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # If you need to save the summary to a file, you could do so here
    # For example:
    # with open("summary_report.txt", "w") as f:
    #     f.write(f"Market: {RiskParameters.market_name}\n")
    #     f.write("TWAP: Shorter TWAP = 10 minutes, Longer TWAP = 1 hour\n")
    #     for key, value in summary.items():
    #         f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()