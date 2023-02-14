import helpers.helpers as helpers
import visualizations.line_chart as lc
import visualizations.bar_chart as bc


def main(file_name, df_ks, df_deltas, df_ls):
    results_name = file_name.replace('_treated', '')
    helpers.create_dir(results_name)
    results_path = helpers.get_results_dir()+results_name

    # Funding visualizations
    # Funding % Paid Daily for Various Anchor Times
    lc.LineChartFunding(df_ks)\
        .create_funding_chart()\
        .write_html(f"{results_path}/Daily funding per anchor time.html")

    # Spread visualizations
    # Percentage difference between bid and ask
    lc.LineChartSpread(df_deltas)\
        .create_spread_chart()\
        .write_html(f"{results_path}/Spread percentage difference.html")

    # Price impact visualizations
    # Effect of lambda
    bc.SlidingBarChartImpact(df_ls)\
        .create_impact_chart()\
        .write_html(f"{results_path}/Effect of lambda.html")
