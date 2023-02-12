import plotly.express as px
import visualizations.data.viz_data_prep as vdp


class LineChart:
    def __init__(self, df, title, xcol, ycol):
        self.df = df
        self.title = title
        self.xcol = xcol
        self.ycol = ycol

    def create_chart(self):
        fig = px.line(self.df[[self.xcol, self.ycol]],
                      x=self.xcol, y=self.ycol)
        fig.update_layout(title=self.title)
        fig.update_layout(xaxis_title=self.xcol, yaxis_title=self.ycol)
        return fig


class LineChartFunding(LineChart):
    def __init__(
            self, df,
            title="Funding % Paid Daily per Anchor Time (alpha = 0.05)",
            xcol='Days', ycol='Percentage of position paid as funding'):
        super().__init__(df, title, xcol, ycol)

    def create_funding_chart(self):
        self.df['Days'] = self.df.index
        self.df = vdp.make_numeric(self.df, 'n=', 'Days')
        self.df['Days'] /= 86400
        self.df["Percentage of position paid as funding"] =\
            self.df['alpha=0.05'] * 2 * 3600 * 24 * 100
        fig = self.create_chart()
        return fig


class LineChartSpread(LineChart):
    def __init__(
            self, df,
            title="Percentage difference b/w bid and ask for various alpha",
            xcol='alpha', ycol='spread_perc'):
        super().__init__(df, title, xcol, ycol)

    def create_spread_chart(self):
        bid = self.df['delta'].apply(lambda x: vdp.bid(100, x, 0, 0))
        ask = self.df['delta'].apply(lambda x: vdp.ask(100, x, 0, 0))
        self.df['spread_perc'] = (ask/bid - 1) * 100
        fig = self.create_chart()
        return fig
