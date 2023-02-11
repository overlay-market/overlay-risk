import plotly.express as px


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
    def funding_chart(self):
        self.df['Days'] = self.df.index
        self.df['Days'] = self.df['Days'].apply(
            lambda x: int(x.replace('n=', ''))/86400)
        self.df["Percentage of position paid as funding"] =\
            self.df['alpha=0.05'] * 2 * 3600 * 24
