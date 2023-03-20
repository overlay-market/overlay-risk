import plotly.express as px
import visualizations.data.viz_data_prep as vdp


class BarChart:
    def __init__(self, df, title, xcol, xname, ycol, yname,
                 clr=None, anim_frame=None, barmode='relative'):
        self.df = df
        self.title = title
        self.xcol = xcol
        self.xname = xname
        self.ycol = ycol
        self.yname = yname
        self.clr = clr
        self.anim_frame = anim_frame
        self.barmode = barmode

    def create_chart(self):
        # Make x axis a string to avoid unwanted math by plotly
        self.df[self.xcol] = self.df[self.xcol].apply(lambda x: str(x))

        # Sort values so they display in order
        if self.anim_frame:
            self.df = self.df.sort_values([self.xcol, self.anim_frame])
        else:
            self.df = self.df.sort_values([self.xcol])

        fig = px.histogram(
            self.df,
            x=self.xcol, y=self.ycol, color=self.clr,
            barmode=self.barmode, animation_frame=self.anim_frame)
        fig.update_layout(title=self.title)
        fig.update_layout(xaxis_title=self.xname, yaxis_title=self.yname)
        return fig


class SlidingBarChartImpact(BarChart):
    def __init__(
            self, df,
            title="Abs percentage change in price due to lambda and volume",
            xcol='perc_volume_y',
            xname='Volume as percentage of cap (over short TWAP)',
            ycol='perc_change', yname='Abs percentage change in price',
            clr='bid_ask', anim_frame='ls', barmode='group'
            ):
        super().__init__(df, title, xcol, xname, ycol, yname,
                         clr, anim_frame, barmode)

    def create_impact_chart(self):
        df_mlt = self.df.reset_index().melt(
            id_vars='index', var_name='q0', value_name='value'
        )
        df_mlt.columns = ['alpha', 'perc_volume', 'ls']

        # Remove prefixes and make numeric
        df_mlt = vdp.make_numeric(df_mlt, 'alpha=', 'alpha')
        df_mlt = vdp.make_numeric(df_mlt, 'q0=', 'perc_volume')

        # Get all volumes against lambdas
        df_mlt = df_mlt.merge(
            df_mlt.perc_volume.drop_duplicates(), how='cross')

        # Get bid and ask prices and percentage change from TWAP
        df_mlt = vdp.bid_ask_perc_change(df_mlt, 'ls', 'perc_volume_y')

        # Group bid and ask for grouped bar plot
        df_mlt = df_mlt.melt(
            id_vars=['alpha', 'perc_volume_x', 'ls',
                     'perc_volume_y', 'bid', 'ask'],
            var_name='bid_ask',
            value_name='perc_change'
        )
        self.df = df_mlt

        fig = self.create_chart()
        fig["layout"].pop("updatemenus")  # Optional; drop animation buttons
        return fig
