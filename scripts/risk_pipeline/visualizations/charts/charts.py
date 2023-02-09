import plotly.express as px


def line_chart(df, title, chartname, xcol, ycol, path):
    # Plot feed
    fig = px.line(df[[xcol, ycol]], x=xcol, y=ycol)
    fig.update_layout(title=title)
    fig.update_layout(xaxis_title=xcol, yaxis_title=ycol)
    fig.write_html(f"{path}/{chartname}.html")


def slider_grouped_bar_chart(df, title, path, chartname, xcol,
                             ycol, grp, slider, x_name, y_name):
    '''
    Grouped bar chart with slider
    '''
    # Make x axis a string to avoid unwanted math by plotly
    df[xcol] = df[xcol].apply(lambda x: str(x))
    # Sort values so they display in order
    df = df.sort_values([xcol, slider])
    fig = px.histogram(
        df,
        x=xcol, y=ycol, color=grp,
        barmode='group', animation_frame=slider
    )
    fig["layout"].pop("updatemenus")  # Optional; drop animation buttons
    fig.update_layout(xaxis_title=x_name, yaxis_title=y_name)
    fig.update_layout(title=title)
    fig.write_html(f"{path}/{chartname}.html")
