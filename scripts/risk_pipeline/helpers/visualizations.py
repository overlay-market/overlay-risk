import plotly.express as px


def time_series(df, title, chartname, xcol, ycol, path):
    # Plot feed
    fig = px.line(df[[xcol, ycol]], x=xcol, y=ycol)
    fig.update_layout(title=title)
    fig.update_layout(xaxis_title=xcol, yaxis_title=ycol)
    fig.write_html(f"{path}/{chartname}.html")
