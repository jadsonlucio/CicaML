import plotly.graph_objects as go
from functools import reduce


def update_traces(fig: go.Figure, color_palette=None):
    """
    Update the traces to figure keeping traces with same legend equals
    """

    def trace_update(trace):
        trace_name = trace.name
        if trace_name not in names:
            names.add(trace_name)
            trace.update(showlegend=True)
            trace_properties[trace_name] = {
                "legendgroup": trace_name,
                "marker": trace["marker"],
            }
        else:
            trace.update(showlegend=False)

        trace.update(**trace_properties[trace_name])

    names = set()
    trace_properties = {}
    fig.for_each_trace(trace_update)

    return fig


def plot_traces(fig, traces_data, func_plot):
    fig = reduce(lambda fig, trace_data: func_plot(fig, trace_data), traces_data, fig)
    fig = update_traces(fig)

    return fig


def dropdown_plot(dict_figs):
    pass
