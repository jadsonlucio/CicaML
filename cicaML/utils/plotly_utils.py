import plotly.graph_objects as go


def hline(fig, xvalue, min_value, max_value, line_kwargs, **kwargs):
    fig.add_trace(
        go.Scatter(x=[min_value, max_value],
                   y=[xvalue, xvalue],
                   mode='lines',
                   line=dict(**line_kwargs),
                   showlegend=True,
                   **kwargs))
