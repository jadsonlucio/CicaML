import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from itertools import cycle, product
from plotly.subplots import make_subplots
from cicaML.utils.plotly.core import update_traces
from cicaML.utils.array import array2string, fill_matrix

COLOR_GENERATOR = cycle(px.colors.qualitative.Light24)


def set_col_color(df, color_column, color_generator=COLOR_GENERATOR):
    colors = {p: next(color_generator) for p in set(df[color_column])}
    df["color"] = df["nome_empresa"].map(lambda name: colors[name])

    return df


def create_grid_subplot(df, col=None, row=None, col_label=None, row_label=None):
    set_cols = list(dict.fromkeys(df[col].values)) if col else [None]
    set_rows = list(dict.fromkeys(df[row].values)) if row else [None]
    add_groud_cols = []
    if row:
        add_groud_cols.append(row)
    if col:
        add_groud_cols.append(col)

    subplot_titles = []
    row_title = row_label or row
    col_title = col_label or col
    for col_v, row_v in product(set_cols, set_rows):
        title = []
        if row:
            title.append(f"{row_title} - {row_v}")
        if col:
            title.append(f"{col_title} - {col_v}")

        subplot_titles.append(",".join(title))

    fig = make_subplots(
        rows=len(set_rows), cols=len(set_cols), subplot_titles=subplot_titles
    )

    return (fig, set_cols, set_rows, add_groud_cols)


def px_grid_plot(
    df,
    group_cols,
    trace_plot_func,
    extra_kwargs=None,
    col=None,
    row=None,
    col_label=None,
    row_label=None,
):
    """
    This function is going to ploy an grap grouping an df based on group_cols variable an plot an
    trace for eatch group in figure.
    """
    extra_kwargs = extra_kwargs or {}

    fig, set_cols, set_rows, add_group_cols = create_grid_subplot(
        df, col=col, row=row, col_label=col_label, row_label=row_label
    )
    group_cols = list(group_cols) + add_group_cols
    group = df.groupby(group_cols, as_index=False, sort=False)

    for key, df_group in group:
        idx_col = set_cols.index(key[group_cols.index(col)]) + 1 if col else 1
        idx_row = set_rows.index(key[group_cols.index(row)]) + 1 if row else 1
        fig.add_trace(
            trace_plot_func(key, df_group, **extra_kwargs), row=idx_row, col=idx_col
        )
    fig = update_traces(fig)

    return fig


def create_annotated_heatmap(
    matrix,
    x=None,
    y=None,
    colorscale=None,
    text=None,
    annotation_text=None,
    block_width=None,
    block_height=None,
    *args,
    **kwargs,
):
    layout_kwargs = {}
    if block_width:
        layout_kwargs["width"] = block_width * len(matrix[0]) + 200
    if block_height:
        layout_kwargs["height"] = block_height * len(matrix)

    if text is not None:
        text = array2string(fill_matrix(text))
    if annotation_text is not None:
        annotation_text = array2string(fill_matrix(annotation_text))

    fig = ff.create_annotated_heatmap(
        matrix,
        x=x,
        y=y,
        colorscale=colorscale,
        text=text,
        annotation_text=annotation_text,
        *args,
        **kwargs,
    )

    fig.update_layout(**layout_kwargs)

    return fig


def px_baxplot(df, y, *args, **kwargs):
    pass


def px_heatmap(
    df,
    row,
    col,
    values,
    hover_text=None,
    annot_text=None,
    colorscale="Oryel",
    *args,
    **kwargs,
):
    annot_text = annot_text or values
    annot_data, hover_data = None, None
    matrix_data = df.pivot(index=row, columns=col, values=values)
    if hover_text:
        hover_data = df.pivot(index=row, columns=col, values=hover_text).values
    if annot_text:
        annot_data = df.pivot(index=row, columns=col, values=annot_text).values

    fig = create_annotated_heatmap(
        matrix_data.values,
        y=list(matrix_data.index.values),
        x=list(matrix_data.columns.values),
        hoverongaps=False,
        colorscale=colorscale,
        text=hover_data,
        annotation_text=annot_data,
        showscale=True,
        *args,
        **kwargs,
    )
    fig.update_xaxes(side="top")

    return fig


def px_stacked_heatmap(
    df,
    row,
    values=None,
    annot_text=None,
    hover_text=None,
    colorscale="blues",
    data_func=None,
    extra_kwargs=None,
):
    data = []
    annot_data = []
    hover_data = []
    y_labels = []
    extra_kwargs = extra_kwargs or {}

    for y_label, df_g in df.groupby(row):
        data_row, annot_row, hover_row = None, None, None
        if data_func:
            data_row, annot_row, hover_row = data_func(y_label, df_g, **extra_kwargs)
        else:
            if values is None:
                raise Exception(
                    "values attribute must be passed when data_func isn't defined"
                )
            data_row = df_g[values].values
            annot_row = df_g[annot_text].values if annot_text else None
            hover_row = df_g[hover_text].values if hover_text else None

        y_labels.append(y_label)
        data.append(data_row)
        if annot_row is not None:
            annot_data.append(annot_row)
        if hover_row is not None:
            hover_data.append(hover_row)

    data = pd.DataFrame(data, dtype=object).fillna(np.nan).values
    fig = create_annotated_heatmap(
        data,
        y=y_labels,
        annotation_text=annot_data or None,
        text=hover_data or None,
        colorscale=colorscale,
        showscale=True,
    )

    return fig
