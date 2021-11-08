from cicaML.tsa.time_serie import TimeSeries


def get_low_stock_filter(sales_serie, stock_serie):
    mean_sales = sales_serie.remove_outliers().mean()
    filter_ = stock_serie < mean_sales * 7
    return filter_


def get_quant_high_low_sales(time_serie):
    def get_quant_low_sales(df, **kwargs):
        min_range, max_range = df.outliers_bounds(**kwargs)

        return sum(df.values < min_range)

    quant_low_sales = (
        time_serie.groupby(time_serie.index.weekday)
        .apply(lambda df: get_quant_low_sales(df, k1=0.5))
        .sum()
    )
    min_value, max_value = time_serie.outliers_bounds(k1=0.5)
    quant_high_sales = (time_serie > max_value).sum()

    return quant_low_sales, quant_high_sales


def get_lost_sales(sales_serie, stock_serie):

    # get sales mean and min bound when stock is high
    filter_low_stock = get_low_stock_filter(sales_serie, stock_serie)
    sales_serie_high_stock_index = sales_serie.index.intersection(
        stock_serie.index[~filter_low_stock]
    )
    sales_serie_high_stock = sales_serie[sales_serie_high_stock_index]
    weekday_high_stock_stats = sales_serie_high_stock.groupby(
        sales_serie_high_stock.index.weekday
    ).agg(
        media=lambda serie: serie.remove_outliers().mean(),
        minimo=lambda serie: serie.remove_outliers().mean()
        - 1.8 * serie.remove_outliers().std(),
    )

    # get sales when stock is low
    sales_low_stock_idx = sales_serie.index.intersection(
        stock_serie.index[filter_low_stock]
    )
    sales_serie_low_stock = sales_serie[sales_low_stock_idx]
    new_serie = {"data": [], "values": []}

    for idx, data, sales in zip(
        sales_low_stock_idx.weekday, sales_low_stock_idx, sales_serie_low_stock.values
    ):

        row = weekday_high_stock_stats.loc[idx]

        if sales < row["minimo"]:
            new_serie["data"].append(data)
            new_serie["values"].append(row["media"] - sales)

    ts = TimeSeries(
        index=new_serie["data"],
        data=new_serie["values"],
        name="Vendas diárias perdidadas",
    )
    return ts
