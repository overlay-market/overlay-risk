import pandas as pd
import numpy as np
from datetime import timedelta


def missing_candlesticks(df, freq, time_col, price_col, filename, path):
    df[time_col] = pd.to_datetime(df[time_col])
    start_date = pd.to_datetime(df[time_col].min())
    end_date = pd.to_datetime(df[time_col].max())

    timestamp_list = []
    current = start_date
    while current < end_date:
        timestamp_list.append(current)
        current += timedelta(minutes=freq/60)

    time_df = pd.DataFrame({time_col: np.array(timestamp_list)})
    breakpoint()
    df = time_df.merge(df, how='left', on=time_col)

    # Report info on missing values
    df_nulls = df.join(df[price_col].notnull().astype(int).cumsum(),
                       rsuffix='_grp')
    null_report = df_nulls.groupby('close_grp').agg(
        {time_col: ['min', 'max', 'count'],
         price_col: ['min', 'max']})
    null_report = null_report[null_report[time_col]['count'] > 1]
    null_report.reset_index(inplace=True)

    for i in range(null_report.shape[0]):
        idx = df[df[time_col] == null_report[time_col]['max'].iloc[i]].index+1
        null_report.at[i, (price_col, 'max')] = df.iloc[idx, :][price_col]
    null_report.columns = null_report.columns.get_level_values(1)
    null_report.drop('', axis=1, inplace=True)
    null_report.columns = ['start_time', 'end_time', 'null_row_count',
                           'pre_null_price', 'post_null_price']
    null_report.start_time += pd.Timedelta(minutes=freq/60)
    null_report.null_row_count -= 1
    null_report['percentage_change'] =\
        abs((null_report.post_null_price/null_report.pre_null_price) - 1) * 100
    null_report.sort_values('percentage_change', ascending=False, inplace=True)
    null_report.to_csv(f'{path}/{filename}_missing_values.csv')
    return df
