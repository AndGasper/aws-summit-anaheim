# Take the consecutive differences of the data
raw_values = series.values
diff_series = series.diff().fillna(0)


# Use the timeseries_to_supervised function from Lab 1 to reshape data from a time series to supervised observations 
def timeseries_to_supervised(data, look_back=1):
    # FILL IN FROM ABOVE LAB 1
    return '<To complete as challenge>'

assert timeseries_to_supervised(diff_series)!='<To complete as challenge>', 'Challenge not completed'

look_back = 1
supervised = '<To complete as challenge>' # To complete as challenge
assert supervised!='<To complete as challenge>', 'Challenge not completed'

supervised_values = supervised.values
supervised.head()

import pandas as pd
import numpy as np

def timeseries_to_supervised(data, look_back=1):
    df = pd.DataFrame(data).copy()
    columns = []
    # np.arrange => ?
    for i in np.arange(look_back,-1,-1):
        df_i = df.shift(i).copy()
        # pd.Series(df_i.columns)
        df_i.columns = pd.Series(df_i.columns).map(lambda x: x+' (t-'+str(i)+')' if i>0 else x+' (t)' ).ravel()
        columns.append(df_i)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df



import pandas as pd
csvList = ['s3://deutsche-boerse-xetra-pds/2018-10-10/2018-10-10_BINS_XETR13.csv',
           's3://deutsche-boerse-xetra-pds/2018-10-10/2018-10-10_BINS_XETR14.csv',
           's3://deutsche-boerse-xetra-pds/2018-10-10/2018-10-10_BINS_XETR15.csv']

raw = pd.concat([pd.read_csv(f, error_bad_lines=False, warn_bad_lines=False) for f in csvList], ignore_index = True)

dt = raw.iloc[0]['Date']
raw.drop(raw.index[raw['Date']!=dt], inplace=True)
raw['DateTime'] = pd.to_datetime(raw['Date'] + ' ' + raw['Time'])
raw.set_index('DateTime', inplace=True)
raw.head()



import pandas as pd
csvList = ['s3://deutsche-boerse-xetra-pds/2018-10-10/2018-10-10_BINS_XETR13.csv',
           's3://deutsche-boerse-xetra-pds/2018-10-10/2018-10-10_BINS_XETR14.csv',
           's3://deutsche-boerse-xetra-pds/2018-10-10/2018-10-10_BINS_XETR15.csv']

raw = pd.concat([pd.read_csv(f, error_bad_lines=False, warn_bad_lines=False) for f in csvList], ignore_index = True)
# print("raw", raw)
# Date 
dt = raw.iloc[0]['Date']
# print("dt", dt)
raw.drop(raw.index[raw['Date']!=dt], inplace=True)
raw['DateTime'] = pd.to_datetime(raw['Date'] + ' ' + raw['Time'])
raw.set_index('DateTime', inplace=True)
print(type(raw))
# return the first n rows
raw.head()