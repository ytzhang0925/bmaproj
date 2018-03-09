import pygrib


def get_model_file_list():
    """Return the list of model data files."""
    # TODO
    pass
    return file_list


def extract_stn_from_file():
    """Extract station info from csv file."""
    import glob
    import pandas as pd
    # list all files corresponding to format 'national_stn_table_*.csv', sort,
    # then pick the last (newest) one.
    stn_file_list = glob.glob('national_stn_table_*.csv')
    stn_file_list.sort()
    df = pd.read_csv(stn_file_list[-1])

    # 2018.3.9: Add df_stns filtering
    df = df[(df['longitude'] >= 110) & (df['longitude'] <= 123)
            & (df['latitude'] >= 30) & (df['latitude'] <= 35)]
    return df


def read_from_single_grbmsg(msg):
    """Read from a single grib message, do resizing, reverse latitude
    order, finally extract lat, lon, and values."""
    from readdata import grbmsg_dict_subset_region
    grbmsg_dict = grbmsg_dict_subset_region(msg, la0=30, la1=35, lo0=110, lo1=123)
    return grbmsg_dict['lat'], grbmsg_dict['lon'], grbmsg_dict['values']


def stn_interpolate(model_file_dic, df_stns):
    """Interpolate model forecast to stations.

    @Variables:
    model_file_list -- A list of model data.
    df_stns -- A pandas.DataFrame of national stations."""
    from scipy.interpolate import interp2d
    import numpy as np
    import pandas as pd
    from functools import reduce
    from collections import defaultdict
    # Get latlon from stations
    lats = df_stns['latitude']
    lons = df_stns['longitude']  # Spelling mistake
    ids = df_stns['id']
    stn_ids = df_stns['stationId']
    interp_results = defaultdict(lambda: list())
    df_results = dict()
    assert len(lats) == len(lons)
    # Loop over the file list, for each file, loop over a list of
    # **shortName, perturbationNumber** and do interpolation to each station.
    # short_name_list = ['tp', '2t', '2d']
    for model, model_file_list in model_file_dic.items():
        for model_file in model_file_list:
            grbmsg = pygrib.open(model_file)[1]
            # Unit conversion
            units_grbmsg = grbmsg['parameterUnits']
            short_name = 'PRCP'
            lat, lon, values = read_from_single_grbmsg(grbmsg)
            if units_grbmsg == 'm':
                values = values * 1000.
            elif units_grbmsg != 'kg m-2':
                print('Other kinds of precipitation units detected: {}'.format(units_grbmsg))

            # Get validity datetime
            validity_datetime = "{:08d}{:02d}".format(grbmsg['validityDate'], grbmsg['validityTime'])
            initial_datetime = "{:08d}{:02d}".format(grbmsg['dataDate'], grbmsg['dataTime'])
            f = interp2d(lon, lat, values)  # bounds_error=True
            var_name = "".join([short_name, '.', model])
            # Do interpolation
            temp_interp_result = np.asarray([f(lo, la)[0] for lo, la in zip(lons, lats)])
            temp_interp_result[temp_interp_result < 0.1] = 0.0
            interp_results[validity_datetime].append(pd.DataFrame({var_name: temp_interp_result,
                                                                   'stn_id': stn_ids}))
    for key_date in interp_results.keys():
        df_results[key_date] = reduce(lambda left, right: pd.merge(left, right, on='stn_id'),
                                      interp_results[key_date])
        # Add observation data to df_results according to key_date
        df_obs = get_station_data(key_date)
        df_results[key_date] = df_results[key_date].merge(df_obs[['datetime', 'stationId', 'PRE_24h']],
                                                          left_on='stn_id', right_on='stationId')
        df_results[key_date] = df_results[key_date].merge(df_stns[['stationId', 'latitude', 'longitude']],
                                                          left_on='stn_id', right_on='stationId', how='left')
        df_results[key_date]['time'] = key_date
    df_results = pd.concat([value for value in df_results.values()])
    df_results.merge(df_stns[['id', 'stationId']], left_on='stn_id', right_on='stationId')
    return df_results


def get_station_data(date_yyyymmddhh):
    """Get station data from huaxin goubi API.
    1. Query all data at one time.
    2. Match to stations"""
    import pandas as pd
    from math import isclose
    import numpy as np
    date_yyyymmdd = date_yyyymmddhh[0:8]
    date_hh = date_yyyymmddhh[8:10]
    url = 'https://data.huaxin-hitec.com/v1/surf/SurfHourData/' \
          'surf_chn_n&PRE_24h&{}&{}&%E5%85%A8%E5%9B%BD&0,3000/surfhour.csv'.format(date_yyyymmdd, date_hh)
    df_obs = pd.read_csv(url)
    # Filtering prcp data
    # 999999: not observed
    df_obs.loc[abs(df_obs['PRE_24h'] - 999999.0) < 1e-5, 'PRE_24h'] = np.nan
    # 999990: 微量
    df_obs.loc[abs(df_obs['PRE_24h'] - 999990.0) < 1e-5, 'PRE_24h'] = 0

    def convert_datetime(dtstr_input):
        """The input datetime string is of format 2018-1-29 08:00:00,
        Convert to 2018012908"""
        from datetime import datetime
        dt = datetime.strptime(dtstr_input, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%Y%m%d%H')
    df_obs['datetime'] = list(map(convert_datetime, df_obs['datetime']))
    return df_obs


def grib2csv(file_dict):
    r"""Convert a dict of grib files to a CSV."""
    pass
    import numpy as np
    from collections import defaultdict
    import pandas as pd
    from scipy.interpolate import interp2d
    from functools import reduce
    # Resolution of lat and lon on the uniformly interpolated grid.
    lat_uniform = np.arange(30, 35, 0.125)
    lon_uniform = np.arange(110, 123, 0.125)
    pivoted_dataframes = defaultdict(dict)  # Dimensions: (Time, Model, (Lat, Lon))
    unpivoted_dataframes = dict()  # Dimensions: (Time, DataFrame: (Model, lat, lon))
    for model, model_file_list in file_dict.items():
        for model_file in model_file_list:
            # Open and read file
            grbmsg = pygrib.open(model_file)[1]
            lat, lon, values = read_from_single_grbmsg(grbmsg)

            # Unit conversion
            units_grbmsg = grbmsg['parameterUnits']
            short_name = 'PRCP'
            if units_grbmsg == 'm':
                values = values * 1000.
            elif units_grbmsg != 'kg m-2':
                print('Other kinds of precipitation units detected: {}'.format(units_grbmsg))

            # Get validity datetime
            initial_datetime = "{:08d}{:02d}".format(grbmsg['validityDate'], grbmsg['validityTime'])
            validity_datetime = "{:08d}{:02d}".format(grbmsg['dataDate'], grbmsg['dataTime'])

            # Do 2d-interpolation on all data to uniform grids.
            f = interp2d(lon, lat, values)  # TODO: May modify parameter 'kind' later
            values_uniform = f(lon_uniform, lat_uniform)
            values_uniform[values_uniform < 0.1] = 0.0

            # TODO:
            # Dimensions: (Time, Model, (Lat, Lon))
            # 1. Put (lat, lon) data into DataFrames with lat, lon axes.
            pivoted_dataframes[validity_datetime][model] = pd.DataFrame(values_uniform,
                                                                        lat_uniform, lon_uniform)
            # 2. Do melting ("Unpivoting") to the DataFrames.
            pivoted_dataframes[validity_datetime][model]['lat'] = pivoted_dataframes[validity_datetime][model].index
            pivoted_dataframes[validity_datetime][model] = pd.melt(pivoted_dataframes[validity_datetime][model],
                                                                   id_vars=['lat'], var_name='lon',
                                                                   value_name='prcp.{}'.format(model))

    # 3. Merge them together as one DataFrame
    for key_date in pivoted_dataframes.keys():
        unpivoted_dataframes[key_date] = reduce(lambda left, right: pd.merge(left, right, on=['lat', 'lon']),
                                                pivoted_dataframes[key_date].values())
        unpivoted_dataframes[key_date]['time'] = key_date
    unpivoted_dataframes = pd.concat([value for value in unpivoted_dataframes.values()])
    return unpivoted_dataframes


def main():
    import glob
    # Generate training data
    df_stns = extract_stn_from_file()
    files_ec = glob.glob('./data2/*ECMF*')
    files_grapes = glob.glob('./data2/*GRAPES*')
    files_t639 = glob.glob('./data2/*T639*')
    file_dic = {'EC': files_ec, 'GRAPES': files_grapes, 'T639': files_t639}
    df_training = stn_interpolate(file_dic, df_stns)
    # print(df_training)
    df_training.to_csv('training_data.csv')

    # Convert nowcast into CSV
    files_ec = glob.glob('./data2/*ECMF*')
    files_grapes = glob.glob('./data2/*GRAPES*')
    files_t639 = glob.glob('./data2/*T639*')
    file_dict = {'EC': files_ec, 'GRAPES': files_grapes, 'T639': files_t639}
    df_nowcast = grib2csv(file_dict)
    df_nowcast.to_csv('nowcast.csv')


if __name__ == '__main__':
    main()
