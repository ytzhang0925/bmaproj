def main():
    """Get airport list from MySQL or other text files.
    Read from MySQL cimiss table.

    This program is separated from the main program. It should only be executed
    when we need to renew the national station list."""
    # import MySQLdb
    # db = MySQLdb.connect(host='59.110.30.137', user='hx',
    #                      passwd='Hx82101401#db', db='cimiss')
    # cursor = db.cursor()
    # sql = 'SELECT * from airportinfo'
    from sqlalchemy import create_engine
    import pandas as pd
    from datetime import datetime

    dt = datetime.now()
    dt_ymdhm = dt.strftime('%Y%m%d%H%M')
    engine = create_engine('mysql://hx:Hx82101401#db@59.110.30.137/cimiss')
    sql = 'SELECT * from sta_info_surf_chn_n'
    df = pd.read_sql_query(sql, engine)
    df = df.loc[df['isLegal'] == 1]
    file_name = ''.join(('national_stn_table_', dt_ymdhm, '.csv'))
    df.to_csv(file_name)


if __name__ == '__main__':
    main()
