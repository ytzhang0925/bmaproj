import pygrib


class InvalidObjError(Exception):
    """Raises the error when function get_index gets an invalid input obj."""

    def __init__(self, expression):
        self.expression = expression


def get_index(obj, query_item):
    """Get the index of query_item in a list or numpy array, raises Error when
    getting an invalid object."""
    import numpy as np
    # First check type of input obj, should be either list or nparray
    if isinstance(obj, list):
        return obj.index(query_item)
    elif isinstance(obj, np.ndarray):
        # Should be a 1D numpy array
        if len(obj.shape) == 1:
            return np.where(obj == query_item)[0][0]
        else:
            raise InvalidObjError("Invalid object for getting index.")
    else:
        raise InvalidObjError("Invalid object for getting index.")


def grbmsg_latlon(msg):
    """Return a tuple of (lat, lon) according to the input grbmsg."""
    from numpy import linspace
    lat = linspace(msg['latitudeOfFirstGridPointInDegrees'],
                   msg['latitudeOfLastGridPointInDegrees'], msg['Nj'])
    lon = linspace(msg['longitudeOfFirstGridPointInDegrees'],
                   msg['longitudeOfLastGridPointInDegrees'], msg['Ni'])
    return lat, lon


def grbmsg_resolve(msg):
    """Return a tuple of (values, lat, lon) according to the input grbmsg."""
    from numpy import linspace
    lat = linspace(msg['latitudeOfFirstGridPointInDegrees'],
                   msg['latitudeOfLastGridPointInDegrees'], msg['Nj'])
    lon = linspace(msg['longitudeOfFirstGridPointInDegrees'],
                   msg['longitudeOfLastGridPointInDegrees'] % 360.0, msg['Ni'])
    values = msg.values
    return values, lat, lon


def get_2d_vis_data(filename):
    """Get the 2D visibility data of a GFS grib file."""
    msgs0 = pygrib.index(filename, 'shortName', 'typeOfLevel')
    msgs_vis = msgs0.select(shortName='vis', typeOfLevel='surface')
    assert(len(msgs_vis)) == 1
    msg_vis = msgs_vis[0]
    del msgs_vis
    forecasttime = msg_vis['startStep']
    msg_vis_dict = grbmsg_dict_subset_region(msg_vis)
    lat = msg_vis_dict['lat']
    lon = msg_vis_dict['lon']
    return forecasttime, lat, lon, msg_vis_dict['values']


def get_3d_data(filename):
    """Get 3D data from ONE grib file."""
    from operator import attrgetter
    import numpy as np
    msgs0 = pygrib.index(filename, 'shortName', 'typeOfLevel')
    # msgs0 = pygrib.open(filename)
    # Select from grbmsgs by variable
    msgst = msgs0.select(shortName='t', typeOfLevel='isobaricInhPa')
    msgsh = msgs0.select(shortName='gh', typeOfLevel='isobaricInhPa')
    # Should contain only one element
    msgsh0 = msgs0.select(shortName='gh', typeOfLevel='isothermZero')
    assert(len(msgsh0) == 1)

    # Sort grbmsgs by lv
    msgst = sorted(msgst, key=attrgetter('level'), reverse=True)
    msgsh = sorted(msgsh, key=attrgetter('level'), reverse=True)
    # Get latlon
    # lat, lon = grbmsg_latlon(msgst[0])
    # Get forecast hour, ['stepRange'] and ['endStep'] are same
    forecasttime = msgst[0]['startStep']
    # Get levels and assume levels of 3 variables are the same
    lvs_t = [item['level'] for item in msgst]
    lvs_h = [item['level'] for item in msgsh]
    assert (lvs_t == lvs_h)
    lv = lvs_t
    # Resizing and converting msgs to dict
    msgst = [grbmsg_dict_subset_region(msg) for msg in msgst]
    msgsh = [grbmsg_dict_subset_region(msg) for msg in msgsh]
    msgsh0 = grbmsg_dict_subset_region(msgsh0[0])
    # Stacking levels together
    msgst = np.stack([msg['values'] for msg in msgst])
    msgsh = np.stack([msg['values'] for msg in msgsh])
    msgsh0value = msgsh0['values']
    lat = msgsh0['lat']
    lon = msgsh0['lon']
    return forecasttime, lv, lat, lon, msgst, msgsh, msgsh0value


def grbmsg2dict(msg):
    msg_dict = dict()
    msg_dict['values'], msg_dict['lat'], msg_dict['lon'] = grbmsg_resolve(msg)
    # msg_dict['forecastTime'] = msg['forecastTime']
    # msg_dict['endStep'] = msg['endStep']
    # msg_dict['stepRange'] = msg['stepRange']
    return msg_dict


def grbmsg_lat_reverse(msg):
    """Reverse the lat array if monotonically decreasing."""
    lat_first = msg['latitudeOfFirstGridPointInDegrees']
    lat_last = msg['latitudeOfLastGridPointInDegrees']
    is_lat_decreasing = (lat_first > lat_last)
    if is_lat_decreasing:
        msg.values = msg.values[::-1, :]
        msg['latitudeOfFirstGridPointInDegrees'] = lat_last
        msg['latitudeOfLastGridPointInDegrees'] = lat_first
        msg['jScansPositively'] = 1
        # print("grbmsg lat reversed.")
    else:
        print("grbmsg lat in increasing order, nothing to do.")


def outer_boundary():
    """Find the outer boundary
    1. left, right
    2. with or without equal
    3. within range, on the range boundary, out of range"""


def grbmsg_dict_subset_region(msg, la0=15, la1=60, lo0=70, lo1=140):
    """Subtracting a region according to lats and lons.
    GFS: **GLOBAL**
    SCMOC: lat: 0-60, lon: 70-140
    Thus - merged: lat: 15-60, lon: 70-140
    """
    import numpy as np
    grbmsg_lat_reverse(msg)
    msg_dict = grbmsg2dict(msg)
    values, lat, lon = msg_dict['values'], msg_dict['lat'], msg_dict['lon']
    # Getting the indices of the coordinates
    # the first [0] gets the numpy array from the list
    # the second [0] gets the value of the element in the numpy array
    # ind_la0 = where(lat == la0)[0][0]
    posits_lat = np.where((lat >= la0) & (lat <= la1))
    ind_la0 = max(0, posits_lat[0][0] - 1)
    ind_la1 = min(len(lat), posits_lat[0][-1] + 1)
    # ind_la0 = np.abs(lat - la0).argmin()
    # ind_la1 = np.abs(lat - la1).argmin()
    posits_lon = np.where((lon >= lo0) & (lon <= lo1))
    ind_lo0 = max(0, posits_lon[0][0] - 1)
    ind_lo1 = min(len(lon), posits_lon[0][-1] + 1)
    # ind_lo0 = np.abs(lon - lo0).argmin()
    # ind_lo1 = np.abs(lon - lo1).argmin()
    # print(ind_la0, ind_la1, ind_lo0, ind_lo1)
    msg_dict['values'] = values[ind_la0:ind_la1 + 1, ind_lo0:ind_lo1 + 1]
    msg_dict['lat'] = lat[ind_la0:ind_la1 + 1]
    msg_dict['lon'] = lon[ind_lo0:ind_lo1 + 1]
    return msg_dict
