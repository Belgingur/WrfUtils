import logging
from datetime import datetime, timedelta

import numpy as np
from pytz import utc


LOG = logging.getLogger('belgingur.wrfout_reader')


WRF_DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'


def parse_wrf_date(s):
    """
    Parses a WRF-style timestamp string into a UTC datetime instance
    :param unicode s:
    :rtype: datetime
    """
    ts = datetime.strptime(s, WRF_DATE_FORMAT)
    ts = ts.replace(tzinfo=utc)
    return ts


class WRFReader(object):
    def __init__(self, nc_data):
        self.vars = nc_data.variables

        self.start_date = parse_wrf_date(nc_data.START_DATE)

        times = [parse_wrf_date(rt.tostring().decode()) for rt in nc_data.variables['Times']]
        time_step = timedelta(seconds=round((times[-1] - times[0]).total_seconds() / (len(times) - 1)))
        self.interval_hours = time_step.total_seconds() / 3600.0

        # Sanity check of the calculated last forecast date with comparison to the real one
        stop = self.start_date + timedelta(hours=self.interval_hours * (len(nc_data.dimensions['Time']) - 1))
        check = abs((times[-1] - stop).total_seconds())
        if check > 60:
            LOG.warning("The last date is different from estimated average by more than one minute")

    def get_timestamps(self, spinup=0):
        timerange = len(self.vars['Times'])
        ts = [self.start_date + timedelta(hours=i * self.interval_hours) for i in range(spinup, timerange)]
        return ts

    def get_variable(self, variable, spinup=0):

        LOG.info("Getting variable %s", variable)

        if variable in self.vars:
            return self.vars[variable][:][spinup:]
        else:
            return getattr(self, variable)()[:][spinup:]

    def temp(self):

        """ Temperature at 2m in degrees Celsius"""

        return self.get_variable("T2") - 273.15

    def wind_speed(self):
        return vec_len(self.get_variable('U10'), self.get_variable('V10'))

    def wind_dir(self):
        sinalpha = self.get_variable('SINALPHA')
        cosalpha = self.get_variable('COSALPHA')
        U = self.get_variable('U10')
        V = self.get_variable('V10')

        U10_true = cosalpha * U + sinalpha * V
        V10_true = -sinalpha * U + cosalpha * V

        return vec_dir(U10_true, V10_true)

    def prec_rate(self):

        """ Precipitation rate. """

        pa = self.get_variable('RAINC') + self.get_variable('RAINNC')
        dp = np.zeros(pa.shape)
        dp[0] = np.NaN
        dp[1:] = pa[1:] - pa[:-1]
        return dp / self.interval_hours

    def snow_ratio(self):
        return self.get_variable("SR")

    def pressure(self):

        """ Surface pressure in hPa. """

        return self.get_variable("PSFC") / 100.0

    def mslp(self):

        """ Mean sea level pressure in hPa. """

        pressure = self.get_variable("PSFC")
        temperature = self.get_variable("T2")
        mixing_ratio = self.get_variable("Q2")
        height = self.get_variable("HGT")
        econst = 0.607717041
        T_vs = temperature * (1.0 + (econst * mixing_ratio))
        T0 = (0.00650 * height) + temperature
        T_v0 = T0 * ( 1.0 + (econst * mixing_ratio))
        T_vmean = 0.50 * (T_v0 + T_vs)
        g0 = 9.810
        R_d = 287.0

        slp = pressure * np.exp((g0 * height) / (R_d * T_vmean))
        return slp / 100.0

    def humidity(self):

        """ Absolute humidity at 2m in kg/kg. """

        return self.get_variable("Q2")

    def rel_hum(self):

        """ Relative humidity at 2m in %. """

        if 'RH2' in self.vars:
            return self.get_variable("RH2")

        eps = 0.622
        q2 = self.humidity()
        q2 = np.maximum(q2, 0.0)
        tk = self.get_variable("T2")
        prs = self.get_variable("PSFC")
        tmp1 = 10.0 * 0.6112 * np.exp(17.67 * (tk - 273.16) / (tk - 29.65))
        tmp2 = eps * tmp1 / (0.01 * prs - (1.0 - eps) * tmp1)
        return 100.0 * np.maximum(np.minimum(q2 / tmp2, 1.0), 0.0)

    def total_clouds(self):

        """ Total cloud cover. """

        pressure = self.pressure_cube()
        rel_hum = self.rel_hum_cube()
        return calc_cloud_cover(pressure, rel_hum)

    def pressure_cube(self):

        """ Pressure at all sigma levels. """

        return self.get_variable('P') + self.get_variable('PB')

    def rel_hum_cube(self):

        """ Relative humidity at all sigma levels. """

        if 'RELHUM' in self.vars:
            return self.get_variable('RELHUM')

        eps = 0.622

        qvapor = self.get_variable("QVAPOR")
        pressure = self.pressure_cube()

        t_pot = self.get_variable('T') + 300.0
        temp_norm = t_pot * (pressure / 100000.0) ** 0.2856

        rel_hum = np.ndarray(qvapor.shape, np.float)

        for i in range(len(qvapor)):
            # QVAPOR = numpy.maximum(self.st_named('QVAPOR', step), 0.0)
            max_qvapor = np.maximum(qvapor[i], 0.0)

            tmp1 = 10.0 * 0.6112 * np.exp(17.67 * (temp_norm[i] - 273.16) / (temp_norm[i] - 29.65))
            tmp2 = eps * tmp1 / (0.01 * pressure[i] - (1.0 - eps) * tmp1)
            rel_hum[i] = 100.0 * np.maximum(np.minimum(max_qvapor / tmp2, 1.0), 0.0)
        return rel_hum


def vec_len(u, v):
    return (u ** 2 + v ** 2) ** 0.5


def vec_dir(u, v):
    return np.mod(270 - (np.arctan2(v, u) * 180 / 3.14159), 360)


def calc_cloud_cover(pressure, rh):
    pre_low = np.ma.masked_outside(pressure, 80000.0, 97000.0)
    pre_mid = np.ma.masked_outside(pressure, 45000.0, 80000.0)
    pre_high = np.ma.masked_outside(pressure, 0.0, 45000.0)

    rh_low = np.ma.array(rh, mask=pre_low.mask, fill_value=0.0)
    rh_mid = np.ma.array(rh, mask=pre_mid.mask, fill_value=0.0)
    rh_high = np.ma.array(rh, mask=pre_high.mask, fill_value=0.0)

    # Find the maximum humidity values in each layer.
    # (Testing the generated array pre_low so that input variables can be scalar)

    cloud_cover_total = np.ndarray((rh.shape[0], rh.shape[2], rh.shape[3]), np.float)
    for i in range(len(pressure)):
        low_rh_slice = rh_low[i].max(axis=0)
        middle_rh_slice = rh_mid[i].max(axis=0)
        high_rh_slice = rh_high[i].max(axis=0)

        # Cloud cover fraction calculated using the magic equations
        cc_low = 4.0 * low_rh_slice * 0.01 - 3.0
        cc_mid = 4.0 * middle_rh_slice * 0.01 - 3.0
        cc_high = 2.5 * high_rh_slice * 0.01 - 1.5

        # We make sure that the cloud-fraction stays between 0 and 1
        cc_low = np.ma.masked_outside(cc_low, 1.0, 0.0)
        cc_mid = np.ma.masked_outside(cc_mid, 1.0, 0.0)
        cc_high = np.ma.masked_outside(cc_high, 1.0, 0.0)

        # Replace masked data with zeros
        cc_low = cc_low.filled(fill_value=0.0)
        cc_mid = cc_mid.filled(fill_value=0.0)
        cc_high = cc_high.filled(fill_value=0.0)

        # Total cloud-cover
        cc_comb = np.array([cc_low, cc_mid, cc_high])
        cc_total = cc_comb.max(axis=0)
        cc_total = np.ma.masked_less(cc_total, 0.0)
        cc_total = cc_total.filled(fill_value=0.0)
        cc_total = np.ma.masked_greater(cc_total, 1.0)
        cc_total = cc_total.filled(fill_value=1.0)

        cloud_cover_total[i] = cc_total
    return cloud_cover_total

