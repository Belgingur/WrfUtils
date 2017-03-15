import logging
from pathlib import Path

import numpy as np

from Elevator import build_interpolators
from utils_testing import nda_from_string
from vertical_interpolation import Interpolator

MY_DIR = Path(__file__).parent
TEST_DIR = MY_DIR / 'test_data'
WRFOUT_AFRICA = str(TEST_DIR / 'wrfout-africa-50.nc')
WRFOUT_AFRICA_DUMP = TEST_DIR / 'wrfout-africa-50.ncdump'

logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)


def min_max_round(a: np.ndarray, d=5):
    return round(np.min(a), d), round(np.max(a), d)


def test_build_interpolators():
    # Writes out the nda_from_string line for z_stag
    # from netCDF4 import Dataset
    # in_ds = Dataset(WRFOUT_AFRICA)
    # ph = in_ds.variables['PH'][1:4, 0:6, ::50, ::50]
    # phb = in_ds.variables['PHB'][1:4, 0:6, ::50, ::50]
    # hgt = in_ds.variables['HGT'][0, ::50, ::50]
    # z_stag = (ph + phb) / 9.81 - hgt
    # nda_to_code('z_stag', z_stag, 1)
    # t = in_ds.variables['T'][1:4, 0:6, ::50, ::50]
    # nda_to_code('t', t, 1)
    # in_ds.close()

    # vertically-staggered sigmal level heights, a sub-grid from africa-50, 3 time steps and 6 bottom sigma levels
    z_stag = nda_from_string((3, 6, 4, 4), 1,
                             '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 556 557 576 572 587 577 583 597 599 597 600 598 590 588 577 587 1194 1197 1237 1230 1262 1239 1253 1282 1286 1282 1289 1284 1266 1271 1239 1261 2478 2487 2570 2561 2622 2574 2601 2664 2671 2664 2678 2667 2628 2658 2574 2620 3776 3792 3918 3910 3997 3925 3964 4060 4072 4062 4081 4065 4006 4069 3930 3993 5088 5113 5282 5278 5388 5291 5343 5473 5488 5476 5503 5479 5399 5500 5305 5382 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 556 557 576 572 587 577 588 597 599 597 603 601 589 587 581 588 1193 1197 1237 1229 1262 1239 1263 1283 1286 1283 1295 1289 1265 1267 1248 1263 2476 2488 2569 2558 2621 2575 2622 2665 2671 2665 2690 2678 2628 2651 2593 2623 3771 3794 3917 3905 3997 3926 3996 4061 4072 4062 4100 4081 4005 4058 3952 3998 5082 5115 5281 5273 5388 5293 5385 5473 5488 5477 5522 5500 5398 5488 5326 5388 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 555 558 576 571 587 577 593 597 599 597 605 603 589 592 584 589 1192 1198 1236 1228 1262 1240 1272 1283 1286 1282 1299 1295 1265 1272 1254 1266 2473 2489 2569 2556 2621 2576 2641 2665 2671 2665 2698 2688 2627 2654 2605 2629 3767 3795 3916 3901 3997 3927 4024 4061 4072 4062 4111 4097 4004 4062 3970 4006 5076 5117 5279 5267 5387 5294 5422 5473 5488 5476 5541 5521 5396 5491 5350 5399')

    # temperature on vertically aligned grid otherwise like that of z_stag
    t = nda_from_string((3, 6, 4, 4), 1,
                        '-170 -171 -99 -122 -57 -96 135 -25 -7 -19 52 14 -44 19 -3 110 -172 -171 -99 -119 -58 -97 131 -25 -8 -19 50 11 -46 56 -5 108 -175 -171 -98 -113 -58 -97 129 -25 -8 -20 49 9 -47 76 -7 106 -177 -169 -96 -104 -58 -97 128 -21 -9 -17 55 7 -47 95 4 106 -177 -166 -94 -95 -57 -97 127 -7 -8 -12 63 6 -47 104 12 105 -177 -161 -89 -87 -54 -97 140 7 -6 -6 68 6 -46 108 18 105 -173 -170 -99 -124 -57 -96 161 -24 -7 -20 65 29 -45 15 17 117 -175 -170 -99 -121 -58 -97 156 -24 -8 -20 62 25 -46 42 14 113 -179 -170 -98 -115 -58 -97 153 -24 -9 -20 60 22 -47 69 12 112 -181 -168 -97 -107 -58 -97 152 -20 -9 -18 60 21 -47 88 12 110 -181 -165 -94 -97 -57 -97 151 -6 -8 -13 64 19 -46 100 11 109 -180 -161 -89 -88 -54 -97 150 7 -6 -6 70 19 -46 106 17 109 -176 -170 -99 -125 -58 -96 183 -24 -7 -21 77 45 -45 41 32 124 -178 -169 -99 -123 -58 -97 177 -24 -9 -21 74 39 -46 40 29 121 -182 -169 -98 -118 -58 -97 174 -24 -9 -21 73 37 -47 65 27 119 -184 -167 -97 -109 -58 -97 171 -18 -9 -19 72 35 -47 89 26 117 -184 -164 -94 -100 -57 -97 170 -5 -8 -14 72 34 -46 100 25 116 -184 -160 -89 -90 -53 -97 169 7 -6 -7 72 33 -46 105 25 116')

    # The extreme sigma levels (except bottom level which is constant 0)
    assert min_max_round(z_stag[:, 0, :, :]) == (0.0, 0.0)
    assert min_max_round(z_stag[:, 1, :, :]) == (55.5, 60.5)  # Partially contains 60
    assert min_max_round(z_stag[:, 2, :, :]) == (119.2, 129.9)
    assert min_max_round(z_stag[:, 3, :, :]) == (247.3, 269.8)  # Fully contains 350
    assert min_max_round(z_stag[:, 4, :, :]) == (376.7, 411.1)
    assert min_max_round(z_stag[:, 5, :, :]) == (507.6, 554.1)  # Partially contains 550

    targets = [60, 350, 550]
    ipor_alig, ipor_stag = build_interpolators(z_stag, targets, True, True)

    assert type(ipor_alig) == Interpolator
    assert type(ipor_stag) == Interpolator

    assert ipor_stag.max_k == 5
    assert ipor_alig.max_k == 4  # Don't need 5th since 550m is completely masked out

    z_tgt = ipor_stag(z_stag)

    # 60m crosses sigma-0 and sigma-1
    assert min_max_round(ipor_stag.vics[0].k_fl) == (0, 1)
    assert np.sum(ipor_stag.vics[0].mask) == 0
    assert round(np.min(z_tgt[:, 0]), 5) == 60
    assert round(np.max(z_tgt[:, 0]), 5) == 60

    # 350m is in sigma-3
    assert min_max_round(ipor_stag.vics[1].k_fl) == (3, 3)
    assert np.sum(ipor_stag.vics[1].mask) == 0
    assert round(np.min(z_tgt[:, 1]), 5) == 350
    assert round(np.max(z_tgt[:, 1]), 5) == 350

    # 550m is in sigma-4 but blows out of the top at 42 points
    assert min_max_round(ipor_stag.vics[2].k_fl) == (4, 4)
    assert np.sum(ipor_stag.vics[2].mask) == 42
    assert np.isnan(np.min(z_tgt[:, 2]))
    assert np.isnan(np.max(z_tgt[:, 2]))
    assert z_tgt[:, 2, 2, 2].tolist() == [550, 550, 550]

    t_tgt = ipor_alig(t)
    assert t_tgt[:, 0, 0, 0].tolist() == [-17.10787269681742, -17.40796311818944, -17.70822147651007]
    assert t_tgt[:, 1, 0, 0].tolist() == [-17.7, -18.1, -18.400000000000002]

    # 550m is completely masked out in the aligne interpolator
    assert np.sum(ipor_alig.vics[2].mask) == 48
