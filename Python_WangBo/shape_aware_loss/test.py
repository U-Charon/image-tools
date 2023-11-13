import osgeo
print(osgeo.__version__)
import tifffile
print(tifffile.__version__)

from osgeo import gdal

out_driver = gdal.GetDriverByName('gtiff')
print(out_driver)

import geopandas
print(geopandas.__version__)

import numpy as np
a = np.zeros(shape=[1,512,512,8])
print(a[..., 3:-2].shape)