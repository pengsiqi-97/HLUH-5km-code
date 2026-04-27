# %%
import numpy as np
import rasterio
import os
from osgeo import gdal, osr, gdal_array, gdalconst,ogr


# %%
path = r"F:/GLC_FCS30Dv282/0022"
files= os.listdir(path)
rf = lambda x:'G:/GLC_FCS30Dv282/0022/' + x
rf1 = lambda x:'G:/GLC_FCS30Dv282/reclass4/2010-cropland/' + x
tf = lambda x:'G:/GLC_FCS30Dv282/reclass4/2020-cropland/' + x

# %%
for file in files: #Iterate through the folder
    
    if os.path.splitext(file)[1] == '.tif':#目录下包含.tif的文件
        
        
        print(file)
        new_raster_path = rf(file)
        old_raster_path = rf1(file)
        with rasterio.open(new_raster_path) as src:
            # Read the Nth band
            inRaster = src.read(21)
            old_src = rasterio.open(old_raster_path)
            old_data = old_src.read(1)
            outCon = np.zeros(inRaster.shape, dtype=np.int8)
            # reclassify
            outCon[((old_data==10)&(inRaster<=20)&(inRaster>0))]=1
            outCon[((old_data==21)&((inRaster == 62) | (inRaster == 72) | (inRaster == 82)))]=2
            outCon[((old_data==20)&((inRaster == 120) | (inRaster == 121) | (inRaster == 122) | (inRaster == 130)))]=3
            outCon[((old_data==30)&((inRaster == 150) | (inRaster == 152) | (inRaster == 153)))]=4
            outCon[((old_data==40)&((inRaster == 200) | (inRaster == 201) | (inRaster == 202)))]=5


            outCon[((old_data==21)&(inRaster<=20)&(inRaster>0))]=10
            outCon[((old_data==20)&(inRaster<=20)&(inRaster>0))]=20
            outCon[((old_data==30)&(inRaster<=20)&(inRaster>0))]=30
            outCon[((old_data==40)&(inRaster<=20)&(inRaster>0))]=40
            outCon[((old_data==0)&(inRaster<=20)&(inRaster>0))]=60
            
            outCon[(old_data==10)&((inRaster == 62) | (inRaster == 72) | (inRaster == 82))]=-1
            outCon[(old_data==10)&((inRaster == 120) | (inRaster == 121) | (inRaster == 122) | (inRaster == 130))]=-2
            outCon[(old_data==10)&((inRaster == 150) | (inRaster == 152) | (inRaster == 153))]=-3
            outCon[(old_data==10)&((inRaster == 200) | (inRaster == 201) | (inRaster == 202))]=-4
            outCon[((old_data==10)&(inRaster==220))] = -5
            outCon[((old_data==10)&(inRaster==210))] = -6

            outCon[(((old_data==21)|(old_data==21))&((inRaster == 150) | (inRaster == 152) | (inRaster == 153)))] = -13
            outCon[(((old_data==20)|(old_data==21))&((inRaster == 150) | (inRaster == 152) | (inRaster == 153)))] = -23
            outCon[(((old_data==21)|(old_data==21))&((inRaster == 200) | (inRaster == 201) | (inRaster == 202)))] = -14
            outCon[(((old_data==20)|(old_data==21))&((inRaster == 200) | (inRaster == 201) | (inRaster == 202)))] = -24
            outCon[((old_data==21)&(inRaster==220))] = -15
            outCon[((old_data==20)&(inRaster==220))] = -25
            outCon[((old_data==21)&(inRaster==210))] = -16
            outCon[((old_data==20)&(inRaster==210))] = -26
            

            outCon[((old_data==30)&((inRaster == 62) | (inRaster == 72) | (inRaster == 82)))]=31
            outCon[((old_data==30)&((inRaster == 120) | (inRaster == 121) | (inRaster == 122) | (inRaster == 130)))]=32
            outCon[((old_data==30)&((inRaster == 200) | (inRaster == 201) | (inRaster == 202)))] = -34
            outCon[((old_data==30)&(inRaster==220))] = -35
            outCon[((old_data==30)&(inRaster==210))] = -36

            outCon[((old_data==40)&((inRaster == 62) | (inRaster == 72) | (inRaster == 82)))]=41
            outCon[((old_data==40)&((inRaster == 120) | (inRaster == 121) | (inRaster == 122) | (inRaster == 130)))]=42
            outCon[((old_data==40)&((inRaster == 150) | (inRaster == 152) | (inRaster == 153)))]=43
            outCon[((old_data==40)&(inRaster==220))] = -45
            outCon[((old_data==40)&(inRaster==210))] = -46
            
            outCon[((old_data==0)&((inRaster == 62) | (inRaster == 72) | (inRaster == 82)))]=61
            outCon[((old_data==0)&((inRaster == 120) | (inRaster == 121) | (inRaster == 122) | (inRaster == 130)))]=62
            outCon[((old_data==0)&((inRaster == 150) | (inRaster == 152) | (inRaster == 153)))]=63
            outCon[((old_data==0)&((inRaster == 200) | (inRaster == 201) | (inRaster == 202)))]=64
            
        # save the reclassified raster
        output_raster_path = tf(file)
    with rasterio.open(output_raster_path, 'w', driver='GTiff', height=outCon.shape[0],
                       width=outCon.shape[1], count=1, dtype=outCon.dtype,
                       crs=src.crs, transform=src.transform,compress='lzw') as dst:
        dst.write(outCon, 1)
        
print('SUCCESS!')


