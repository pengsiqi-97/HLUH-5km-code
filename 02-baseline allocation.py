import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 0. Global Path Configuration 
# ==========================================
CONFIG = {
    "dir_reclass_cropland": "data/reclass/cropland",
    "dir_area_index": "data/area_index10",
    "dir_ucl": "data/psq_process/ucl_10",
    "dir_pda": "data/psq_process/WDPA_10",
    "dir_conser": "data/conservation_au",
    "dir_process_base": "data/cropland/process",
    "dir_final_base": "data/cropland/final",
    "dir_vector": "data/country_vector",
    "excel_stats": "data/FAOSTAT_data_2025.xlsx"
}

# create output paths if they don't exist
def rf(x): return os.path.join(CONFIG["dir_reclass_cropland"], x)
def anf(x): return os.path.join(CONFIG["dir_area_index"], x)
def uclf(x): return os.path.join(CONFIG["dir_ucl"], x)
def PDAf(x): return os.path.join(CONFIG["dir_pda"], x)
def conserf(x): return os.path.join(CONFIG["dir_conser"], x)
def moutf(country, x): return os.path.join(CONFIG["dir_process_base"], country.lower(), x)
def foutf(country, x): return os.path.join(CONFIG["dir_final_base"], country.lower(), x)

# ==========================================
# 1. Core Area Calculation Functions 
# ==========================================

def calculate_weighted_area(raster_path, fraction_path, PDA_path, conser_path, region, target_value, pixel_area, ratio):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        if region.intersects(raster_box).any():
            data = src.read(1)
            mask = np.zeros(data.shape, dtype=bool)

            for geom in region.geometry:
                shape_mask = rasterize(
                    [(geom, 1)], out_shape=data.shape, transform=src.transform, fill=0, dtype='uint8'
                )
                mask |= shape_mask == 1

            with rasterio.open(fraction_path) as frac_src:
                fraction_data = frac_src.read(1)
                with rasterio.open(PDA_path) as PDA_src:
                    PDA_mask = PDA_src.read(1)
                    valid_mask = (data == target_value) & mask & (PDA_mask != 1)
                    weighted_area = np.sum(fraction_data[valid_mask] * pow(10, -4)) * 100 * pow(10, -6) * ratio
                    return weighted_area
    return 0

def calculate_all(cross_filelist, region, area_need):
    area4 = [0, 0, 0, 0]
    total_area_10 = sum([calculate_weighted_area(rf(f), anf(f), PDAf(f), conserf(f), region, 1, 100, 0.9) for f in cross_filelist if f.endswith('.tif')])
    area4[0] = total_area_10
    if area4[0] >= area_need: return area4

    total_area_20 = sum([calculate_weighted_area(rf(f), anf(f), PDAf(f), conserf(f), region, 3, 100, 0.9) for f in cross_filelist if f.endswith('.tif')])
    total_area_21 = sum([calculate_weighted_area(rf(f), anf(f), PDAf(f), conserf(f), region, 2, 100, 0.3) for f in cross_filelist if f.endswith('.tif')])
    area4[1] = total_area_10 + total_area_20 + total_area_21
    if area4[1] >= area_need: return area4

    total_area_30 = sum([calculate_weighted_area(rf(f), anf(f), PDAf(f), conserf(f), region, 4, 100, 0.9) for f in cross_filelist if f.endswith('.tif')])
    area4[2] = area4[1] + total_area_30
    if area4[2] >= area_need: return area4

    total_area_40 = sum([calculate_weighted_area(rf(f), anf(f), PDAf(f), conserf(f), region, 5, 100, 0.9) for f in cross_filelist if f.endswith('.tif')])
    area4[3] = area4[2] + total_area_40
    return area4

def find_area(country_name, data):
    result = data.loc[data['country'].str.lower() == country_name.lower(), 'cropland']
    return result.iloc[0] if not result.empty else None

def cross_filefind(filelist, region):
    cross_filelist = []
    for file in filelist:
        if file.endswith('.tif'):
            with rasterio.open(rf(file)) as src:
                bounds = src.bounds
                raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                if region.intersects(raster_box).any():
                    src_data = src.read(1)
                    mask = np.zeros(src_data.shape, dtype=bool)
                    for geom in region.geometry:
                        shape_mask = rasterize([(geom, 1)], out_shape=src_data.shape, transform=src.transform, fill=0, dtype='uint8')
                        mask |= shape_mask == 1
                    if np.any(mask):
                        cross_filelist.append(file)
    return cross_filelist

# ==========================================
# 2. Spatial Allocation & Index Functions 
# ==========================================

def allocate_spatial_by_level(input_raster_path, output_raster_path, PDA_path, conser_path, region, level):
    with rasterio.open(input_raster_path) as src:
        bounds = src.bounds
        raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        if region.intersects(raster_box).any():
            src_data = src.read(1)
            dst_data = np.zeros(src_data.shape, dtype='uint8')
            mask = np.zeros(src_data.shape, dtype=bool)

            for geom in region.geometry:
                shape_mask = rasterize([(geom, 1)], out_shape=src_data.shape, transform=src.transform, fill=0, dtype='uint8')
                mask |= shape_mask == 1
                
            with rasterio.open(PDA_path) as PDA_src:
                PDA_mask = PDA_src.read(1)
                
            if level == 1:
                dst_data[(src_data == 1) & mask & (PDA_mask != 1)] = 1
            elif level == 2:
                dst_data[((src_data == 1) | (src_data == 3)) & mask & (PDA_mask != 1)] = 1
                dst_data[(src_data == 2) & mask & (PDA_mask != 1)] = 2
            elif level == 3:
                dst_data[((src_data == 1) | (src_data == 3) | (src_data == 4)) & mask & (PDA_mask != 1)] = 1
                dst_data[(src_data == 2) & mask & (PDA_mask != 1)] = 2
            elif level == 4:
                dst_data[((src_data == 1) | (src_data == 3) | (src_data == 4) | (src_data == 5)) & mask & (PDA_mask != 1)] = 1
                dst_data[(src_data == 2) & mask & (PDA_mask != 1)] = 2

            with rasterio.open(output_raster_path, 'w', driver='GTiff', height=dst_data.shape[0],
                       width=dst_data.shape[1], count=1, dtype=dst_data.dtype,
                       crs=src.crs, transform=src.transform, compress='lzw') as dst:
                dst.write(dst_data, 1)

def calculate_i(cross_filelist, region, area_needed, target_value):
    for i in range(100, -1, -10):
        area = 0
        for file in cross_filelist:
            with rasterio.open(rf(file)) as src, rasterio.open(PDAf(file)) as PDA_src, \
                 rasterio.open(anf(file)) as frac_src, rasterio.open(uclf(file)) as ucl_src:
                
                src_data = src.read(1)
                mask = np.zeros(src_data.shape, dtype=bool)
                for geom in region.geometry:
                    shape_mask = rasterize([(geom, 1)], out_shape=src_data.shape, transform=src.transform, fill=0, dtype='uint8')
                    mask |= shape_mask == 1

                valid_mask = (src_data == target_value) & mask & (ucl_src.read(1) > i) & (PDA_src.read(1) != 1)
                area += np.sum(frac_src.read(1)[valid_mask] * pow(10, -4)) * 100 * pow(10, -6) * 0.9

        if area >= area_needed and i == 100: i = 90
        if area >= area_needed: break
        if area < area_needed and i == 0: i = -9        
    return i + 10

def calculate_i20(cross_filelist, region, area_needed):
    for i in range(100, -1, -10):
        area = 0
        for file in cross_filelist:
            with rasterio.open(rf(file)) as src, rasterio.open(PDAf(file)) as PDA_src, \
                 rasterio.open(anf(file)) as frac_src, rasterio.open(uclf(file)) as ucl_src:
                
                src_data = src.read(1)
                mask = np.zeros(src_data.shape, dtype=bool)
                for geom in region.geometry:
                    shape_mask = rasterize([(geom, 1)], out_shape=src_data.shape, transform=src.transform, fill=0, dtype='uint8')
                    mask |= shape_mask == 1

                ucl_data = ucl_src.read(1)
                pda_data = PDA_src.read(1)
                frac_data = frac_src.read(1)

                valid_mask1 = (src_data == 3) & mask & (ucl_data > i) & (pda_data != 1)
                valid_mask2 = (src_data == 2) & mask & (ucl_data > i) & (pda_data != 1)
                
                area += np.sum(frac_data[valid_mask1] * pow(10, -4)) * 100 * pow(10, -6) * 0.9 + \
                        np.sum(frac_data[valid_mask2] * pow(10, -4)) * 100 * pow(10, -6) * 0.3
        
        if area >= area_needed and i == 100: i = 90
        if area >= area_needed: break
        if area < area_needed and i == 0: i = -9
    return i + 10

# ==========================================
# 3. Sequential Allocation & Processing Functions 
# ==========================================

def process_file_ucl(file, region, i, target_value, fraction_path, ucl_path, PDA_path, conser_path):
    try:
        with rasterio.open(rf(file)) as src:
            src_data = src.read(1)
            mask = np.zeros(src_data.shape, dtype=bool)
            for geom in region.geometry:
                shape_mask = rasterize([(geom, 1)], out_shape=src_data.shape, transform=src.transform, fill=0, dtype='uint8')
                mask |= shape_mask == 1

            with rasterio.open(PDA_path) as PDA_src, rasterio.open(fraction_path) as frac_src, rasterio.open(ucl_path) as ucl_src:
                PDA_mask = PDA_src.read(1)
                fraction_data = frac_src.read(1)
                ucl_data = ucl_src.read(1)
                valid_mask = (src_data == target_value) & mask & (ucl_data == i) & (PDA_mask != 1)
                weighted_area = np.sum(fraction_data[valid_mask] * pow(10, -4)) * 100 * pow(10, -6) * 0.9

            return weighted_area, valid_mask
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return 0, None

def process_file_ucl20(file, region, i, fraction_path, ucl_path, PDA_path, conser_path):
    try:
        with rasterio.open(rf(file)) as src:
            src_data = src.read(1)
            mask = np.zeros(src_data.shape, dtype=bool)
            for geom in region.geometry:
                shape_mask = rasterize([(geom, 1)], out_shape=src_data.shape, transform=src.transform, fill=0, dtype='uint8')
                mask |= shape_mask == 1

            with rasterio.open(PDA_path) as PDA_src, rasterio.open(fraction_path) as frac_src, rasterio.open(ucl_path) as ucl_src:
                PDA_mask = PDA_src.read(1)
                fraction_data = frac_src.read(1)
                ucl_data = ucl_src.read(1)
                valid_mask1 = (src_data == 3) & mask & (ucl_data == i) & (PDA_mask != 1)
                valid_mask2 = (src_data == 2) & mask & (ucl_data == i) & (PDA_mask != 1)
                weighted_area = np.sum(fraction_data[valid_mask1] * pow(10, -4)) * 100 * pow(10, -6) * 0.9 + \
                                np.sum(fraction_data[valid_mask2] * pow(10, -4)) * 100 * pow(10, -6) * 0.3

            return weighted_area, valid_mask1, valid_mask2
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return 0, None, None

def allocate_part_sequential(cross_filelist, region, area_needed, target_value, start_i, country):
    area = 0
    dst_data_dict = {file: None for file in cross_filelist}
    
    for file in cross_filelist:
        fraction_path = anf(file) 
        ucl_path = uclf(file)
        PDA_path = PDAf(file)
        with rasterio.open(rf(file)) as src, rasterio.open(PDA_path) as PDA_src, \
             rasterio.open(fraction_path) as frac_src, rasterio.open(ucl_path) as ucl_src:
                src_data = src.read(1)
                mask = np.zeros(src_data.shape, dtype=bool)
                for geom in region.geometry:
                    shape_mask = rasterize([(geom, 1)], out_shape=src_data.shape, transform=src.transform, fill=0, dtype='uint8')
                    mask |= shape_mask == 1
                valid_mask = (src_data == target_value) & mask & (ucl_src.read(1) > start_i) & (PDA_src.read(1) != 1)
                weighted_area = np.sum(frac_src.read(1)[valid_mask] * pow(10, -4)) * 100 * pow(10, -6) * 0.9
        
        area += weighted_area
        if dst_data_dict[file] is None: dst_data_dict[file] = valid_mask.copy()
        else: dst_data_dict[file] |= valid_mask

    for i in range(start_i, -1, -1):
        total_area_for_i = 0
        results = []
        for file in cross_filelist:
            weighted_area, valid_mask = process_file_ucl(file, region, i, target_value, anf(file), uclf(file), PDAf(file), conserf(file))
            results.append((file, weighted_area, valid_mask))
            total_area_for_i += weighted_area

        if total_area_for_i >= area_needed - area:
            ratio = (area_needed - area) / total_area_for_i
            for file, weighted_area, valid_mask in results:
                valid_indices = np.argwhere(valid_mask)
                sample_size = int(len(valid_indices) * ratio)
                if sample_size > 0:
                    sampled_indices = valid_indices[np.random.choice(len(valid_indices), sample_size, replace=False)]
                    sampled_mask = np.zeros_like(valid_mask, dtype=bool)
                    sampled_mask[tuple(sampled_indices.T)] = True
                    with rasterio.open(anf(file)) as frac_src:
                        area += np.sum(frac_src.read(1)[sampled_mask] * pow(10, -4)) * 100 * pow(10, -6) * 0.9
                    if dst_data_dict[file] is None: dst_data_dict[file] = sampled_mask
                    else: dst_data_dict[file][sampled_mask] = 1
            break
        else:
            for file, weighted_area, valid_mask in results:
                area += weighted_area
                if dst_data_dict[file] is None: dst_data_dict[file] = valid_mask
                else: dst_data_dict[file] |= valid_mask

    for file, dst_data in dst_data_dict.items():
        with rasterio.open(rf(file)) as src:
            if dst_data is None: dst_data = np.zeros((src.height, src.width), dtype=bool)
            with rasterio.open(foutf(country, file), 'w', **src.meta) as dst:
                dst.write(dst_data, 1)
    return area

def allocate_part20_sequential(cross_filelist, region, area_needed, start_i, country):
    area = 0
    dst_data_dict = {file: None for file in cross_filelist}
    
    for file in cross_filelist:
        with rasterio.open(rf(file)) as src, rasterio.open(PDAf(file)) as PDA_src, \
             rasterio.open(anf(file)) as frac_src, rasterio.open(uclf(file)) as ucl_src:
                src_data = src.read(1)
                mask = np.zeros(src_data.shape, dtype=bool)
                for geom in region.geometry:
                    shape_mask = rasterize([(geom, 1)], out_shape=src_data.shape, transform=src.transform, fill=0, dtype='uint8')
                    mask |= shape_mask == 1
                
                ucl_data = ucl_src.read(1)
                pda_mask = PDA_src.read(1)
                valid_mask1 = (src_data == 3) & mask & (ucl_data > start_i) & (pda_mask != 1)
                valid_mask2 = (src_data == 2) & mask & (ucl_data > start_i) & (pda_mask != 1)
                
                fraction_data = frac_src.read(1)
                weighted_area = np.sum(fraction_data[valid_mask1] * pow(10, -4)) * 100 * pow(10, -6) * 0.9 + \
                                np.sum(fraction_data[valid_mask2] * pow(10, -4)) * 100 * pow(10, -6) * 0.3
        
        area += weighted_area
        if dst_data_dict[file] is None:
            dst_data_dict[file] = np.zeros_like(src_data, dtype=int)
        dst_data_dict[file][valid_mask1] = 1
        dst_data_dict[file][valid_mask2] = 2

    for i in range(start_i, -1, -1):
        total_area_for_i = 0
        results = []
        for file in cross_filelist:
            weighted_area, valid_mask1, valid_mask2 = process_file_ucl20(file, region, i, anf(file), uclf(file), PDAf(file), conserf(file))
            results.append((file, weighted_area, valid_mask1, valid_mask2))
            total_area_for_i += weighted_area

        if total_area_for_i >= area_needed - area:
            ratio = (area_needed - area) / total_area_for_i
            for file, weighted_area, valid_mask1, valid_mask2 in results:
                valid_indices1 = np.argwhere(valid_mask1)
                sample_size1 = int(len(valid_indices1) * ratio)
                if sample_size1 > 0:
                    sampled_indices1 = valid_indices1[np.random.choice(len(valid_indices1), sample_size1, replace=False)]
                    sampled_mask1 = np.zeros_like(valid_mask1, dtype=bool)
                    sampled_mask1[tuple(sampled_indices1.T)] = True
                else: sampled_mask1 = np.zeros_like(valid_mask1, dtype=bool)
                
                valid_indices2 = np.argwhere(valid_mask2)
                sample_size2 = int(len(valid_indices2) * ratio)
                if sample_size2 > 0:
                    sampled_indices2 = valid_indices2[np.random.choice(len(valid_indices2), sample_size2, replace=False)]
                    sampled_mask2 = np.zeros_like(valid_mask2, dtype=bool)
                    sampled_mask2[tuple(sampled_indices2.T)] = True
                else: sampled_mask2 = np.zeros_like(valid_mask2, dtype=bool)

                with rasterio.open(anf(file)) as frac_src:
                    fraction_data = frac_src.read(1)
                    if sample_size1 > 0: area += np.sum(fraction_data[sampled_mask1] * pow(10, -4)) * 100 * pow(10, -6) * 0.9
                    if sample_size2 > 0: area += np.sum(fraction_data[sampled_mask2] * pow(10, -4)) * 100 * pow(10, -6) * 0.3

                if dst_data_dict[file] is None: dst_data_dict[file] = np.zeros_like(valid_mask1, dtype=int)
                if sample_size1 > 0: dst_data_dict[file][sampled_mask1] = 1
                if sample_size2 > 0: dst_data_dict[file][sampled_mask2] = 2
            break
        else:
            for file, weighted_area, valid_mask1, valid_mask2 in results:
                area += weighted_area
                if dst_data_dict[file] is None: dst_data_dict[file] = np.zeros_like(valid_mask1, dtype=int)
                dst_data_dict[file][valid_mask1] = 1
                dst_data_dict[file][valid_mask2] = 2

    for file, dst_data in dst_data_dict.items():
        with rasterio.open(rf(file)) as src:
            if dst_data is None: dst_data = np.zeros((src.height, src.width), dtype=int)
        
        with rasterio.open(moutf(country, file)) as old_src:
            old_data = old_src.read(1)
            dst_data[(old_data == 1)] = 1
            dst_data[(old_data == 2)] = 2        
            
        with rasterio.open(rf(file)) as src:
            with rasterio.open(foutf(country, file), 'w', **src.meta) as dst:
                dst.write(dst_data, 1)
    return area

def allocate_part3040_sequential(cross_filelist, region, area_needed, target_value, start_i, country):
    area = 0
    dst_data_dict = {file: None for file in cross_filelist}
    
    for file in cross_filelist:
        with rasterio.open(rf(file)) as src, rasterio.open(PDAf(file)) as PDA_src, \
             rasterio.open(anf(file)) as frac_src, rasterio.open(uclf(file)) as ucl_src:
                src_data = src.read(1)
                mask = np.zeros(src_data.shape, dtype=bool)
                for geom in region.geometry:
                    shape_mask = rasterize([(geom, 1)], out_shape=src_data.shape, transform=src.transform, fill=0, dtype='uint8')
                    mask |= shape_mask == 1
                valid_mask = (src_data == target_value) & mask & (ucl_src.read(1) > start_i) & (PDA_src.read(1) != 1)
                weighted_area = np.sum(frac_src.read(1)[valid_mask] * pow(10, -4)) * 100 * pow(10, -6) * 0.9
        
        area += weighted_area
        if dst_data_dict[file] is None:
            dst_data_dict[file] = np.zeros_like(src_data, dtype=int)
        dst_data_dict[file][valid_mask] = 1

    for i in range(start_i, -1, -1):
        total_area_for_i = 0
        results = []
        for file in cross_filelist:
            weighted_area, valid_mask = process_file_ucl(file, region, i, target_value, anf(file), uclf(file), PDAf(file), conserf(file))
            results.append((file, weighted_area, valid_mask))
            total_area_for_i += weighted_area

        if total_area_for_i >= area_needed - area:
            ratio = (area_needed - area) / total_area_for_i
            for file, weighted_area, valid_mask in results:
                valid_indices = np.argwhere(valid_mask)
                sample_size = int(len(valid_indices) * ratio)
                if sample_size > 0:
                    sampled_indices = valid_indices[np.random.choice(len(valid_indices), sample_size, replace=False)]
                    sampled_mask = np.zeros_like(valid_mask, dtype=bool)
                    sampled_mask[tuple(sampled_indices.T)] = True
                    with rasterio.open(anf(file)) as frac_src:
                        area += np.sum(frac_src.read(1)[sampled_mask] * pow(10, -4)) * 100 * pow(10, -6) * 0.9
                    if dst_data_dict[file] is None: dst_data_dict[file] = np.zeros_like(valid_mask, dtype=int)
                    dst_data_dict[file][sampled_mask] = 1
            break
        else:
            for file, weighted_area, valid_mask in results:
                area += weighted_area
                if dst_data_dict[file] is None: dst_data_dict[file] = np.zeros_like(valid_mask, dtype=int)
                dst_data_dict[file][valid_mask] = 1

    for file, dst_data in dst_data_dict.items():
        with rasterio.open(rf(file)) as src:
            if dst_data is None: dst_data = np.zeros((src.height, src.width), dtype=int)
            
        with rasterio.open(moutf(country, file)) as old_src:
            old_data = old_src.read(1)
            dst_data[(old_data == 1)] = 1
            dst_data[(old_data == 2)] = 2        
            
        with rasterio.open(rf(file)) as src:
            with rasterio.open(foutf(country, file), 'w', **src.meta) as dst:
                dst.write(dst_data, 1)
    return area

# ==========================================
# 4. Main function
# ==========================================

def process_country(country, data):
    try:
        country_formatted = country.replace(" ", "_").replace(".", "_")
        area_needed = find_area(country, data)
        
        if area_needed is None or np.isnan(area_needed): return ("non_found", country)

        os.makedirs(os.path.join(CONFIG["dir_process_base"], country_formatted.lower()), exist_ok=True)
        os.makedirs(os.path.join(CONFIG["dir_final_base"], country_formatted.lower()), exist_ok=True)

        vector_file = os.path.join(CONFIG["dir_vector"], f"{country}.shp")
        if not os.path.exists(vector_file): return ("non_found_vector", country)

        region = gpd.read_file(vector_file)
        files = os.listdir(CONFIG["dir_reclass_cropland"])
        cross_filelist = cross_filefind(files, region)
        
        area = calculate_all(cross_filelist, region, area_needed)

        if area[0] >= area_needed:
            target_value = 1
            start_i = calculate_i(cross_filelist, region, area_needed, target_value)
            allocate_part_sequential(cross_filelist, region, area_needed, target_value, start_i, country_formatted)
            return ("already", country)

        elif area[1] >= area_needed:
            for file in cross_filelist:
                allocate_spatial_by_level(rf(file), moutf(country_formatted, file), PDAf(file), conserf(file[:-4]) + ".tif", region, level=1)
            area_needed -= area[0]
            start_i = calculate_i20(cross_filelist, region, area_needed)
            allocate_part20_sequential(cross_filelist, region, area_needed, start_i, country_formatted)
            return ("already", country)

        elif area[2] >= area_needed:
            for file in cross_filelist:
                allocate_spatial_by_level(rf(file), moutf(country_formatted, file), PDAf(file), conserf(file[:-4]) + ".tif", region, level=2)
            area_needed -= area[1]
            target_value = 4
            start_i = calculate_i(cross_filelist, region, area_needed, target_value)
            allocate_part3040_sequential(cross_filelist, region, area_needed, target_value, start_i, country_formatted)
            return ("already", country)

        elif area[3] >= area_needed:
            for file in cross_filelist:
                allocate_spatial_by_level(rf(file), moutf(country_formatted, file), PDAf(file), conserf(file[:-4]) + ".tif", region, level=3)
            area_needed -= area[2]
            target_value = 5
            start_i = calculate_i(cross_filelist, region, area_needed, target_value)
            allocate_part3040_sequential(cross_filelist, region, area_needed, target_value, start_i, country_formatted)
            return ("already", country)

        else:
            for file in cross_filelist:
                allocate_spatial_by_level(rf(file), foutf(country_formatted, file), PDAf(file), conserf(file[:-4]) + ".tif", region, level=4)
            return ("exceed", country)

    except Exception as e:
        return ("error", (country, str(e)))

# ==========================================
# 5. Execution Script
# ==========================================

if __name__ == "__main__":
    print("Initializing process...")
    
    sheet_name = 'cropland'
    try:
        stats_data = pd.read_excel(CONFIG["excel_stats"], sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: Statistics file not found at {CONFIG['excel_stats']}")
        exit(1)

    cities = []
    for root, dirs, fileshp in os.walk(CONFIG["dir_vector"]):
        for file in fileshp:
            if file.endswith('.shp'):
                cities.append(file[:-4])
    
    non_found, non_found_vector, exceed, already = [], [], [], []
    
    max_workers = 2
    print(f"Starting parallel processing with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_country = {executor.submit(process_country, country, stats_data): country for country in cities}

        for future in as_completed(future_to_country):
            country = future_to_country[future]
            try:
                status, result_country = future.result()
                print(f"{country} -> {status}")
                if status == "non_found": non_found.append(result_country)
                elif status == "non_found_vector": non_found_vector.append(result_country)
                elif status == "exceed": exceed.append(result_country)
                elif status == "already": already.append(result_country)
                elif status == "error": print(f"⚠️ Error in {result_country[0]}: {result_country[1]}")
            except Exception as e:
                print(f"❌ Critical failure on {country}: {e}")

    print("\n==== Final Statistics ====")
    print(f"Total processed: {len(cities)}")
    print(f"Missing cropland data: {len(non_found)} ->", non_found)
    print(f"Missing vector files: {len(non_found_vector)} ->", non_found_vector)
    print(f"Area exceeded capacities: {len(exceed)} ->", exceed)
    print(f"Successfully completed: {len(already)}")