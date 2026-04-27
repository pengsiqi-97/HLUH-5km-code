# %%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from scipy.ndimage import uniform_filter

# ==========================================
# 1. Global Path Configuration 
# ==========================================
CONFIG = {
    "demand_excel": "data/merged_FAOHYDE35-cropland.xlsx",
    "shapefile": "data/World_countries.shp",
    "grid_area": "data/grid_005_area.tif",
    
    # Base directories for fractional inputs
    "dir_weight": "data/weight/cropland",
    "dir_urban_recent": "data/fraction005/result/urban",
    "dir_urban_hist": "data/fraction005/tif/urban_fraction_005",
    "dir_snow": "data/fraction005/snow",
    "dir_wdpa": "data/fraction/WDPA",
    "max_fraction": "data/fraction005/max_land_fraction_resample_clip.tif",
    
    # Output directory
    "out_dir": "data/fraction005/result/cropland"
}

# ==========================================
# 2. Optimized Allocation Algorithm 
# ==========================================
def neighborhood_diffusion(arr, ksize=3):
    
    if ksize <= 1:
        return arr
    arr2 = uniform_filter(arr.astype(np.float64), size=ksize, mode='nearest')
    return (arr + arr2) / 2

def constrained_allocation(
    fao_area, weight, available_area, max_iter=50, soft_cap_ratio=0.95,
    diffusion_kernel=3, expansion_factor=1.10, contraction_factor=0.90
):
    """cropland allocation"""
    weight = weight.astype(np.float64)
    available_area = available_area.astype(np.float64)

    valid_mask = available_area > 0
    if valid_mask.sum() == 0:
        return np.zeros_like(weight), fao_area

    weight = weight * valid_mask
    weight = np.where(weight > 0, weight, 1e-6)

    wsum = weight.sum()
    if wsum == 0:
        return np.zeros_like(weight), fao_area

    raw_alloc = weight / wsum * fao_area
    cap = available_area * soft_cap_ratio
    allocated = np.minimum(raw_alloc, cap)

    for _ in range(max_iter):
        residual = fao_area - allocated.sum()
        if abs(residual) < 1e-6:
            break

        if residual > 0:
            expanded_weight = weight ** expansion_factor
            if diffusion_kernel > 1:
                expanded_weight = neighborhood_diffusion(expanded_weight, diffusion_kernel)

            remain_capacity = available_area - allocated
            remain_capacity[remain_capacity < 1e-6] = 0

            usable_weight = expanded_weight * (remain_capacity > 0)
            wsum2 = usable_weight.sum()
            if wsum2 == 0: break

            add_alloc = np.minimum(usable_weight / wsum2 * residual, remain_capacity)
            allocated += add_alloc
        else:
            contr_weight = weight ** contraction_factor
            if diffusion_kernel > 1:
                contr_weight = neighborhood_diffusion(contr_weight, diffusion_kernel)

            excess = -residual
            can_remove = allocated.copy()
            can_remove[can_remove < 1e-6] = 0

            usable_weight = contr_weight * (can_remove > 0)
            wsum2 = usable_weight.sum()
            if wsum2 == 0: break

            remove_alloc = np.minimum(usable_weight / wsum2 * excess, can_remove)
            allocated -= remove_alloc

        allocated = np.minimum(allocated, available_area)

    final_residual = fao_area - allocated.sum()
    return allocated, final_residual

# ==========================================
# 3. Main Processing Logic 
# ==========================================
def read_raster_with_nodata(path):
    """Helper to safely read rasters and zero out NoData."""
    with rasterio.open(path) as src:
        data = src.read(1).astype('float32')
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = 0.0
    return np.nan_to_num(data, nan=0.0)

def process_allocation_for_year(year, df_demand, countries_gdf, grid_area):
    """Process spatial allocation for a specific year."""
    print(f"\n--- Processing year: {year} ---")
    
    # 1. Load the historical weighting map
    weight_path = os.path.join(CONFIG["dir_weight"], f"Wcrop_weight_final_{year}.tif")
    if not os.path.exists(weight_path):
        print(f"cannot find weight file,skipping the year {year}: {weight_path}")
        
        return
        
    with rasterio.open(weight_path) as src:
        weight = src.read(1)
        profile = src.profile
        transform = src.transform

    # 2. calculate Available Area
    # Select different input data based on the year
    if year >= 1986:
        urban_path = os.path.join(CONFIG["dir_urban_recent"], f"urban_{year}_final_fraction.tif")
        urban_fraction = read_raster_with_nodata(urban_path)
        
        snow_path = os.path.join(CONFIG["dir_snow"], "snowwater-fraction.tif")
        snow_fraction = read_raster_with_nodata(snow_path)
        
        wdpa_path = os.path.join(CONFIG["dir_wdpa"], "WDPA_10.tif")
        wdpa_fraction = read_raster_with_nodata(wdpa_path)
        
        conser_path = os.path.join(CONFIG["dir_wdpa"], "conservation_au10.tif")
        conser_fraction = read_raster_with_nodata(conser_path)
        
        available_area = grid_area * (1 - urban_fraction - snow_fraction - wdpa_fraction - conser_fraction)
    else:
        # For historical data before 1986
        urban_path = os.path.join(CONFIG["dir_urban_hist"], f"{year}_clip.tif")
        urban_fraction = read_raster_with_nodata(urban_path)
        max_fraction = read_raster_with_nodata(CONFIG["max_fraction"])
        
        available_area = grid_area * (max_fraction - urban_fraction)
        
    available_area[available_area < 0] = 0

    # 3. Retrieve the demand data for the current year
    df_year = df_demand[df_demand['year'] == year].copy()
    allocated_total = np.zeros_like(weight, dtype=np.float32)
    
    country_miss = []
    country_noweight = []
    country_residual = []

    # 4. allocation by country
    for _, row in df_year.iterrows():
        country_raw = row['country']
        country_name = str(country_raw).lower().replace(' ', '_').replace('.', '_')
        
        # Correction of abnormal data
        fao_area = row['fao_filled_value'] * 10
        if pd.isna(fao_area) or country_name in ['china', 'brazil']:
            fao_area = row['hyde_value']
            
        print(f"allocating {country_name}: FAO area = {row['fao_filled_value']:.2f} km², HYDE area = {row['hyde_value']:.2f} km², used area = {fao_area:.2f} km²")
        
        
        country_geom = countries_gdf[countries_gdf['NAME'] == country_name]
        if country_geom.empty:
            country_miss.append(country_name)
            continue

        geom = [mapping(g) for g in country_geom.geometry]
        mask = geometry_mask(geometries=geom, transform=transform, invert=True, out_shape=weight.shape).astype(np.uint8)

        weight_masked = weight * mask
        area_masked = available_area * mask

        if (weight_masked * area_masked).sum() == 0:
            country_noweight.append(country_name)
            continue

        # process allocation with constraints
        allocated, residual = constrained_allocation(fao_area, weight_masked, area_masked)
        allocated_total += allocated
        
        if abs(residual) > 1e-6:
            country_residual.append((country_name, residual))

    # 5. output
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    profile.update(dtype='float32', compress='lzw', nodata=-9999.0)
    allocated_total[allocated_total < 0] = 0
    
    # save the output as area
    out_area_path = os.path.join(CONFIG["out_dir"], f"cropland_{year}_final.tif")
    with rasterio.open(out_area_path, 'w', **profile) as dst:
        dst.write(allocated_total, 1)
        
    # save the output as fraction
    allocated_fraction = allocated_total / grid_area
    allocated_fraction = np.nan_to_num(allocated_fraction, nan=0.0, posinf=0.0, neginf=0.0)
    out_frac_path = os.path.join(CONFIG["out_dir"], f"cropland_{year}_final_fraction.tif")
    with rasterio.open(out_frac_path, 'w', **profile) as dst:
        dst.write(allocated_fraction, 1)

    print(f"✅ Year {year} processing completed. Number of countries with unallocated residuals: {len(country_residual)}")

# ==========================================
# 4. Execution Controller (执行控制器)
# ==========================================
if __name__ == "__main__":
    
    # 1. loading global static data
    print("loading table and vector data...")
    df_demand = pd.read_excel(CONFIG["demand_excel"], sheet_name='cropland')
    
    countries_gdf = gpd.read_file(CONFIG["shapefile"], encoding='ISO-8859-1')
    countries_gdf['NAME'] = countries_gdf['NAME'].str.lower().str.replace(' ', '_').str.replace('.', '_')
    
    with rasterio.open(CONFIG["grid_area"]) as area_src:
        grid_area = area_src.read(1)

    # 2. define the years to run
    target_years = list(range(1900, 1950, 10)) + list(range(1950, 2024, 1))
    
    # 3. auto-run allocation for each year
    for year in target_years:
        process_allocation_for_year(year, df_demand, countries_gdf, grid_area)