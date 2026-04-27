import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterstats import zonal_stats
from tqdm import tqdm

# ==========================================
# 1. Global Config & Path Management
# ==========================================
CONFIG = {
    "shapefile": "data/world-countries.shp",
    "grid_area": "data/grid_005_area.tif",
    
    # Cropland
    "cropland": {
        "ref_fraction": "data/fraction005/cropland2023.tif", # baseline year
        "hist_fraction_dir": "data/fraction005/cropland_historical",
        "suitability_dir": "data/fraction005/cropland_suitability",
        "demand_excel": "data/merged_FAOHYDE35-demand.xlsx",
        "demand_sheet": "cropland",
        "out_excel_dir": "data/croplandexcel",
        "out_tif_dir": "data/fraction005/result/cropland"
    },
    
    # Grazing
    "grazing": {
        "ref_fraction": "data/fraction005/grazing2023.tif",
        "hist_fraction_dir": "data/fraction005/grazing_historical",
        "suitability_dir": "data/fraction005/grazing_suitability", 
        "demand_excel": "data/merged_FAOHYDE35-demand.xlsx",
        "demand_sheet": "grazing",
        "out_excel_dir": "data/grazingexcel",
        "out_tif_dir": "data/fraction005/result/grazing"
    }
}

# ==========================================
# 2. main functions
# ==========================================
def process_yearly_allocation(year, land_type, gdf):
    """
    land_type: "cropland" 或 "grazing"
    """
    cfg = CONFIG[land_type]
    
    # Dynamic path construction for the current year ---
    raster1 = cfg["ref_fraction"]
    raster2 = os.path.join(cfg["hist_fraction_dir"], f"{land_type}_{year}_historical_fraction.tif")
    
    
    raster3 = os.path.join(cfg["suitability_dir"], f"{land_type}_suitability_{year}_clip.tif")
    
    
    if not os.path.exists(raster2):
        print(f"Historical raster not found, skipping {year}: {raster2}")
        return

    
    with rasterio.open(raster1) as src1, rasterio.open(raster2) as src2:
        arr1 = src1.read(1)
        arr2 = src2.read(1)
        profile = src1.profile
        transform = src1.transform

    nodata1 = profile.get('nodata', -9999)
    nodata2 = src2.nodata if src2.nodata is not None else -9999

    mask1 = (arr1 > 0) & (~np.isclose(arr1, nodata1))
    mask2 = (arr2 > 0) & (~np.isclose(arr2, nodata2))

    only1 = (mask1 & ~mask2).astype(np.uint8)
    only2 = (mask2 & ~mask1).astype(np.uint8)
    both  = (mask1 & mask2).astype(np.uint8)

    
    def count_pixels_in_memory(mask_array):
        stats = zonal_stats(gdf, mask_array, stats=["sum"], affine=transform, nodata=0)
        return [s['sum'] if s['sum'] is not None else 0 for s in stats]

    gdf['Gt1'] = count_pixels_in_memory(only1)
    gdf['Gt2'] = count_pixels_in_memory(only2)
    gdf['Gt3'] = count_pixels_in_memory(both)

    
    os.makedirs(cfg["out_excel_dir"], exist_ok=True)
    gt_output = os.path.join(cfg["out_excel_dir"], f"Gt_{year}.xlsx")
    gdf[['NAME', 'Gt1', 'Gt2', 'Gt3']].to_excel(gt_output, index=False)

    
    df_demand = pd.read_excel(cfg["demand_excel"], sheet_name=cfg["demand_sheet"])
    
    
    df_demand.columns = [c.lower() for c in df_demand.columns] 
    demand_col = "name" if "name" in df_demand.columns else "country"
    
    df_demand = df_demand[(df_demand['year'] == year)][[demand_col, "demand"]].dropna()
    df_demand = df_demand.rename(columns={demand_col: "country"})
    df_demand['country'] = df_demand['country'].astype(str).str.lower().str.replace(' ', '_').str.replace('.', '_')

    
    df_gt = gdf[['NAME', 'Gt1', 'Gt2']].copy()
    df_gt = df_gt.rename(columns={"NAME": "country"})
    df_gt['country'] = df_gt['country'].astype(str).str.lower().str.replace(' ', '_').str.replace('.', '_')
    df_gt = df_gt.drop_duplicates(subset="country")

    
    df_merged = pd.merge(df_demand, df_gt, how="inner", on="country")
    df_merged["Ct1"] = (df_merged["demand"] * df_merged["Gt1"]).round().astype(int)
    df_merged["Ct2"] = ((1 - df_merged["demand"]) * df_merged["Gt2"]).round().astype(int)
    
    ct_output = os.path.join(cfg["out_excel_dir"], f"{land_type}_FAO_data_with_ct_{year}.xlsx")
    df_merged.to_excel(ct_output, index=False)

    
    param_df = df_merged.dropna(subset=["Ct1", "Ct2"]).set_index('country')
    
    lu_power = min((year - 1500) * (1 / 510), 1.0)
    
    
    ref = np.where(arr1 == nodata1, 0, arr1)
    ref = np.where(ref > 1, 1, ref)
    hist = np.where(arr2 == nodata2, 0, arr2)

    result = np.zeros_like(ref, dtype=np.float32)

    for _, row in gdf.iterrows():
        country_id = str(row['NAME']).lower().replace(' ', '_').replace('.', '_')
        if country_id not in param_df.index:
            continue

        Ct1 = int(param_df.loc[country_id, "Ct1"])
        Ct2 = int(param_df.loc[country_id, "Ct2"])
        
        
        cmask = geometry_mask([row.geometry], transform=transform, invert=True, out_shape=ref.shape)

        
        m3 = (ref > 0) & (hist > 0) & cmask
        result[m3] = lu_power * ref[m3] + (1 - lu_power) * hist[m3]

        
        m1 = (ref > 0) & (hist == 0) & cmask
        values1 = ref[m1]
        idx1 = np.argwhere(m1)
        if len(values1) > 0:
            top1 = np.argsort(values1)[-min(Ct1, len(values1)):]
            for i in top1:
                y, x = idx1[i]
                result[y, x] = lu_power * ref[y, x]

        
        m2 = (ref == 0) & (hist > 0) & cmask
        values2 = hist[m2]
        idx2 = np.argwhere(m2)
        if len(values2) > 0:
            top2 = np.argsort(values2)[-min(Ct2, len(values2)):]
            for i in top2:
                y, x = idx2[i]
                result[y, x] = (1 - lu_power) * hist[y, x]

    
    os.makedirs(cfg["out_tif_dir"], exist_ok=True)
    out_tif = os.path.join(cfg["out_tif_dir"], f"W{land_type}_weight_final_{year}.tif")
    
    profile.update(dtype='float32', nodata=0, compress='lzw')
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(result, 1)

    print(f"✅ {land_type.capitalize()} {year} 处理完成: {out_tif}")


# ==========================================
# 3. main section
# ==========================================
if __name__ == "__main__":
    
    
    print("Loading global vector data...")
    global_gdf = gpd.read_file(CONFIG["shapefile"])
    
    #
    years_to_run_cropland = range(1900,2023) 
    for y in years_to_run_cropland:
        process_yearly_allocation(year=y, land_type="cropland", gdf=global_gdf)
        
    # 示例 2：处理草地 (1981 - 2023)
    years_to_run_grazing = range(1981, 2023)
    for y in years_to_run_grazing:
        process_yearly_allocation(year=y, land_type="grazing", gdf=global_gdf)