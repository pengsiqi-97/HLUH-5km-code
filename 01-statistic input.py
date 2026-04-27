# %%
import os
import pandas as pd
import numpy as np

# ==========================================
# 0. Setup Directories for Reproducibility
# ==========================================
INPUT_DIR = "./data"
OUTPUT_DIR = "./output"


os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# ==========================================
# PART 1: Cropland Data Processing (FAO & HYDE)/replace that for grazing land when calculating that
# ==========================================
print("Starting Part 1: Processing Cropland Data...")

fao_path = os.path.join(INPUT_DIR, "FAOSTAT_data.xlsx")
hyde_path = os.path.join(INPUT_DIR, "HYDE35-statistic.xlsx")

fao_df = pd.read_excel(fao_path, sheet_name='cropland')
hyde_df = pd.read_excel(hyde_path, sheet_name='cropland')

# Convert HYDE to long format
value_vars = [col for col in hyde_df.columns if col not in ["country", "region"]]
hyde_long = pd.melt(
    hyde_df,
    id_vars=["country"],
    value_vars=value_vars,
    var_name="year",
    value_name="hyde_value"
)

# Standardize columns and types
fao_df = fao_df.rename(columns={"Country": "country", "Value": "fao_value"})
fao_df["year"] = fao_df["year"].astype(str)
hyde_long["year"] = hyde_long["year"].astype(str)
fao_df["country"] = fao_df["country"].str.strip()
hyde_long["country"] = hyde_long["country"].str.strip()

# Merge data
merged_df = pd.merge(hyde_long, fao_df, on=["country", "year"], how="left")

# Data preparation for estimation
merged_df["year"] = merged_df["year"].astype(int)
merged_df = merged_df.sort_values(by=["country", "year"]).reset_index(drop=True)

# Dual-direction estimation function
def estimate_dual_direction(group):
    result = []
    years = group["year"].values
    fao_values = group["fao_value"].values
    hyde_values = group["hyde_value"].values
    fao_available = [i for i, v in enumerate(fao_values) if not pd.isna(v)]

    for idx, year in enumerate(years):
        if not pd.isna(fao_values[idx]):
            result.append(fao_values[idx])
        else:
            prev_idx = next((i for i in reversed(fao_available) if i < idx), None)
            next_idx = next((i for i in fao_available if i > idx), None)
            estimates = []
            if prev_idx is not None and hyde_values[prev_idx] not in [0, np.nan]:
                ratio = (fao_values[prev_idx] * 10) / hyde_values[prev_idx]
                estimates.append(hyde_values[idx] * ratio / 10)
            if next_idx is not None and hyde_values[next_idx] not in [0, np.nan]:
                ratio = (fao_values[next_idx] * 10) / hyde_values[next_idx]
                estimates.append(hyde_values[idx] * ratio / 10)
            result.append(np.mean(estimates) if estimates else np.nan)
    return pd.Series(result, index=group.index)

# Apply estimation
merged_df["fao_filled_value"] = merged_df.groupby("country").apply(estimate_dual_direction).reset_index(drop=True)

# Fill missing countries with raw HYDE values
fao_country_set = set(merged_df.loc[merged_df["fao_value"].notna(), "country"].unique())
all_country_set = set(merged_df["country"].unique())
condition = merged_df["country"].isin(all_country_set - fao_country_set)
merged_df.loc[condition, "fao_filled_value"] = merged_df.loc[condition, "hyde_value"]

# Demand calculation logic
def compute_demand(group):
    ac_1500 = group.loc[group['year'] == 1500, 'fao_filled_value'].values
    ac_2010 = group.loc[group['year'] == 2010, 'fao_filled_value'].values

    if len(ac_1500) == 0 or len(ac_2010) == 0 or np.isnan(ac_1500[0]) or np.isnan(ac_2010[0]):
        group['demand'] = np.nan
        return group

    diff_total = abs(ac_2010[0] - ac_1500[0])

    def calc(row):
        if row['year'] >= 2010: return 1.0
        if row['year'] < 1500: return 0.0
        if pd.isna(row['fao_filled_value']): return np.nan
        return 0.0 if diff_total == 0 else min(1.0, abs(row['fao_filled_value'] - ac_1500[0]) / diff_total)

    group['demand'] = group.apply(calc, axis=1)
    return group

merged_df = merged_df.groupby('country').apply(compute_demand)
merged_df['diff_filled'] = merged_df['hyde_value'] - merged_df['fao_filled_value'] * 10

# Export Cropland results
cropland_out_path = os.path.join(OUTPUT_DIR, "merged_FAOHYDE35-cropland-demand.xlsx")
merged_df.to_excel(cropland_out_path, index=False)
print(f"✅ Part 1 Complete: Cropland data saved to {cropland_out_path}\n")


# %%
# ==========================================
# PART 2: Urban Data Processing (GLC & HYDE)
# ==========================================
print("Starting Part 2: Processing Urban Data...")

glc_path = os.path.join(INPUT_DIR, "GLC-urban.xlsx")
glc_df = pd.read_excel(glc_path, sheet_name='urban')
hyde_df_urban = pd.read_excel(hyde_path, sheet_name='urban')

# Convert HYDE Urban to long format
hyde_urban_vars = [col for col in hyde_df_urban.columns if col not in ["country", "region"]]
hyde_long_urban = pd.melt(
    hyde_df_urban,
    id_vars=["country"],
    value_vars=hyde_urban_vars,
    var_name="year",
    value_name="hyde_value"
)
hyde_long_urban["year"] = hyde_long_urban["year"].astype(int)

# Convert GLC to long format
glc_value_vars = [col for col in glc_df.columns if col not in ["Country", "region"]]
glc_long = pd.melt(
    glc_df,
    id_vars=["Country"],
    value_vars=glc_value_vars,
    var_name="year",
    value_name="GLC_value"
)
glc_long = glc_long.rename(columns={"Country": "country"})
glc_long["year"] = glc_long["year"].astype(int)

# Pivot HYDE for backwards calculation
hyde_pivot = hyde_long_urban.pivot(index="country", columns="year", values="hyde_value")
years = sorted([y for y in hyde_pivot.columns if 1900 <= y <= 1985])

# Initialize GLC starting point (1985)
glc_1985 = glc_long[glc_long["year"] == 1985].set_index("country")["GLC_value"]
glc_est = pd.DataFrame(index=hyde_pivot.index, columns=years)
glc_est.loc[:, 1985] = glc_1985

# Backwards iteration (1985 to 1900)
for y in reversed(years):
    if y == 1985: continue
    next_y = y + 1
    # Prevent division by zero
    hyde_ratio = np.where(hyde_pivot[next_y] == 0, 0, hyde_pivot[y] / hyde_pivot[next_y])
    glc_est[y] = glc_est[next_y] * hyde_ratio

# Reshape and merge
glc_est_long = glc_est.reset_index().melt(
    id_vars="country",
    var_name="year",
    value_name="GLC_value"
)
glc_est_long["year"] = glc_est_long["year"].astype(int)

glc_post1985 = glc_long[glc_long["year"] > 1985]
final_glc = pd.concat([glc_est_long, glc_post1985], ignore_index=True)
final_glc = final_glc.sort_values(by=["country", "year"])

# Export Urban results
urban_out_path = os.path.join(OUTPUT_DIR, "GLC-urban-iter1900.xlsx")
final_glc.to_excel(urban_out_path, index=False)
print(f"✅ Part 2 Complete: GLC Urban data iteratively gap-filled and saved to {urban_out_path}")
print("All data processing finished successfully!")