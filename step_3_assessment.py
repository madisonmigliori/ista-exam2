hospital_df.dtypes

hospital_df_clean = hospital_df_clean.copy()

hospital_df_clean["Number of Readmissions"] = pd.to_numeric(hospital_df_clean["Number of Readmissions"], errors="coerce")
hospital_df_clean["Start Date"] = pd.to_datetime(hospital_df_clean["Start Date"])
hospital_df_clean["End Date"] = pd.to_datetime(hospital_df_clean["End Date"])
