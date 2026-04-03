import pandas as pd

df_raw = pd.read_excel(r"C:\Users\alnam\OneDrive\Documents\ration_project1\data\merged_output.xlsx")
print(df_raw.columns.tolist())
print(df_raw['employment_type'].value_counts().head(20))  # adjust column name if needed