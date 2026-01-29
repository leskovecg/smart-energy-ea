import pandas as pd

orig = pd.read_csv(r"C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea\data\simulation_security_labels_n-1.csv")
latest = pd.read_csv(r"C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea\data\from_minio\simulation_security_labels_n-1_latest.csv")

orig_cols = set(orig.columns)
latest_cols = set(latest.columns)

only_in_orig = orig_cols - latest_cols
only_in_latest = latest_cols - orig_cols

print("Only in original:", only_in_orig)
print("Only in latest:", only_in_latest)
