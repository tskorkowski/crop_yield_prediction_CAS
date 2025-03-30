import numpy as np
import pandas as pd

# Load the existing histogram file
histogram = np.load(r'4_Data_Sample/histogram-08001-2017.npy', allow_pickle=True)
print(histogram.shape, type(histogram))

# Convert the elements to numpy arrays and select the last 10 columns
reduced_histogram = pd.DataFrame(histogram)[[f"histogram_{x}" for x in range(200, 210)] + ['fips', 'year', 'month']]
reduced_histogram.to_csv(r'4_Data_Sample/histogram-08001-2017_reduced.csv', index=False)

print("Reduced histogram saved as 'histogram-08001-2017-reduced.csv'")
