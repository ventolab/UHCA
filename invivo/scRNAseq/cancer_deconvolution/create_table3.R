path_to_results = '/nfs/team292/vl6/CancerDeconvolution/output_ENDOMETRIAL_CANCER/'
out = normaliseExposures(paste0(path_to_results, '_fitExposures.tsv'))
data <- out$exposures

# Extract the "Intercept" row
intercept_values <- data["Intercept", ]

# Create a logical matrix indicating if cell type exceeds intercept value
above_intercept <- sweep(data, 2, intercept_values, FUN = ">")
above_intercept <- above_intercept[-nrow(data), ] # Exclude the "Intercept" row

# Compute fraction of samples for each cell type
fraction_above_intercept <- rowMeans(above_intercept)

# Compute fraction of samples above intercept and convert to percentage
fraction_above_intercept <- rowMeans(above_intercept) * 100

# Combine results into a data frame
results <- data.frame(
  Cell_Type = rownames(above_intercept),
  Percentage_Above_Intercept = fraction_above_intercept
)

# View results
print(results)
