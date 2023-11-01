import numpy as np
import matplotlib.pyplot as plt

# Data
years_worked = np.array([1, 2, 3, 4, 5, 6])
cakes_made = np.array([6500, 7805, 10835, 11230, 15870, 16387])

# Scatter plot
plt.scatter(years_worked, cakes_made, label='Data Points', color='b')

# Find the line of best fit (linear regression)
slope, intercept = np.polyfit(years_worked, cakes_made, 1)
line_of_best_fit = slope * years_worked + intercept

# Plot the line of best fit
plt.plot(years_worked, line_of_best_fit, label='Line of Best Fit', color='r')

# Calculate the correlation coefficient (r)
r = np.corrcoef(years_worked, cakes_made)[0, 1]

# Add labels and legend
plt.xlabel('Years Worked')
plt.ylabel('Cakes Made')
plt.title('Scatter Plot with Line of Best Fit')
plt.legend()

# Show the plot
plt.show()

# Print the equation of the line
print(f'Equation of the line: Cakes Made = {slope:.2f} * Years Worked + {intercept:.2f}')

# Calculate how many cakes he will make after working 10 years
cakes_after_10_years = slope * 10 + intercept
print(f'Predicted number of cakes after 10 years: {cakes_after_10_years:.2f}')

# Correlation coefficient
print(f'Correlation Coefficient (r): {r:.2f}')

# Determine the type of correlation
if r > 0:
    correlation_type = "Positive"
elif r < 0:
    correlation_type = "Negative"
else:
    correlation_type = "No Correlation"

print(f'Type of Correlation: {correlation_type}')

#Correlation Coefficient (r): 0.982
#Type of Correlation: Positive
#Using the linear regression equation, the predicted number of Crabby cakes he will make after working 10 years is approximately 22,142.16.