import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = 'netvar_Heartfulness_Institute_Event_Rollin_2.csv'  # Update path as necessary
data = pd.read_csv(file_path)

# Calculate the cumulative sum of the Network Coherence values
data['Cumulative Coherence'] = data['Network Coherence'].cumsum()

# Plotting
plt.figure(figsize=(12, 6))

# Plot the cumulative coherence (red curve)
plt.plot(data['Timestamp'], data['Cumulative Coherence'], color='red', label='Cumulative Coherence')

# Plot the envelope (blue curve)
plt.plot(data['Timestamp'], data['envelope'], color='blue', label='Envelope (95% Confidence)')

# Add labels and legend
plt.xlabel("Timestamp")
plt.ylabel("Values")
plt.title("Cumulative Network Coherence and Envelope")
plt.legend()
plt.grid(True)

# Display the plot
plt.show()