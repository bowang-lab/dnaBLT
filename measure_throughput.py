import plotly.express as px
import pandas as pd
import re

# Read the file
file_path = "batchesvstime.txt"  # Update with the actual file path
with open(file_path, "r") as file:
    lines = file.readlines()

# Extract data
batches = []
times = []
pattern = re.compile(r"Processed (\d+) batches in ([\d\.]+) seconds")

for line in lines:
    match = pattern.search(line)
    if match:
        batches.append(int(match.group(1)))
        times.append(float(match.group(2)))

# Create DataFrame
df = pd.DataFrame({"Batches Processed": batches, "Time (seconds)": times})

# Create a line chart
fig = px.line(
    df,
    x="Batches Processed",
    y="Time (seconds)",
    title="Processing Time vs Batches",
    labels={
        "Batches Processed": "Batches Processed",
        "Time (seconds)": "Time (Seconds)",
    },
    markers=True,
)

# Show the plot
fig.show()
