from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

# Dummy workload
total = sum(i * i for i in range(1_000_000))

emissions = tracker.stop()
print(f"Emissions: {emissions} kg CO2eq")
