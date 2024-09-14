import numpy as np
import matplotlib.pyplot as plt

def generate_c_s_qam_distances(dmin, M, N, E_avg, number_of_spikes):
    n = M // N  # Number of symbols per circle
    radii = np.zeros(N)
  
    R1 = dmin / (2 * np.sin(np.pi / n))  # Initial radius based on the first circle
    radii[0] = R1
    
    # Calculate the second circle using specified formula
    if N > 1:
        radii[1] = dmin * np.sin(np.radians(60)) + R1 * np.cos(np.pi / 8)

    # Calculate subsequent circles based on the previous circle and dmin
    for i in range(2, N):
        radii[i] = radii[i-2] + dmin

    # Calculate the remaining energy for the spikes
    sum_R_sqr = sum(r**2 * n for r in radii[:-1]) + radii[-1]**2 * (n - number_of_spikes)
    remaining_energy = M * E_avg - sum_R_sqr

    if remaining_energy <= 0:
        raise ValueError("Energy for spikes is not available, try adjusting parameters.")
    
    total_radius_sqr_for_spikes = remaining_energy / number_of_spikes 
    spike_distance = np.sqrt(total_radius_sqr_for_spikes)  # Calculate spike distance
 
    return radii, spike_distance

def place_c_s_qam_symbols(R, M, N, spike_distance, spike_number):
    n = M // N  # Symbols per circle
    symbols = []
    for i, r in enumerate(R):
        on_axes = (i + 1) % 2 == 0
        start_angle = 0 if on_axes else np.pi / n
        # Determine symbols for non-spike circles
        for j in range(n if i < N - 1 else n - spike_number):
            angle = start_angle + j * 2 * np.pi / n
            symbol = np.array([r * np.cos(angle), r * np.sin(angle)])
            symbols.append(symbol)

    # Place spike symbols on the outermost circle
    start_angle = (n - spike_number) * 2 * np.pi / n
    for j in range(spike_number):
        angle = start_angle + j * 2 * np.pi / n
        symbol = np.array([spike_distance * np.cos(angle), spike_distance * np.sin(angle)])
        symbols.append(symbol)

    return symbols

d_min = 0.28
M = 64
N = 8

# Generate CQAM radii
c_s_qam_radii, spike_distance = generate_c_s_qam_distances(d_min, M, N, 1, 1)
# Place symbols on the CQAM constellation
c_s_qam_symbols = place_c_s_qam_symbols(c_s_qam_radii, M, N, spike_distance, 1)
print(c_s_qam_radii)
print(sum(r**2 * 8 for r in c_s_qam_radii[:-1]) + c_s_qam_radii[-1]**2 * (8 - 1) + 1 * spike_distance**2)
# Plotting the constellation
plt.figure(figsize=(6, 6))
for symbol in c_s_qam_symbols:
    plt.plot(symbol[0], symbol[1], 'bo')
for r in c_s_qam_radii:
    circle = plt.Circle((0, 0), r, color='r', fill=False)
    plt.gca().add_artist(circle)
plt.grid(True)
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.title('C-s-QAM Constellation')
plt.axis('equal')
plt.show()
