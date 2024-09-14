import numpy as np
import matplotlib.pyplot as plt


def generate_cqam_radii(dmin, M, N, E_avg, number_of_spikes):
    symbols_per_circle = [4, 6, 6]  # Updated symbol distribution
    n = M // N  # Symbols per circle without spikes
    radii = np.zeros(N)
  
    # Calculate initial radius based on the first circle
    R1 = dmin / (2 * np.sin(np.pi / symbols_per_circle[0]))
    radii[0] = R1

    radii[1]=radii[0]+dmin
    radii[2]=dmin*np.sqrt(3)/2+radii[1]*np.cos(np.pi/6)
    # Calculate remaining energy for spikes
    sum_R_sqr =radii[0]**2*4+radii[1]**2*6 +radii[2]**2 * (6 - 1)
    print(sum_R_sqr)
    remaining_energy = M * E_avg - sum_R_sqr

    if remaining_energy <= 0:
        raise ValueError("Energy for spikes is not available, try adjusting parameters.")

    spike_distance = np.sqrt(remaining_energy / number_of_spikes)
    return radii, spike_distance

def place_cqam_symbols(R, symbols_per_circle, spike_distance, spike_number):
    symbols = []
    for i, r in enumerate(R):
        n = symbols_per_circle[i]  # Use the number of symbols per circle
        start_angle = 0 if i % 2 == 0 else np.pi / n  # Staggered start angle

        for j in range(n):
            angle = start_angle + j * 2 * np.pi / n
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            symbols.append([x, y])

    # Place spike symbols
    last_radius = R[-1] + spike_distance
    for i in range(spike_number):
        angle = i * 2 * np.pi / spike_number
        x = last_radius * np.cos(angle)
        y = last_radius * np.sin(angle)
        symbols.append([x, y])

    return np.array(symbols)


d_min = 0.4
M = 16
N = 3
E_avg = 1
number_of_spikes = 1
symbols_per_circle = [4, 6, 6]  # The new distribution for the circles

# Generate radii
radii, spike_distance = generate_cqam_radii(d_min, M, N, E_avg, number_of_spikes)

# Place symbols
symbols = place_cqam_symbols(radii, symbols_per_circle, spike_distance, number_of_spikes)
#print(radii)
#print(radii[0]**2*4+radii[1]**2*6 +radii[2]**2 * (6 - 1) + 1 * spike_distance**2)
print(spike_distance)
# Plotting
plt.figure(figsize=(6, 6))
for symbol in symbols:
    plt.plot(symbol[0], symbol[1], 'bo')
for r in radii:
    circle = plt.Circle((0, 0), r, color='r', fill=False)
    plt.gca().add_artist(circle)
plt.grid(True)
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.title('CQAM Constellation with Adjusted Distribution')
plt.axis('equal')
plt.show()
