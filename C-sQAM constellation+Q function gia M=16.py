import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Calculate the Q-function
def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))
def generate_c_s_qam_distances(dmin, M, N,E_avg,number_of_spikes):
    # Number of symbols per circle
    n = M // N
    
    # Initialize radii list with R1
       # Initialize radii list with R1
    radii = np.zeros(N)
  
    R1 = dmin / (2 * np.sin(np.pi / n))
    radii[0] = R1
    if N > 1:
        radii[1] = np.sqrt(dmin**2 + R1**2 - 2 * R1 * R1 * np.cos(2 * np.pi / n))

    for i in range(2, N):
        R_prev = radii[i-1]
        R_prev_2 = radii[i-2]
        angle_diff = 2 * np.pi / n

        # Calculate radius to maintain dmin with the previous level
        R_next_from_prev = np.sqrt(dmin**2 + R_prev**2 - 2 * R_prev * dmin * np.cos(angle_diff))

        # Calculate radius to maintain dmin with level i-2, taking into account possible angular offset
        angle_offset = (np.pi / n) * ((i % 2) * 2 - (i % 2))
        R_next_from_prev_2 = np.sqrt(dmin**2 + R_prev_2**2 - 2 * R_prev_2 * dmin * np.cos(2 * angle_diff + angle_offset))

        radii[i] = max(R_next_from_prev, R_next_from_prev_2)
    sum_R_sqr = sum(r**2 * n for r in radii[:-1]) + radii[-1]**2 * (n - number_of_spikes)
    remaining_energy = M * E_avg - sum_R_sqr  # This is the energy budget for the spikes

    # Calculate the spike distance based on remaining energy
    if remaining_energy <= 0:
        raise ValueError("Energy for spikes is not available, try adjusting parameters.")
    
    total_radius_sqr_for_spikes = remaining_energy / number_of_spikes 
    # The spike distance is the additional radius needed beyond the outermost radius
    spike_distance = np.sqrt(total_radius_sqr_for_spikes) 
 
    
    return radii, spike_distance



def place_c_s_qam_symbols(R, M, N,spike_distance,spike_number):
    n = M // N  # Symbols per circle
    symbols = []
    additional_spike_distance = spike_distance - R[-1]
    for i, r in enumerate(R[:-1]):
        # Determine if symbols on this circle should be on the axes
        on_axes = (i + 1) % 2 == 0

        # Starting angle: Align with axes for even circles, offset for odd
        start_angle = 0 if on_axes else np.pi / n

        for j in range(n):
            angle = start_angle + j * 2 * np.pi / n
            symbol = np.array([r * np.cos(angle), r * np.sin(angle)])
            symbols.append(symbol)

        # Place non-spike symbols on the outermost circle
    r = R[-1]  # Radius for the outermost circle
    for j in range(n - spike_number):
        angle = j * 2 * np.pi / n
        symbol = np.array([r * np.cos(angle), r * np.sin(angle)])
        symbols.append(symbol)
    
    # Place spike symbols
    # Starting angle for spikes can be offset from the last non-spike symbol
    start_angle = (n - spike_number) * 2 * np.pi / n
    for j in range(spike_number):
        angle = start_angle + j * 2 * np.pi / n
        # Spike distance is the total radius from the origin to the spike
        symbol = np.array([spike_distance * np.cos(angle), spike_distance * np.sin(angle)])
        symbols.append(symbol)

    return symbols



d_min = 0.4
M = 16
N = 4

# Generate CQAM radii
c_s_qam_radii,spike_distance = generate_c_s_qam_distances(d_min, M, N,1,1)
    # Place symbols on the CQAM constellation
c_s_qam_symbols = place_c_s_qam_symbols(c_s_qam_radii, M, N,spike_distance,1)
print(c_s_qam_radii)
print(sum(r**2*4 for r in c_s_qam_radii[:-1])+c_s_qam_radii[-1]**2 * (4 - 1)+1*spike_distance**2)
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


# Probability of error Pe for given Eb/No in dB and radii
def Pe(snr_dB,N,radii,spike_distance, nu,number_of_spikes,energy_harvested):
    # Convert dB to linear scale
    
    snr_linear=10 ** (snr_dB / 10.0)
 
    gama= M-number_of_spikes*(spike_distance**2)
    gama= gama/(M-1)
    N0=1/snr_linear
    dmid=2*radii[N-1]*np.sin(np.pi/4)
    pmax= Q((spike_distance-dmid)/np.sqrt(2*N0))
    Pe_sum = 0
    for i, (R, nu_i) in enumerate(zip(radii, nu)):
        # Skip the level if nu(i) is 0
        if nu_i == 0:
            continue
        Pe_sum += nu_i * Q(np.sqrt((3*6* snr_linear) / (len(radii) - 1)) * np.sqrt(R**2-energy_harvested*R**2))
    p= Pe_sum / sum(nu)
    ser=(M-number_of_spikes)/M * p +number_of_spikes*pmax
    return ser


# Given constellation parameters and nu values

symbols_per_level = 4
N = 4
nu = [4, 2, 1, 1]  # Number of nearest neighbors for each level
dmin=0.4
number_of_spikes=1
E_avg=1
# Calculate the minimum distance and the radii

radii,spike_distance = generate_c_s_qam_distances(dmin, M, N, E_avg,number_of_spikes)

# Range of Eb/No values in dB and their corresponding Pe values
Eb_No_dB_values = np.arange(0, 22, 1)


Pe_values = [Pe(Eb_No_dB,N, radii,spike_distance, nu,number_of_spikes,energy_harvested=0) for Eb_No_dB in Eb_No_dB_values]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(Eb_No_dB_values, Pe_values, 'o-', label='Probability of Error (Pe)')
plt.yscale('log')
plt.xlabel('Eb/No (dB)')
plt.ylabel('Probability of Error (Pe)')
plt.title('Pe vs. Eb/No for 4-Level Circular Constellation')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(Eb_No_dB_values)
plt.xlim(0,20)
plt.xticks(np.arange(0, 22, 2)) 
plt.legend()
plt.show()

