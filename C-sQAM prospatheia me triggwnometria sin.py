import numpy as np
import matplotlib.pyplot as plt

def generate_c_s_qam_distances(dmin, M, N,E_avg,number_of_spikes):
    # Number of symbols per circle
    n = M // N
    
    # Initialize radii list with R1
       # Initialize radii list with R1
    radii = np.zeros(N)
  
    R1 = dmin / (2 * np.sin(np.pi / n))
    radii[0] = R1
    if N > 1:
        if(dmin>R1):
          radii[1] = R1*np.sin(np.pi/n/2)+np.sqrt(R1**2*np.sin(np.pi/n/2)**2+dmin**2 - R1**2)
        if(dmin<R1):
           radii[1] = R1*np.sin(np.pi/n/2)+np.sqrt(R1**2*np.sin(np.pi/n/2)**2-dmin**2 + R1**2)  
    for i in range(2, N):
        R_prev = radii[i-1]
        R_prev_2 = radii[i-2]
        angle_diff = 2 * np.pi / n

        # Calculate radius to maintain dmin with the previous level
        R_next_from_prev = R_prev*np.sin(np.pi/n/2)+np.sqrt(R_prev**2*np.sin(np.pi/n/2)**2-dmin**2 + R_prev**2)

        # Calculate radius to maintain dmin with level i-2, taking into account possible angular offset
        angle_offset = (np.pi / n) * ((i % 2) * 2 - (i % 2))
        R_next_from_prev_2 = dmin+R_prev_2



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



d_min = 0.3
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