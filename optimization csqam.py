import numpy as np
import matplotlib.pyplot as plt

def generate_c_s_qam_distances_opt(dmin, M, N,E_avg,number_of_spikes):
    # Number of symbols per circle
    n = M // N
    
    # Initialize radii list with R1
       # Initialize radii list with R1
    radii = np.zeros(N)
  
    R1 = dmin / (2 * np.sin(np.pi / n))
    radii[0] = R1
    if N > 1:
        radii[1] = np.sqrt(dmin**2 + R1**2 - 2 * R1 * R1 * np.cos(2 * np.pi / n))
        
             
    if N> 2:
        theta = np.pi / 8
        R2 = radii[1]
        R3_candidate = R2 * np.cos(theta) + np.sqrt(R2**2 * np.cos(theta)**2 - R2**2 + dmin**2)
        radii[2] = R3_candidate

    sum_R_sqr = sum(r**2 * n for r in radii[:-1]) + radii[-1]**2 * (8 - number_of_spikes)
    sum_R_sqr = radii[0]**2*4+radii[1]**2*4+radii[2]**2*7
    remaining_energy = M * E_avg - sum_R_sqr  # This is the energy budget for the spikes

    # Calculate the spike distance based on remaining energy
    if remaining_energy <= 0:
        raise ValueError("Energy for spikes is not available, try adjusting parameters.")
    
    total_radius_sqr_for_spikes = remaining_energy / number_of_spikes 
    # The spike distance is the additional radius needed beyond the outermost radius
    spike_distance = np.sqrt(total_radius_sqr_for_spikes) 
 
    
    return radii, spike_distance



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
        radii[1] = np.sqrt(dmin**2 + R1**2 - 2 * R1 * R1 * np.cos(2 * np.pi / n))
        
             
    for i in range(2, N):
        R_prev = radii[i-1]
        R_prev_2 = radii[i-2]
        angle_diff = 2 * np.pi / n

        # Calculate radius to maintain dmin with the previous level
        R_next_from_prev = np.sqrt(dmin**2 + R_prev**2 - 2 * R_prev * R_prev * np.cos(angle_diff))

        # Calculate radius to maintain dmin with level i-2, taking into account possible angular offset
       
        angle_offset = (np.pi / n) * ((i % 2) * 2 - (i % 2))
        R_next_from_prev_2 = np.sqrt(dmin**2 + R_prev_2**2 - 2 * R_prev_2 * R_prev_2 * np.cos(2 * angle_diff + angle_offset))

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
symbols_per_circle = [4, 4, 8]
# Generate CQAM radii
c_s_qam_radii,spike_distance = generate_c_s_qam_distances_opt(d_min, M, N,1,1)
    # Place symbols on the CQAM constellation

print(c_s_qam_radii)
print(spike_distance)
print(sum(r**2*4 for r in c_s_qam_radii[:-1])+c_s_qam_radii[-1]**2 * (4 - 1)+1*spike_distance**2)
# Plotting the constellation




d_min_values = np.linspace(0.001, 0.6, 50)
papr_values = []
papr_values1 =[]
for d_min in d_min_values:
    radii, spike_distance = generate_c_s_qam_distances_opt(d_min, 16, 3, 1, 1)
    max_power = spike_distance**2
    avg_power = 1  # Given that E_avg = 1
    papr = max_power / avg_power
    papr_values.append(papr)
    radii1, spike_distance1 = generate_c_s_qam_distances(d_min, 16, 4, 1, 1)
    max_power1 = spike_distance1**2
    avg_power1 = 1  # Given that E_avg = 1
    papr1 = max_power1 / avg_power1
    papr_values1.append(papr)
plt.figure(figsize=(8, 5))
plt.plot(d_min_values, papr_values, label='CS-QAM, M=16, N=3')
plt.plot(d_min_values, papr_values1, label='CS-QAM, M=16, N=4')
plt.xlabel('$d_{\text{min}}$')
plt.ylabel('PAPR')
plt.title('PAPR vs $d_{\text{min}}$ for CS-QAM with $M=16$, $N=4$')
plt.grid(True)
plt.legend()
plt.show()