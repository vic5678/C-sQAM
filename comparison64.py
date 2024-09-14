
import numpy as np
import matplotlib.pyplot as plt

def generate_cqam_radii(dmin, M, N,E_avg):
    # Number of symbols per circle
    n = M // N
    
    # Initialize radii list with R1
    radii = np.zeros(N)
  
    R1 = dmin / (2 * np.sin(np.pi / n))
    radii[0] = R1
    if N > 1:
        radii[1] = np.sqrt(dmin**2 + R1**2 - 2 * R1 * dmin * np.cos(2 * np.pi / n))

    for i in range(2, N):
        R_prev = radii[i-1]
        R_prev_2 = radii[i-2]
        angle_diff = 2 * np.pi / n

        # Calculate radius to maintain dmin with the previous level
        R_next_from_prev = np.sqrt(dmin**2 + R_prev**2 - 2 * R_prev * dmin * np.cos(angle_diff))

        # Calculate radius to maintain dmin with level i-2, taking into account possible angular offset
        angle_offset = (np.pi / n) * ((i % 2) * 2 - (i % 2))
        R_next_from_prev_2 = dmin+R_prev_2

        radii[i] = max(R_next_from_prev, R_next_from_prev_2)

    if N > 1:
       radii[N-1]= np.sqrt(N - np.sum(radii[:N-1]**2))

    return radii
# Probability of erro

def place_cqam_symbols(R, M, N):
    n = M // N  # Symbols per circle
    symbols = []

    for i, r in enumerate(R):
        # Determine if symbols on this circle should be on the axes
        on_axes = (i + 1) % 2 == 0

        # Starting angle: Align with axes for even circles, offset for odd
        start_angle = 0 if on_axes else np.pi / n

        for j in range(n):
            angle = start_angle + j * 2 * np.pi / n
            symbol = np.array([r * np.cos(angle), r * np.sin(angle)])
            symbols.append(symbol)

    return symbols


def generate_c_s_qam_distances(dmin, M, N,E_avg,number_of_spikes):
    # Number of symbols per circle
    n = M // N
    
    # Initialize radii list with R1
       # Initialize radii list with R1
    radii = np.zeros(N)
  
    R1 = dmin / (2 * np.sin(np.pi / n))
    radii[0] = R1
    if N > 1:
        radii[1] = np.sqrt(dmin**2 + R1**2 - 2 * R1 * dmin * np.cos(2 * np.pi / n))

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


def generate_spike_qam(base_size, num_spikes, d_min, E_avg):
    # Calculate the number of points per side for the square QAM
    side_length = int(np.sqrt(base_size))
    
    # Determine the step size based on d_min
    step_size = d_min / np.sqrt(2)  # diagonal distance
    
    # Generate the base square QAM constellation points
    x_coords = np.linspace(-step_size * (side_length // 2), step_size * (side_length // 2), side_length)
    y_coords = np.linspace(-step_size * (side_length // 2), step_size * (side_length // 2), side_length)
    xx, yy = np.meshgrid(x_coords, y_coords)
    base_qam_points = xx.flatten() + 1j * yy.flatten()
    
    # Calculate the half-diagonal (distance to a corner point)
    half_diagonal = np.sqrt((max(x_coords) ** 2) + (max(y_coords) ** 2))

    # Calculate the sum of squared distances of the points from the origin,
    # excluding the energy contribution from the corner points that will become spikes
    sum_f_distances = np.sum(np.abs(base_qam_points) ** 2) - min(num_spikes,4 ) * (half_diagonal ** 2)


    # Calculate the total energy budget based on E_avg and base_size
    total_energy_budget = base_size * E_avg

    # Remaining energy for spikes
    remaining_energy_for_spikes = total_energy_budget - sum_f_distances

    if remaining_energy_for_spikes <= 0:
        raise ValueError("Not enough energy budget for spikes, increase E_avg or reduce base_size")

    # Calculate the spike distance
    spike_energy_per_spike = remaining_energy_for_spikes / min(num_spikes, 4)  # Energy per spike
    spike_distance = np.sqrt(spike_energy_per_spike)  # Distance from the origin for each spike

    # Determine the positions for the corner points to be replaced by spikes
    corner_indices = [
        0,  # Bottom-left
        side_length - 1,  # Bottom-right
        len(base_qam_points) - side_length,  # Top-left
        len(base_qam_points) - 1  # Top-right
    ]
    

    # Replace the corner points with spikes, ensuring they maintain the minimum distance d_min
    for i in range(min(num_spikes,4 )):  # Ensure we do not exceed the four corners
        index = corner_indices[i]
        # Move the corner point outwards by spike_distance to create the spike
        angle = np.angle(base_qam_points[index])
        base_qam_points[index] += np.exp(1j * angle) * spike_distance

    return base_qam_points,spike_distance


def calculate_papr_spike_qam(base_size, num_spikes, d_min, E_avg):
    qam_points,spike_distance1= generate_spike_qam(base_size, num_spikes, d_min, E_avg)
    # Calculate PAPR: max power / average power
    papr = spike_distance1 ** 2 / E_avg
    return papr



def calculate_papr_cqam(d_min, M, N, E_avg):
    radii = generate_cqam_radii(d_min, M, N, E_avg)
    rmax = radii[-1]
    papr = rmax**2 / E_avg
    return papr

def calculate_papr_cs_qam(d_min, M, N, E_avg, num_spikes):
    radii, spike_distance = generate_c_s_qam_distances(d_min, M, N, E_avg, num_spikes)
    rmax = spike_distance if num_spikes > 0 else radii[-1]
    papr = rmax**2 / E_avg
    return papr

# Initialize arrays to store PAPR values for different configurations
papr_values_cqam_n = []
papr_values_cqam_n8 = []
papr_values_cs_qam_i1 = []
papr_values_cs_qam_i2 = []
papr_values_cs_qam_i3 = []
papr_values_cs_qam_i4 = []
papr_values_cs_qam_i5 = []
papr_values_cs_qam_i6 = []
papr_values_cs_qam_i7 = []
papr_values_cs_qam_i8 = []
papr_values_cs_qam_i11 = []
papr_values_cs_qam_i12 = []
papr_values_s_qam_i1 = []
papr_values_s_qam_i2 = []
papr_values_s_qam_i3 = []
papr_values_s_qam_i4 = []
papr_values_c_qam_i1 = []
papr_values_c_qam_i2 = []

# ... more arrays for i=2, i=3, and i= ...

d_min_values = np.linspace(0, 0.29, 100)

# Calculate PAPR for each d_min value for C-QAM and C-s-QAM
for d_min in d_min_values:
    
    
    papr_cs_qam_i1 = calculate_papr_cs_qam(d_min, 64, 8, 1, 1)
    papr_cs_qam_i2 = calculate_papr_cs_qam(d_min, 64, 8, 1, 2)
    papr_cs_qam_i3 = calculate_papr_cs_qam(d_min, 64, 8, 1, 3)
    papr_cs_qam_i4 = calculate_papr_cs_qam(d_min, 64, 8, 1,4)
    papr_cs_qam_i5 = calculate_papr_cs_qam(d_min, 64, 8, 1,5)
    papr_cs_qam_i6 = calculate_papr_cs_qam(d_min, 64, 8, 1,6)
    papr_cs_qam_i7 = calculate_papr_cs_qam(d_min, 64, 8, 1,7)
    papr_cs_qam_i8 = calculate_papr_cs_qam(d_min, 64, 8, 1,8)
    papr_cs_qam_i11 = calculate_papr_cs_qam(d_min, 64, 6, 1,1)
    papr_cs_qam_i12 = calculate_papr_cs_qam(d_min, 64, 6, 1,2)

    papr_s_qam_i1 = calculate_papr_spike_qam(64, 1, d_min, 1)
    papr_s_qam_i2 = calculate_papr_spike_qam(64, 2, d_min, 1)
    papr_s_qam_i3 = calculate_papr_spike_qam(64, 3, d_min, 1)
    papr_s_qam_i4 = calculate_papr_spike_qam(64, 4, d_min, 1)
    papr_c_qam_i1 = calculate_papr_cqam(d_min,64, 8, 1)
    papr_c_qam_i2 = calculate_papr_cqam(d_min,64, 6, 1)
    
    # ... calculations for i=2, i=3, and i= ...

    papr_values_cs_qam_i1.append(papr_cs_qam_i1)
   
    papr_values_cs_qam_i2.append(papr_cs_qam_i2)
    papr_values_cs_qam_i3.append(papr_cs_qam_i3)
    papr_values_cs_qam_i4.append(papr_cs_qam_i4)
    papr_values_cs_qam_i5.append(papr_cs_qam_i5)
    papr_values_cs_qam_i6.append(papr_cs_qam_i6)
    papr_values_cs_qam_i7.append(papr_cs_qam_i7)
    papr_values_cs_qam_i8.append(papr_cs_qam_i8)
    papr_values_cs_qam_i11.append(papr_cs_qam_i11)
    papr_values_cs_qam_i12.append(papr_cs_qam_i12)

    papr_values_s_qam_i1.append(papr_s_qam_i1)
    papr_values_s_qam_i2.append(papr_s_qam_i2)
    papr_values_s_qam_i3.append(papr_s_qam_i3)
    papr_values_s_qam_i4.append(papr_s_qam_i4)
    # ... append values for i=2, i=3, and i= ...
    papr_values_c_qam_i1.append(papr_c_qam_i1)
    papr_values_c_qam_i2.append(papr_c_qam_i2)
# Plotting
plt.figure(figsize=(10, 6))


plt.plot(d_min_values, papr_values_cs_qam_i1, label='C-s-QAM, M=64, N=8, i=1')
plt.plot(d_min_values, papr_values_cs_qam_i2, label='C-s-QAM, M=64, N=8, i=2')
plt.plot(d_min_values, papr_values_cs_qam_i3, label='C-s-QAM, M=64, N=8, i=3')
plt.plot(d_min_values, papr_values_cs_qam_i4, label='C-s-QAM, M=64, N=8, i=4')
plt.plot(d_min_values, papr_values_cs_qam_i5, label='C-s-QAM, M=64, N=8, i=5')
plt.plot(d_min_values, papr_values_cs_qam_i6, label='C-s-QAM, M=64, N=8, i=6')
plt.plot(d_min_values, papr_values_cs_qam_i7, label='C-s-QAM, M=64, N=8, i=7')
plt.plot(d_min_values, papr_values_cs_qam_i8, label='C-s-QAM, M=64, N=8, i=8')
plt.plot(d_min_values, papr_values_cs_qam_i11, label='C-s-QAM, M=64, N=6, i=1')
plt.plot(d_min_values, papr_values_cs_qam_i12, label='C-s-QAM, M=64, N=6, i=2')
plt.plot(d_min_values, papr_values_s_qam_i1, label='s-QAM, M=64,  i=1')
plt.plot(d_min_values, papr_values_s_qam_i2, label='s-QAM, M=64,  i=2')
plt.plot(d_min_values, papr_values_s_qam_i3, label='s-QAM, M=64,  i=3')
plt.plot(d_min_values, papr_values_s_qam_i4, label='s-QAM, M=64,  i=4')
plt.plot(d_min_values, papr_values_c_qam_i1, label='c-QAM, M=64,  N=8')
plt.plot(d_min_values, papr_values_c_qam_i2, label='c-QAM, M=64,  N=6' )
# ... plot lines for i=2, i=3, and i= ...

plt.grid(True)
plt.xlabel('$d_{min}$')
plt.ylabel('PAPR')
plt.title('PAPR vs $d_{min}$ Comparison')
plt.legend()
plt.show()

    