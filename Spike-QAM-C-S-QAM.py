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
        #radii[1] = np.sqrt(dmin**2 + R1**2 - 2 * dmin * R1 * np.cos(np.radians(105)))
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
    sum_f_distances = np.sum(np.abs(base_qam_points) ** 2) - min(num_spikes,4) * (half_diagonal ** 2)


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
    for i in range(min(num_spikes, 4)):  # Ensure we do not exceed the four corners
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

def calculate_papr_cs_qam(d_min, M, N, E_avg, num_spikes):
    # Use generate_c_s_qam_distances to get radii and spike distance
    R, spike_distance = generate_c_s_qam_distances(d_min, M, N, E_avg, num_spikes)
    # The PAPR is the power of the spike / average power
    papr = spike_distance ** 2 / E_avg
    return papr

def plot_papr_dmin_comparison():
    d_min_values = np.linspace(0, 0.63, 50)
    
 
    papr_spike_qam1 = [calculate_papr_spike_qam(16, 1, d_min, 1) for d_min in d_min_values]
    papr_spike_qam2 = [calculate_papr_spike_qam(16, 2, d_min, 1) for d_min in d_min_values]
    papr_spike_qam3 = [calculate_papr_spike_qam(16, 3, d_min, 1) for d_min in d_min_values]
    papr_spike_qam4 = [calculate_papr_spike_qam(16, 4, d_min, 1) for d_min in d_min_values]

    papr_cs_qam_i1 = [calculate_papr_cs_qam(d_min, 16, 4, 1, 1) for d_min in d_min_values]
    papr_cs_qam_i2 = [calculate_papr_cs_qam(d_min, 16, 4, 1, 2) for d_min in d_min_values]
    papr_cs_qam_i3 = [calculate_papr_cs_qam(d_min, 16, 4, 1, 3) for d_min in d_min_values]
    papr_cs_qam_i4 = [calculate_papr_cs_qam(d_min, 16, 4, 1, 4) for d_min in d_min_values]

 
    plt.figure(figsize=(10, 6))
    plt.plot(d_min_values, papr_spike_qam1, label='Spike-QAM, M=16, Spikes=1')
    plt.plot(d_min_values, papr_spike_qam2, label='Spike-QAM, M=16, Spikes=2')
    plt.plot(d_min_values, papr_spike_qam3, label='Spike-QAM, M=16, Spikes=3')
    plt.plot(d_min_values, papr_spike_qam4, label='Spike-QAM, M=16, Spikes=4')
    plt.plot(d_min_values, papr_cs_qam_i1, label='C-S-QAM, M=16, N=8, Spikes=1')
    plt.plot(d_min_values, papr_cs_qam_i2, label='C-S-QAM, M=16, N=8, Spikes=2')
    plt.plot(d_min_values, papr_cs_qam_i3, label='C-S-QAM, M=16, N=8, Spikes=3')
    plt.plot(d_min_values, papr_cs_qam_i4, label='C-S-QAM, M=16, N=8, Spikes=4')

    
   
   


    plt.xlabel('$d_{min}$')
    plt.ylabel('PAPR')
    plt.title('PAPR vs. $d_{min}$ Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot the comparison
plot_papr_dmin_comparison()