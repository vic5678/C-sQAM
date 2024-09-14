import numpy as np
import matplotlib.pyplot as plt

def generate_cqam_radii(d_min, M, N,E_avg):
    # Number of symbols per circle
    n = M // N
    
    # Initialize radii list with R1
    R = [d_min / (2 * np.sin(np.pi / n))]
    
    # Generate radii for inner circles (R2 to RN-1) ensuring d_min is maintained
    for i in range(1, N-1):
        # Start with the previous radius and increment until d_min condition is met for all points
        R_next = R[-1] + d_min / 2  # Initial guess for the next radius
        while True:
            # Check if all points on the new circle maintain at least d_min distance
            # from all points on all previous circles
            valid = True
            for theta in np.linspace(0, 2 * np.pi, n, endpoint=False):
                new_point = np.array([R_next * np.cos(theta), R_next * np.sin(theta)])
                for r in R:
                    for theta_prev in np.linspace(0, 2 * np.pi, n, endpoint=False):
                        prev_point = np.array([r * np.cos(theta_prev), r * np.sin(theta_prev)])
                        if np.linalg.norm(new_point - prev_point) < d_min:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                R.append(R_next)
                break
            else:
                R_next += 0.01  # Small increment to try a larger radius
    
    # Compute RN based on constant average energy constraint
    # Assuming equal energy distribution across circles for simplicity
    # Target average energy
    sum_R_sqr = sum(r**2 for r in R[:-1])  # Sum of squares of all but the last radius
    R_N_sqr = (N * E_avg - sum_R_sqr)  # Solve for RN^2 from the energy equation
    if R_N_sqr > R[-1]**2:
        R.append(np.sqrt(R_N_sqr))
    else:
        raise ValueError("Unable to maintain constant average energy with given d_min and N. Try adjusting parameters.")
    
    return R

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
papr_values_cqam_n4 = []

papr_values_cs_qam_i1 = []
papr_values_cs_qam_i2 = []
papr_values_cs_qam_i3 = []
papr_values_cs_qam_i4 = []
# ... more arrays for i=2, i=3, and i=4 ...

d_min_values = np.linspace(0, 0.5, 100)

# Calculate PAPR for each d_min value for C-QAM and C-s-QAM
for d_min in d_min_values:
    papr_cqam_n4 = calculate_papr_cqam(d_min, 16, 4, 4)
   
    papr_cs_qam_i1 = calculate_papr_cs_qam(d_min, 16, 4, 4, 1)
    papr_cs_qam_i2 = calculate_papr_cs_qam(d_min, 16, 4, 4, 2)
    papr_cs_qam_i3 = calculate_papr_cs_qam(d_min, 16, 4, 4, 3)
    papr_cs_qam_i4 = calculate_papr_cs_qam(d_min, 16, 4, 4, 4)
    # ... calculations for i=2, i=3, and i=4 ...


    papr_values_cs_qam_i1.append(papr_cs_qam_i1)
    papr_values_cs_qam_i2.append(papr_cs_qam_i2)
    papr_values_cs_qam_i3.append(papr_cs_qam_i3)
    papr_values_cs_qam_i4.append(papr_cs_qam_i4)
    # ... append values for i=2, i=3, and i=4 ...

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(d_min_values, papr_values_cqam_n4, label='CQAM, M=16, N=4')

plt.plot(d_min_values, papr_values_cs_qam_i1, label='C-s-QAM, M=16, N=4, i=1')
plt.plot(d_min_values, papr_values_cs_qam_i2, label='C-s-QAM, M=16, N=4, i=2')
plt.plot(d_min_values, papr_values_cs_qam_i3, label='C-s-QAM, M=16, N=4, i=3')
plt.plot(d_min_values, papr_values_cs_qam_i4, label='C-s-QAM, M=16, N=4, i=4')
# ... plot lines for i=2, i=3, and i=4 ...

plt.grid(True)
plt.xlabel('$d_{min}$')
plt.ylabel('PAPR')
plt.title('PAPR vs $d_{min}$ Comparison')
plt.legend()
plt.show()

    