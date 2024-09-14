import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

# Parameters
M = 16
num_spikes = 1
d_min = 0.4
E_avg = 1
PAPR = 10.3
N = 4

# SNR range in dB
snr_db = np.linspace(0, 25, 10)  # Reduced number of points for faster simulation
snr_linear = 10**(snr_db / 10)

def generate_c_s_qam_distances(dmin, M, N, E_avg, number_of_spikes):
    n = M // N
    radii = np.zeros(N)
  
    R1 = dmin / (2 * np.sin(np.pi / n))
    radii[0] = R1
    if N > 1:
        radii[1] = np.sqrt(dmin**2 + R1**2 - 2 * R1 * R1 * np.cos(2 * np.pi / n))

    for i in range(2, N):
        R_prev = radii[i-1]
        R_prev_2 = radii[i-2]
        angle_diff = 2 * np.pi / n
        R_next_from_prev = np.sqrt(dmin**2 + R_prev**2 - 2 * R_prev * dmin * np.cos(angle_diff))
        angle_offset = (np.pi / n) * ((i % 2) * 2 - (i % 2))
        R_next_from_prev_2 = np.sqrt(dmin**2 + R_prev_2**2 - 2 * R_prev_2 * dmin * np.cos(2 * angle_diff + angle_offset))
        radii[i] = max(R_next_from_prev, R_next_from_prev_2)
    
    sum_R_sqr = sum(r**2 * n for r in radii[:-1]) + radii[-1]**2 * (n - number_of_spikes)
    remaining_energy = M * E_avg - sum_R_sqr  # This is the energy budget for the spikes

    if remaining_energy <= 0:
        raise ValueError("Energy for spikes is not available, try adjusting parameters.")
    
    total_radius_sqr_for_spikes = remaining_energy / number_of_spikes 
    spike_distance = np.sqrt(total_radius_sqr_for_spikes) 
 
    return radii, spike_distance

def place_c_s_qam_symbols(R, M, N, spike_distance, spike_number):
    n = M // N  # Symbols per circle
    symbols = []
    additional_spike_distance = spike_distance - R[-1]
    for i, r in enumerate(R[:-1]):
        on_axes = (i + 1) % 2 == 0
        start_angle = 0 if on_axes else np.pi / n
        for j in range(n):
            angle = start_angle + j * 2 * np.pi / n
            symbol = r * np.exp(1j * angle)
            symbols.append(symbol)

    r = R[-1]  # Radius for the outermost circle
    for j in range(n - spike_number):
        angle = j * 2 * np.pi / n
        symbol = r * np.exp(1j * angle)
        symbols.append(symbol)
    
    start_angle = (n - spike_number) * 2 * np.pi / n
    for j in range(spike_number):
        angle = start_angle + j * 2 * np.pi / n
        symbol = spike_distance * np.exp(1j * angle)
        symbols.append(symbol)

    return np.array(symbols)

def ml_detection(received_symbols, constellation_points):
    detected_symbols = []
    for symbol in received_symbols:
        distances = np.abs(symbol - constellation_points)
        detected_symbols.append(constellation_points[np.argmin(distances)])
    return np.array(detected_symbols)

def monte_carlo_simulation(M, N, radii, spike_distance, spike_number, snr_linear, num_symbols=100000):
    ser_mc = np.zeros(len(snr_linear))
    constellation_points = place_c_s_qam_symbols(radii, M, N, spike_distance, spike_number)
    
    for idx, snr in enumerate(snr_linear):
        errors = 0
        noise_variance = 1 / (2*snr)
        
        for _ in range(num_symbols // M):
            transmitted_symbols = np.random.choice(constellation_points, M)
            noise = np.sqrt(noise_variance) * (np.random.randn(M) + 1j * np.random.randn(M))
            received_symbols = transmitted_symbols + noise
            detected_symbols = ml_detection(received_symbols, constellation_points)
            errors += np.sum(transmitted_symbols != detected_symbols)
        
        ser_mc[idx] = errors / num_symbols
    
    return ser_mc


#def calculate_ser_csqam(snr_dB, N, radii, spike_distance, number_of_spikes, energy_harvested):
    nu = [5, 2, 1, 1]
    snr_linear = 10 ** (snr_dB / 10.0)
    gama = M - number_of_spikes * (spike_distance ** 2)
    gama = gama / (M - 1)
    N0 = 1 / snr_linear
    dmid = 2 * radii[N-1] * np.sin(np.pi / 4)
    pmax = Q((spike_distance - dmid) / np.sqrt(2 * N0))

    Pe_sum = 0
    for i, (R, nu_i) in enumerate(zip(radii, nu)):
        effective_power = R * 2 - energy_harvested * R ** 2
        if nu_i == 0:
            continue
        Pe_sum += nu_i * Q(np.sqrt((snr_linear*gama )) * np.sqrt(np.sqrt(effective_power)) * np.sin(np.pi / 4))
    p = Pe_sum / sum(nu)
    ser = (M - number_of_spikes) / M * p + number_of_spikes * pmax
    return ser

def calculate_ser_csqam(snr_dB, N, radii, spike_distance, number_of_spikes, energy_harvested):
    nu = [5, 2, 1, 1]
    snr_linear = 10 ** (snr_dB / 10.0)
    #gama = M - number_of_spikes * (spike_distance ** 2)
    #gama = gama / (M - number_of_spikes)
    gama = (M - number_of_spikes) / M
    #gama=1-1/16*number_of_spikes*spike_distance**2
    N0 = 1 / snr_linear
    #dmid = radii[N-1] * np.cos(np.pi / 4)
    #pmax = Q((spike_distance - dmid) / np.sqrt(2 * N0))
    pmax = Q((spike_distance-radii[-2]) / np.sqrt(2 * N0))
    R0=radii[0]
    ser = 0
    
    for i, r in enumerate(radii):
        n_neigh = nu[i]
        epsilon = energy_harvested
        effective_power = R0**2 - epsilon * R0**2
        Q_arg = np.sqrt(effective_power *2 *gama * snr_linear)* np.sin(np.pi /4)
        ser += n_neigh * Q(Q_arg)
    p=ser / N
    ser = (M - number_of_spikes) / M * p + number_of_spikes * pmax /M
    return ser

#def calculate_ser_csqam(snr_dB, N, radii, spike_distance, number_of_spikes, energy_harvested):
    nu = [5, 2, 1, 1]
    snr_linear = 10 ** (snr_dB / 10.0)
    gama = M - number_of_spikes * (spike_distance ** 2)
    gama = gama / (M - 1)
    N0 = 1 / snr_linear
    dmid = radii[N-2] * np.sin(np.pi / 4)
    pmax = Q((spike_distance - dmid) / np.sqrt(2 * N0))
    R0=radii[0]
    Pe_sum = 0
    for i, (R, nu_i) in enumerate(zip(radii, nu)):
        effective_power = R0 ** 2 - energy_harvested * R0 ** 2
        if nu_i == 0:
            continue
        Pe_sum += nu_i * Q(np.sqrt((snr_linear*gama*2 )) * np.sqrt(effective_power)* np.sin(np.pi / 4))
    p = Pe_sum / N
    ser = (M - number_of_spikes) / M * p + number_of_spikes * pmax /M
    return ser
# Generate radii and symbols for C-S-QAM
cqam_radii_with_spikes, spike_distance_cqam = generate_c_s_qam_distances(d_min, M, N, E_avg, num_spikes)

# Monte Carlo simulation for SER
ser_mc = monte_carlo_simulation(M, N, cqam_radii_with_spikes, spike_distance_cqam, num_spikes, snr_linear)

# Analytical SER calculation
ser_analytical = calculate_ser_csqam(snr_db, N, cqam_radii_with_spikes, spike_distance_cqam, num_spikes, 0)

# Plot the results
plt.figure()
plt.semilogy(snr_db, ser_mc, 'o-', label='Monte Carlo Simulation')
plt.semilogy(snr_db, ser_analytical, 'x-', label='Analytical SER')
plt.xlabel('SNR (dB)')
plt.ylabel('Symbol Error Rate (SER)')
plt.title('SER vs. SNR for C-S-QAM')
plt.grid(True)
plt.legend()
plt.show()
