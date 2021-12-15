import numpy as np
from numba import jit
from collections import namedtuple

PI = np.pi

# Define TES DC model
def tes_dc_model(squid_input_inductor = None, shunt_resistor = None, temperature_focal_plane = None, tes_normal_resistance = None, tes_log_sensitivity_alpha = None, tes_leg_thermal_carrier_exponent = None, tes_normal_time_constant = None, optical_loading_power = None, tes_saturation_power = None, tes_transition_temperature = None, tes_leg_thermal_conductivity = None, tes_heat_capacity = None, biasing_current = None):

	tesdc = namedtuple('tes_dc_model', ['squid_input_inductor', 'shunt_resistor', 'temperature_focal_plane', 'tes_normal_resistance', 'tes_log_sensitivity_alpha', 'tes_leg_thermal_carrier_exponent', 'tes_normal_time_constant', 'optical_loading_power', 'tes_saturation_power', 'tes_transition_temperature', 'tes_leg_thermal_conductivity', 'tes_heat_capacity', 'biasing_current'])

	if squid_input_inductor == None:
	 
		squid_input_inductor = 65.e-6 # SQUID input inductor [henry]
		
	else: 
	
		squid_input_inductor = squid_input_inductor
		
	if shunt_resistor == None:

		shunt_resistor = 0.02 # shunt resistor [ohm]
		
	else:
	
		shunt_resistor = shunt_resistor
	
	if temperature_focal_plane == None:
	
		temperature_focal_plane = 0.1 # focal plane temperature [kelvin]
		
	else: 
	
		temperature_focal_plane = temperature_focal_plane
		
	if tes_normal_resistance == None:
	
		tes_normal_resistance = 1. # TES noraml resistance [ohm]
		
	else:
	
		tes_normal_resistance = tes_normal_resistance
	
	if tes_log_sensitivity_alpha == None:
	
		tes_log_sensitivity_alpha = 100. # TES alpha = dlogR/dlogT
		
	else:
	
		tes_log_sensitivity_alpha = tes_log_sensitivity_alpha
		
	if tes_leg_thermal_carrier_exponent == None:
	
		tes_leg_thermal_carrier_exponent = 4. # phonons (for electrons change to 2)
		
	else:
	
		tes_leg_thermal_carrier_exponent = tes_leg_thermal_carrier_exponent
		
	if tes_normal_time_constant == None:
	
		tes_normal_time_constant = 33.e-3 # thermal time-constant in normal state [seconds]
		
	else:
	
		tes_normal_time_constant = tes_normal_time_constant
		
	if optical_loading_power == None:
	
		optical_loading_power = 0.5e-12 # optical loading power [watts]
		
	else:
	
		optical_loading_power = optical_loading_power
		
	if tes_saturation_power == None:
	
		tes_saturation_power = 2.5*0.5e-12 # tes saturation power [watts]
		
	else:
	
		tes_saturation_power = tes_saturation_power
	
	if tes_transition_temperature == None:
	
		tes_transition_temperature = 1.71*0.1 # tes transition temperature [kelvin]
		
	else:
	
		tes_transition_temperature = tes_transition_temperature
		
	if tes_leg_thermal_conductivity == None:
	
		tes_leg_thermal_conductivity = 2.5*0.5e-12/((1.71*0.1)**4.-0.1**4.)*4.*(1.71*0.1)**(4.-1.) # tes thermal conductivity [watts/kelvin]
		
	else:
	
		tes_leg_thermal_conductivity = tes_leg_thermal_conductivity
		
	if tes_heat_capacity == None:
	
		tes_heat_capacity = 33.e-3*2.5*0.5e-12/((1.71*0.1)**4.-0.1**4.)*4.*(1.71*0.1)**(4.-1.) # tes heat capacity [joule/kelvin]
		
	else: 
	
		tes_heat_capacity = tes_heat_capacity
	
	if biasing_current == None:
	
		biasing_current = 33.e-6 # bias current [ampere]
		
	else:
	
		biasing_current = biasing_current
		
	return tesdc(squid_input_inductor, shunt_resistor, temperature_focal_plane, tes_normal_resistance, tes_log_sensitivity_alpha, tes_leg_thermal_carrier_exponent, tes_normal_time_constant, optical_loading_power, tes_saturation_power, tes_transition_temperature, tes_leg_thermal_conductivity, tes_heat_capacity, biasing_current)

# Define TES AC model	
def tes_ac_model(squid_input_inductor = None, shunt_resistor = None, temperature_focal_plane = None, tes_normal_resistance = None, tes_log_sensitivity_alpha = None, tes_leg_thermal_carrier_exponent = None, tes_normal_time_constant = None, optical_loading_power = None, tes_saturation_power = None, tes_transition_temperature = None, tes_leg_thermal_conductivity = None, tes_heat_capacity = None, biasing_current_amplitude = None, ac_frequency = None, mux_frequency = None, mux_lc_inductor = None, mux_lc_capacitor = None):

	tesac = namedtuple('tes_ac_model', ['squid_input_inductor', 'shunt_resistor', 'temperature_focal_plane', 'tes_normal_resistance', 'tes_log_sensitivity_alpha', 'tes_leg_thermal_carrier_exponent', 'tes_normal_time_constant', 'optical_loading_power', 'tes_saturation_power', 'tes_transition_temperature', 'tes_leg_thermal_conductivity', 'tes_heat_capacity', 'biasing_current_amplitude', 'ac_frequency', 'mux_frequency', 'mux_lc_inductor', 'mux_lc_capacitor'])

	if squid_input_inductor == None:
	 
		squid_input_inductor = 10.e-9 # SQUID input inductor [henry]
		
	else: 
	
		squid_input_inductor = squid_input_inductor
		
	if shunt_resistor == None:

		shunt_resistor = 0.02 # shunt resistor [ohm]
		
	else:
	
		shunt_resistor = shunt_resistor
	
	if temperature_focal_plane == None:
	
		temperature_focal_plane = 0.1 # focal plane temperature [kelvin]
		
	else: 
	
		temperature_focal_plane = temperature_focal_plane
		
	if tes_normal_resistance == None:
	
		tes_normal_resistance = 1. # TES noraml resistance [ohm]
		
	else:
	
		tes_normal_resistance = tes_normal_resistance
	
	if tes_log_sensitivity_alpha == None:
	
		tes_log_sensitivity_alpha = 100. # TES alpha = dlogR/dlogT
		
	else:
	
		tes_log_sensitivity_alpha = tes_log_sensitivity_alpha
		
	if tes_leg_thermal_carrier_exponent == None:
	
		tes_leg_thermal_carrier_exponent = 4. # phonons (for electrons change to 2)
		
	else:
	
		tes_leg_thermal_carrier_exponent = tes_leg_thermal_carrier_exponent
		
	if tes_normal_time_constant == None:
	
		tes_normal_time_constant = 33.e-3 # thermal time-constant in normal state [seconds]
		
	else:
	
		tes_normal_time_constant = tes_normal_time_constant
		
	if optical_loading_power == None:
	
		optical_loading_power = 0.5e-12 # optical loading power [watts]
		
	else:
	
		optical_loading_power = optical_loading_power
		
	if tes_saturation_power == None:
	
		tes_saturation_power = 2.5*0.5e-12 # tes saturation power [watts]
		
	else:
	
		tes_saturation_power = tes_saturation_power
	
	if tes_transition_temperature == None:
	
		tes_transition_temperature = 1.71*0.1 # tes transition temperature [kelvin]
		
	else:
	
		tes_transition_temperature = tes_transition_temperature
		
	if tes_leg_thermal_conductivity == None:
	
		tes_leg_thermal_conductivity = 2.5*0.5e-12/((1.71*0.1)**4.-0.1**4.)*4.*(1.71*0.1)**(4.-1.) # tes thermal conductivity [watts/kelvin]
		
	else:
	
		tes_leg_thermal_conductivity = tes_leg_thermal_conductivity
		
	if tes_heat_capacity == None:
	
		tes_heat_capacity = 33.e-3*2.5*0.5e-12/((1.71*0.1)**4.-0.1**4.)*4.*(1.71*0.1)**(4.-1.) # tes heat capacity [joule/kelvin]
		
	else: 
	
		tes_heat_capacity = tes_heat_capacity
	
	if biasing_current_amplitude == None:
	
		biasing_current_amplitude = 33.e-6 * np.sqrt(2.0) # bias current [ampere]
		
	else:
	
		biasing_current_amplitude = biasing_current_amplitude
		
	if ac_frequency == None:
	
		ac_frequency = 1.e6 # bias current [ampere]
		
	else:
	
		ac_frequency = ac_frequency
		
	if mux_frequency == None:
	
		mux_frequency = 1.e6 # bias current [ampere]
		
	else:
	
		mux_frequency = mux_frequency
	
	if mux_lc_inductor == None:
	
		mux_lc_inductor = 65.e-6 # bias current [ampere]
		
	else:
	
		mux_lc_inductor = mux_lc_inductor
	
	if mux_lc_capacitor == None:
	
		mux_lc_capacitor =  1. / (65.e-6 * (2. * PI * 1.e6) * (2. * PI * 1.e6)) # bias current [ampere]
		
	else:
	
		mux_lc_capacitor = mux_lc_capacitor
		
	return tesac(squid_input_inductor, shunt_resistor, temperature_focal_plane, tes_normal_resistance, tes_log_sensitivity_alpha, tes_leg_thermal_carrier_exponent, tes_normal_time_constant, optical_loading_power, tes_saturation_power, tes_transition_temperature, tes_leg_thermal_conductivity, tes_heat_capacity, biasing_current_amplitude, ac_frequency, mux_frequency, mux_lc_inductor, mux_lc_capacitor)

# TES resistance vs. temperature dc
@jit(nopython=True)
def resistance_vs_temperature_dc(temperature, tes = tes_dc_model()):

	# Read content of the TES model
	squid_input_inductor = tes.squid_input_inductor
	shunt_resistor = tes.shunt_resistor 
	temperature_focal_plane = tes.temperature_focal_plane
	tes_normal_resistance = tes.tes_normal_resistance
	tes_log_sensitivity_alpha = tes.tes_log_sensitivity_alpha 
	tes_leg_thermal_carrier_exponent = tes.tes_leg_thermal_carrier_exponent 
	tes_normal_time_constant = tes.tes_normal_time_constant 
	optical_loading_power = tes.optical_loading_power 
	tes_saturation_power = tes.tes_saturation_power 
	tes_transition_temperature = tes.tes_transition_temperature 
	tes_leg_thermal_conductivity = tes.tes_leg_thermal_conductivity 
	tes_heat_capacity = tes.tes_heat_capacity 
	biasing_current = tes.biasing_current

	# Define R vs. T steepness from alpha value using arctan model
	steepness_rt = tes_log_sensitivity_alpha * PI * (tes_transition_temperature * tes_transition_temperature + 1.) * tes_normal_resistance / 2. / tes_transition_temperature
	
	return tes_normal_resistance * (np.arctan((temperature - tes_transition_temperature) * steepness_rt) + PI / 2.) / PI
	
# TES resistance vs. temperature ac
@jit(nopython=True)
def resistance_vs_temperature_ac(temperature, tes = tes_ac_model()):

	# Read content of the TES model
	squid_input_inductor = tes.squid_input_inductor
	shunt_resistor = tes.shunt_resistor 
	temperature_focal_plane = tes.temperature_focal_plane
	tes_normal_resistance = tes.tes_normal_resistance
	tes_log_sensitivity_alpha = tes.tes_log_sensitivity_alpha 
	tes_leg_thermal_carrier_exponent = tes.tes_leg_thermal_carrier_exponent 
	tes_normal_time_constant = tes.tes_normal_time_constant 
	optical_loading_power = tes.optical_loading_power 
	tes_saturation_power = tes.tes_saturation_power 
	tes_transition_temperature = tes.tes_transition_temperature 
	tes_leg_thermal_conductivity = tes.tes_leg_thermal_conductivity 
	tes_heat_capacity = tes.tes_heat_capacity 
	biasing_current_amplitude = tes.biasing_current_amplitude
	ac_frequency = tes.ac_frequency
	mux_frequency = tes.mux_frequency
	mux_lc_inductor = tes.mux_lc_inductor
	mux_lc_capacitor = tes.mux_lc_capacitor

	# Define R vs. T steepness from alpha value using arctan model
	steepness_rt = tes_log_sensitivity_alpha * PI * (tes_transition_temperature * tes_transition_temperature + 1.) * tes_normal_resistance / 2. / tes_transition_temperature
	
	return tes_normal_resistance * (np.arctan((temperature - tes_transition_temperature) * steepness_rt) + PI / 2.) / PI

# Coupled differential equations dc
@jit(nopython=True)
def differential_equations_dc(tes_current, tes_temperature, bias_current, loading_power, bath_temperature, tes = tes_dc_model()):

	# Define resistance vs. temperature relation
	R = resistance_vs_temperature_dc(tes_temperature, tes)
	
	# Read content of the TES model
	squid_input_inductor = tes.squid_input_inductor
	shunt_resistor = tes.shunt_resistor 
	temperature_focal_plane = tes.temperature_focal_plane
	tes_normal_resistance = tes.tes_normal_resistance
	tes_log_sensitivity_alpha = tes.tes_log_sensitivity_alpha 
	tes_leg_thermal_carrier_exponent = tes.tes_leg_thermal_carrier_exponent 
	tes_normal_time_constant = tes.tes_normal_time_constant 
	optical_loading_power = tes.optical_loading_power 
	tes_saturation_power = tes.tes_saturation_power 
	tes_transition_temperature = tes.tes_transition_temperature 
	tes_leg_thermal_conductivity = tes.tes_leg_thermal_conductivity 
	tes_heat_capacity = tes.tes_heat_capacity 
	biasing_current = tes.biasing_current

	V = bias_current * R * shunt_resistor / (R + shunt_resistor)
	Pb = tes_leg_thermal_conductivity * (tes_temperature**tes_leg_thermal_carrier_exponent - bath_temperature**tes_leg_thermal_carrier_exponent) / (tes_leg_thermal_carrier_exponent * tes_temperature**(tes_leg_thermal_carrier_exponent - 1.))
	Pr = V * V / R

	return [(V - tes_current * shunt_resistor - tes_current * R) / squid_input_inductor, (-Pb + Pr + loading_power) / tes_heat_capacity]
	
# Coupled differential equations ac
@jit(nopython=True)
def differential_equations_ac(tes_current, tes_dcurrent_dt, tes_temperature, time, bias_current_amplitude, loading_power, bath_temperature, tes = tes_ac_model()):

	# Define resistance vs. temperature relation
	R = resistance_vs_temperature_ac(tes_temperature, tes)
	
	# Read content of the TES model
	squid_input_inductor = tes.squid_input_inductor
	shunt_resistor = tes.shunt_resistor 
	temperature_focal_plane = tes.temperature_focal_plane
	tes_normal_resistance = tes.tes_normal_resistance
	tes_log_sensitivity_alpha = tes.tes_log_sensitivity_alpha 
	tes_leg_thermal_carrier_exponent = tes.tes_leg_thermal_carrier_exponent 
	tes_normal_time_constant = tes.tes_normal_time_constant 
	optical_loading_power = tes.optical_loading_power 
	tes_saturation_power = tes.tes_saturation_power 
	tes_transition_temperature = tes.tes_transition_temperature 
	tes_leg_thermal_conductivity = tes.tes_leg_thermal_conductivity 
	tes_heat_capacity = tes.tes_heat_capacity 
	biasing_current_amplitude = tes.biasing_current_amplitude
	ac_frequency = tes.ac_frequency
	mux_frequency = tes.mux_frequency
	mux_lc_inductor = tes.mux_lc_inductor
	mux_lc_capacitor = tes.mux_lc_capacitor
	
	dVdt = bias_current_amplitude * R * shunt_resistor * 2. * PI * ac_frequency * np.cos(2. * PI * ac_frequency * time) / (R + shunt_resistor)
	Pb = tes_leg_thermal_conductivity * (tes_temperature**tes_leg_thermal_carrier_exponent - bath_temperature**tes_leg_thermal_carrier_exponent) / (tes_leg_thermal_carrier_exponent * tes_temperature**(tes_leg_thermal_carrier_exponent - 1.))
	Pr = tes_current * tes_current * R

	return [tes_dcurrent_dt, (dVdt - tes_dcurrent_dt * R - tes_current / mux_lc_capacitor) / (mux_lc_inductor + squid_input_inductor), (-Pb + Pr + loading_power) / tes_heat_capacity]

# Solve & update Dc TES I & T with Runge-Kutta method
@jit(nopython=True)
def TesDcRungeKuttaSolver(time_array, bias_current_array, loading_power_array, bath_temperature_array, tes_current, tes_temperature, tes = tes_dc_model()):

	# Read content of the TES model
	squid_input_inductor = tes.squid_input_inductor
	shunt_resistor = tes.shunt_resistor 
	temperature_focal_plane = tes.temperature_focal_plane
	tes_normal_resistance = tes.tes_normal_resistance
	tes_log_sensitivity_alpha = tes.tes_log_sensitivity_alpha 
	tes_leg_thermal_carrier_exponent = tes.tes_leg_thermal_carrier_exponent 
	tes_normal_time_constant = tes.tes_normal_time_constant 
	optical_loading_power = tes.optical_loading_power 
	tes_saturation_power = tes.tes_saturation_power 
	tes_transition_temperature = tes.tes_transition_temperature 
	tes_leg_thermal_conductivity = tes.tes_leg_thermal_conductivity 
	tes_heat_capacity = tes.tes_heat_capacity 
	biasing_current = tes.biasing_current
	
	# Define time length and dt
	length = len(time_array)
	dt = abs(time_array[1]-time_array[0])

	# Initial conditions
	T0 = tes_transition_temperature
	I0 = biasing_current * shunt_resistor / (resistance_vs_temperature_dc(T0, tes) + shunt_resistor)

	for i in range(int(length)):
	
		kI1_tmp, kT1_tmp = differential_equations_dc(I0, T0, bias_current_array[i], loading_power_array[i], bath_temperature_array[i], tes)
		kI1 = dt*kI1_tmp
		kT1 = dt*kT1_tmp
	
		I0_tmp = I0+0.5*kI1
		T0_tmp = T0+0.5*kT1
		kI2_tmp, kT2_tmp = differential_equations_dc(I0_tmp, T0_tmp, bias_current_array[i], loading_power_array[i], bath_temperature_array[i], tes)
		kI2 = dt*kI2_tmp
		kT2 = dt*kT2_tmp
		
		I0_tmp = I0+0.5*kI2
		T0_tmp = T0+0.5*kT2
		kI3_tmp, kT3_tmp = differential_equations_dc(I0_tmp, T0_tmp, bias_current_array[i], loading_power_array[i], bath_temperature_array[i], tes)
		kI3 = dt*kI3_tmp
		kT3 = dt*kT3_tmp
	
		I0_tmp = I0+kI3
		T0_tmp = T0+kT3
		kI4_tmp, kT4_tmp = differential_equations_dc(I0_tmp, T0_tmp, bias_current_array[i], loading_power_array[i], bath_temperature_array[i], tes)
		kI4 = dt*kI4_tmp
		kT4 = dt*kT4_tmp

		I0 = I0+(kI1+2*kI2+2*kI3+kI4)/6.
		T0 = T0+(kT1+2*kT2+2*kT3+kT4)/6.
		tes_current[i]=I0
		tes_temperature[i]=T0
	
	return tes_current, tes_temperature

# Solve & update Ac TES I & T with Runge-Kutta method
@jit(nopython=True)
def TesAcRungeKuttaSolver(time_array, bias_current_amplitude_array, loading_power_array, bath_temperature_array, tes_current, tes_temperature, tes = tes_ac_model()):

	# Read content of the TES model
	squid_input_inductor = tes.squid_input_inductor
	shunt_resistor = tes.shunt_resistor 
	temperature_focal_plane = tes.temperature_focal_plane
	tes_normal_resistance = tes.tes_normal_resistance
	tes_log_sensitivity_alpha = tes.tes_log_sensitivity_alpha 
	tes_leg_thermal_carrier_exponent = tes.tes_leg_thermal_carrier_exponent 
	tes_normal_time_constant = tes.tes_normal_time_constant 
	optical_loading_power = tes.optical_loading_power 
	tes_saturation_power = tes.tes_saturation_power 
	tes_transition_temperature = tes.tes_transition_temperature 
	tes_leg_thermal_conductivity = tes.tes_leg_thermal_conductivity 
	tes_heat_capacity = tes.tes_heat_capacity 
	biasing_current_amplitude = tes.biasing_current_amplitude
	ac_frequency = tes.ac_frequency
	mux_frequency = tes.mux_frequency
	mux_lc_inductor = tes.mux_lc_inductor
	mux_lc_capacitor = tes.mux_lc_capacitor
	
	# Define time length and dt
	length = len(time_array)
	dt = abs(time_array[1]-time_array[0])

	# Initial conditions
	T0 = tes_transition_temperature 
	J0 = 0.
	I0 = biasing_current_amplitude * shunt_resistor / (resistance_vs_temperature_ac(T0, tes) + shunt_resistor)

	for i in range(int(length)):
		kI1_tmp, kJ1_tmp, kT1_tmp = differential_equations_ac(I0, J0, T0, time_array[i], bias_current_amplitude_array[i], loading_power_array[i], bath_temperature_array[i])
		kI1 = dt*kI1_tmp
		kJ1 = dt*kJ1_tmp
		kT1 = dt*kT1_tmp
	
		I0_tmp = I0+0.5*kI1
		J0_tmp = J0+0.5*kJ1
		T0_tmp = T0+0.5*kT1
		kI2_tmp, kJ2_tmp, kT2_tmp = differential_equations_ac(I0_tmp, J0_tmp, T0_tmp, time_array[i], bias_current_amplitude_array[i], loading_power_array[i], bath_temperature_array[i])
		kI2 = dt*kI2_tmp
		kJ2 = dt*kJ2_tmp
		kT2 = dt*kT2_tmp
		
		I0_tmp = I0+0.5*kI2
		J0_tmp = J0+0.5*kJ2
		T0_tmp = T0+0.5*kT2
		kI3_tmp, kJ3_tmp, kT3_tmp = differential_equations_ac(I0_tmp, J0_tmp, T0_tmp, time_array[i], bias_current_amplitude_array[i], loading_power_array[i], bath_temperature_array[i])
		kI3 = dt*kI3_tmp
		kJ3 = dt*kJ3_tmp
		kT3 = dt*kT3_tmp
	
		I0_tmp = I0+kI3
		J0_tmp = J0+kJ3
		T0_tmp = T0+kT3
		kI4_tmp, kJ4_tmp, kT4_tmp = differential_equations_ac(I0_tmp, J0_tmp, T0_tmp, time_array[i], bias_current_amplitude_array[i], loading_power_array[i], bath_temperature_array[i])
		kI4 = dt*kI4_tmp
		kJ4 = dt*kJ4_tmp
		kT4 = dt*kT4_tmp

		I0 = I0+(kI1+2*kI2+2*kI3+kI4)/6.
		J0 = J0+(kJ1+2*kJ2+2*kJ3+kJ4)/6.
		T0 = T0+(kT1+2*kT2+2*kT3+kT4)/6.
		tes_current[i]=I0
		tes_temperature[i]=T0
	
	return tes_current, tes_temperature
