'''
Jeswin Mathew
CSE-3313-001
03-12-2020
'''

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import freqz

#music file, Fs
song_sample, sampling_freq = sf.read('1700325127_1700323947794_asdh.wav')


#L
filter_len = 101

#Fc origigal 7500, kita coba 1000, 2000, dsb
low_cutoff_freq = 7500





zero_index = [0]

#Ft
normalized_transition_freq_low = low_cutoff_freq/sampling_freq


#setting the initial weights filled with zeros in the array
filter_coefficients = filter_len * zero_index

#M
filter_order = filter_len - 1

two_pi = 2*np.pi

i=0
w_num=[]
w_den = []

#taking half of the M value
half_filter = filter_order/2
two_pi = 2*np.pi

#low-pass filter algorithm
two_norm_freq_low = 2*normalized_transition_freq_low
for index in range(filter_len):
	if(index != half_filter):
		#dividing the numerator and denominator for the equation
		w_num = np.sin(two_pi*normalized_transition_freq_low*(index-half_filter))
		w_den = np.pi*(index-half_filter)
		filter_coefficients[index] = w_num/w_den
	else:
		filter_coefficients[index] = two_norm_freq_low

#variable for hamming window values
hamming_win_weights = zero_index * filter_len

eq1 = 0.54
coeff = 0.46

for i in range(filter_len):
	h_num = two_pi*i
	h_cos = np.cos(h_num/filter_order)
	hamming_win_weights[i] = (eq1 - coeff*(h_cos))


hamming_window1 = []

windowed_output = zero_index * filter_len
for i in range(filter_len):
	windowed_output[i] = filter_coefficients[i] * hamming_win_weights[i]


x,y = freqz(filter_coefficients,1)
plt.plot(x,abs(y))

a,b = freqz(windowed_output,1)
plt.plot(a,abs(b))

hamming_window2 = []

cleanFile = np.convolve(windowed_output,song_sample)
sf.write('cleanMusic.wav',cleanFile,sampling_freq)

plt.title('Frequency Response')
plt.legend(['original','windowed'])
plt.show()  