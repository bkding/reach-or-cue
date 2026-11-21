# Decision grid and labels.
STEP      = 0.05 # s
PRE_ONSET = 0.5  # s, positive window before hand start
MIN_RT    = 0.1  # s, shorter reaction times are false starts

# Series types.
WEIGHT_BLOCK = 1 # BlockType of weight series

# Feature sets.
FEATURE_SETS = ('EMG', 'EEG', 'EEG+EMG')

# Numerical floor.
EPS = 1e-12
