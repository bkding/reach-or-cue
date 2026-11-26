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

# Pseudo-online event rule.
TARGET_FP  = 1.0 # per min
REFRACTORY = 1.0 # s

# Hit windows relative to hand start: (before, after).
WINDOWS     = {'B': (1.0, 0.15), 'A': (0.5, 0.1)} # s
MAIN_WINDOW = 'B'
STOP_GUARD  = 0.5                                 # s, mask after hand stop

# Slope test.
SEARCH       = 2.0  # s, first trigger after LED
PRE_CUE      = 1.0  # s, trigger before LED counts as pre-cue
MAX_CENSORED = 50.0 # %, slope reported only at or below this

# Simulated detector streams.
SIM_DETECT  = 0.8  # kept fraction of true events
SIM_JITTER  = 0.05 # s
SIM_CUE_LAG = 0.25 # s, cue-locked delay
SIM_MOV_LAG = 0.1  # s, lead of movement-locked events
