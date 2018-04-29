#from mingus.midi import fluidsynth
import numpy as np
import matplotlib.pyplot as plt
from note_generator import converter





array = np.load("autocrop.npy")
for row in array:
  if(row[1] != "solanahtari"):
    print(row)