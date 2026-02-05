"""
Automate the checking of the AQUA models.

- Need to make sure that the batch and single neuron models are 
consistent with each other.

- Need to test different edge cases basically.



"""

import numpy as np
from AQUA_general import *
from batchAQUA_general import *
import unittest

class Test_AQUA(unittest.TestCase):

    def test_singleNeuron():
        # make sure that 