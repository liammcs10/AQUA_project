"""
Unit testing for the AQUA_general and batchAQUA classes

Designed to check different levels of complexity in both classes are working and consistent with one another.

Checks:
    - Check that you can initialise a single neuron with autapse halfway through a previous sim.
    - Check that batch and single neuron are consistent with each other
        - neuron with no autapse
        - neuron with 0 time delay
        - neuron with positive time delay
        - neuron with different time delay to the previous one
        - FS neuron
    
"""


import numpy as np
from AQUA_general import AQUA
from batchAQUA_general import batchAQUA, pad_list
import unittest



class TestAQUA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        RS_NONE = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
        'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}
        
        RS_0 = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
        'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0.03, 'f': 8.0, 'tau': 0.}

        RS_05 = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
        'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0.03, 'f': 8.0, 'tau': 0.5}

        RS_08 = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
        'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0.03, 'f': 8.0, 'tau': 0.8}

        FS = {'name': 'FS', 'C': 20, 'k': 1, 'v_r': -55, 'v_t': -40, 'v_peak': 25,
        'a': 0.2, 'b': -2, 'c': -45, 'd': 0, 'e': 0.0, 'f': 0.0, 'tau': 0.0}

        N_neurons = 5

        x_start = np.array([[-65, 0., 0.] for i in range(N_neurons)])
      
        t_start = np.array([0. for i in range(N_neurons)])

        T = 500         # time of sim in ms
        dt = 0.01       # timestep
        N_iter = int(T/dt)

        I_inj = 100 * np.ones((N_neurons, N_iter))

        ## SINGLE NEURONS

        # neuron with no autapse
        neuron_none = AQUA(RS_NONE)
        neuron_none.Initialise(x_start[0], t_start[0])
        cls.X_NONE, T_NONE, _ = neuron_none.update_RK2(dt, N_iter, I_inj[0, :])

        # RS neuron with tau = 0.
        neuron_0 = AQUA(RS_0)
        neuron_0.Initialise(x_start[0], t_start[0])
        cls.X_0, T_0, _ = neuron_0.update_RK2(dt, N_iter, I_inj[0, :])

        # RS neuron with tau = 0.5
        neuron_05 = AQUA(RS_05)
        neuron_05.Initialise(x_start[0], t_start[0])
        cls.X_05, T_05, _ = neuron_05.update_RK2(dt, N_iter, I_inj[0, :])

        # RS neuron with tau = 0.8
        neuron_08 = AQUA(RS_08)
        neuron_08.Initialise(x_start[0], t_start[0])
        cls.X_08, T_08, _ = neuron_08.update_RK2(dt, N_iter, I_inj[0, :])


        # FS neuron
        neuron_FS = AQUA(FS)
        neuron_FS.Initialise(x_start[0], t_start[0])
        cls.X_FS, T_FS, _ = neuron_FS.update_RK2(dt, N_iter, I_inj[0, :])

        ## Initialise halfway through
        cls.idx = 5000  # initialise ~50ms into the run
        x_half = np.array([cls.X_NONE[:, cls.idx], cls.X_0[:, cls.idx], cls.X_05[:, cls.idx], cls.X_08[:, cls.idx], cls.X_FS[:, cls.idx]])
        t_half = np.array([T_NONE[cls.idx], T_0[cls.idx], T_05[cls.idx], T_08[cls.idx], T_FS[cls.idx]])

        w_prev = [cls.X_NONE[2, cls.idx-int(RS_NONE["tau"]/dt):cls.idx].tolist(), 
                    cls.X_0[2, cls.idx-int(RS_0["tau"]/dt):cls.idx].tolist(), 
                    cls.X_05[2, cls.idx-int(RS_05["tau"]/dt):cls.idx].tolist(), 
                    cls.X_08[2, cls.idx-int(RS_08["tau"]/dt):cls.idx].tolist(), 
                    cls.X_FS[2, cls.idx-int(FS["tau"]/dt):cls.idx].tolist()]
        w_prev = pad_list(w_prev, pad_end = False)


        # reinitialise neuron_05 halfway through. Needs an autapse here.
        neuron_05.Initialise(x_half[2], t_half[2])
        cls.X_half, _, _ = neuron_05.update_RK2(dt, N_iter, I_inj[0, :], w_prev = w_prev[2, :])

        ## BATCH RUNS

        batchParams = [RS_NONE, RS_0, RS_05, RS_08, FS]
        batch = batchAQUA(batchParams)
        batch.Initialise(x_half, t_half)
        cls.X_b, _, _ = batch.update_batch(dt, N_iter, I_inj, w_prev)

        ## END OF SETUP
    

    def test_singleNeuron_halfway(self):
        # tests that the base AQUA_class successfully initialises mid-run.
        self.assertTrue((self.X_05[:, self.idx:self.idx+100] == self.X_half[:, :100]).all(), 
                        msg = "Base AQUA fails to initialise mid_run.")
    
    def test_none_v_batch(self):
        # Make sure base and batch AQUA are consistent when no autapse present
        self.assertTrue((self.X_NONE[:, self.idx:self.idx+100] == self.X_b[0, :, :100]).all(),
                        msg = "Base and batch AQUA are inconsistent when no autapse present")

    def test_00_v_batch(self):
        # Make sure base and batch AQUA are consistent when no autapse present
        self.assertTrue((self.X_0[:, self.idx:self.idx+100] == self.X_b[1, :, :100]).all(),
                        msg = "Base and batch AQUA are inconsistent when autapse has no delay")

    def test_05_v_batch(self):
        # Make sure base and batch AQUA are consistent when no autapse present
        self.assertTrue((self.X_05[:, self.idx:self.idx+100] == self.X_b[2, :, :100]).all(),
                        msg = "Base and batch AQUA are inconsistent when autapse has delay 0.5")
    
    def test_08_v_batch(self):
        # Make sure base and batch AQUA are consistent when no autapse present
        self.assertTrue((self.X_08[:, self.idx:self.idx+100] == self.X_b[3, :, :100]).all(),
                        msg = "Base and batch AQUA are inconsistent when autapse has delay 0.8")
    
    def test_FS_v_batch(self):
        # Make sure base and batch AQUA are consistent when no autapse present
        self.assertTrue((self.X_FS[:, self.idx:self.idx+100] == self.X_b[4, :, :100]).all(),
                        msg = "Base and batch AQUA are inconsistent when autapse has delay 0.8")



if __name__ == "__main__":
    unittest.main()

