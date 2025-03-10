


class Simulation:
    """
    This object creates a simulation
    
    Attributes
    ---------
    l : float
        parameter l 
    m : float
        parameter m
    h : float
        parameter h

    """

    def __init__(self, L, M, H):
        
        self.l = float(L)
        self.m = float(M)
        self.h = float(H)

    def get_total(self):
        """
        Returns the l + m
        """
        return self.l + self.m

    