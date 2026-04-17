from random import random

class DemandNode():
    def __init__(self, mean, sd):
        self.parents = []
        self.mean = mean
        self.sd = sd

    def random_demand(self):
        return random.normalvariate(self.mean, self.sd)
    
    def get_parents(self):
        return self.parents
    
    def add_parent(self, parent):
        self.parents.append(parent)