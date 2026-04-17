from random import random

class SupplyNode():
    def __init__(self, p_down, capacity):
        self.children = []
        self.is_active = True
        self.p_down = p_down
        self.capacity = capacity

    def add_child(self, child):
        self.children.append(child)

    def satisfy_demand(self, demand):
        demand_satisfied = demand if (self.capacity - demand >= 0) else self.capacity
        self.capacity = max(0, self.capacity - demand)
        return demand_satisfied
    
    def update_active(self):
        self.is_active = random.random() > self.p_down

    def get_status(self):
        return self.is_active
    
    def get_children(self):
        return self.children