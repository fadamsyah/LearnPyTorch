# Superconvergence
# OneCycleLR with 3 phase for SGD and Adam Optimizer
class OptimOneCycleLR():
    def __init__(self, optimizer, init_lr, max_lr, min_lr, epochs,
                 steps_per_epoch, pct_rise=0.3, pct_fall=None,
                 pct_const=0., pct_stdy=0., base_momentum=0.85,
                 max_momentum=0.95):
        self.optimizer = optimizer
        self.total_steps = epochs * steps_per_epoch
        self.count_step = 0
        
        if pct_fall is None: pct_fall=pct_rise
        if pct_rise > 1. or pct_rise <= 0. or pct_fall > 1. or pct_fall <= 0. or pct_const > 1. or pct_const < 0. or pct_stdy > 1. or pct_stdy < 0.:
            raise TypeError("The pct_rise, pct_fall, pct_const, and pct_stdy must lie between 0 and 1.")
        if pct_rise + pct_const + pct_fall + pct_stdy >= 1.:
            raise TypeError("The pct_rise + pct_const + pct_fall + pct_stdy must be less than 1.")
        
        pct_end = 1. - pct_rise - pct_const - pct_fall - pct_stdy
        pct_rise_steps = float(pct_rise * self.total_steps)
        pct_const_steps = float(pct_const * self.total_steps)
        pct_fall_steps = float(pct_fall * self.total_steps)
        pct_stdy_steps = float(pct_stdy * self.total_steps)
        pct_end_steps = float(pct_end * self.total_steps)
        assert pct_rise_steps + pct_const_steps + pct_fall_steps + \
                pct_stdy_steps + pct_end_steps == self.total_steps
        self.schedule_phases = [{'end_step': pct_rise_steps,
                                 'slope_lr': (max_lr - init_lr) / pct_rise_steps,
                                 'slope_mtm': (base_momentum - max_momentum) / pct_rise_steps},
                                {'end_step': pct_rise_steps + pct_const_steps,
                                 'slope_lr': 0.,
                                 'slope_mtm': 0.},
                                {'end_step': pct_rise_steps + pct_const_steps + pct_fall_steps,
                                 'slope_lr': (init_lr - max_lr) / pct_fall_steps,
                                 'slope_mtm': (max_momentum - base_momentum) / pct_fall_steps},
                                {'end_step': pct_rise_steps + pct_const_steps + pct_fall_steps + pct_stdy_steps,
                                 'slope_lr': 0.,
                                 'slope_mtm': 0.},
                                {'end_step': pct_rise_steps + pct_const_steps + pct_fall_steps + pct_stdy_steps + pct_end_steps,
                                 'slope_lr': (min_lr - init_lr) / pct_end_steps,
                                 'slope_mtm': 0.}]
        
        self.optimizer.param_groups[0]['lr'] = init_lr
        if type(self.optimizer).__name__ == 'SGD':
            self.optimizer.param_groups[0]['momentum'] = max_momentum
        elif type(self.optimizer).__name__ in ['AdamW', 'Adam']:
            self.optimizer.param_groups[0]['betas'][0] = max_momentum
        else:
            raise Exception("Sorry, the optimizer is not recognized")  
    
    def step(self):
        for phase in range(len(self.schedule_phases)):
            if self.schedule_phases[phase]['end_step'] > self.count_step:
                break
                
        self.count_step += 1

        self.optimizer.param_groups[0]['lr'] += self.schedule_phases[phase]['slope_lr']
        if type(self.optimizer).__name__ == 'SGD':
            self.optimizer.param_groups[0]['momentum'] += self.schedule_phases[phase]['slope_mtm']
            return self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['momentum']
        elif type(self.optimizer).__name__ in ['AdamW', 'Adam']:
            self.optimizer.param_groups[0]['betas'][0] += self.schedule_phases[phase]['slope_mtm']
            return self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['betas'][0]
        else:
            raise Exception("Sorry, the optimizer is not recognized")   
        