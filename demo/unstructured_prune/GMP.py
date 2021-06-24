class GMP:
    def __init__(
            self,
            pruner,
            stable_iterations,
            pruning_iterations,
            pruning_steps,
            ratio,
            initial_ratio, ):
        self.pruner = pruner
        self.stable_iterations = stable_iterations
        self.pruning_iterations = pruning_iterations
        self.pruning_steps = pruning_steps
        self.ratio = ratio
        self.initial_ratio = initial_ratio

        self._prepare_training_hyper_parameters()

    def _prepare_training_hyper_parameters(self):
        self.ratios_stack = []
        self.ratio_increment_period = int(self.pruning_iterations /
                                          self.pruning_steps)
        for i in range(self.pruning_steps):
            ratio_tmp = ((i / self.pruning_steps) - 1.0)**3 + 1
            ratio_tmp = ratio_tmp * (self.ratio - self.initial_ratio
                                     ) + self.initial_ratio
            self.ratios_stack.append(ratio_tmp)
        self.ratios_stack.reverse()

    def step(self, cur_iteration):
        ori_ratio = self.pruner.ratio
        if len(
                self.ratios_stack
        ) > 0 and cur_iteration >= self.stable_iterations and cur_iteration < self.stable_iterations + self.pruning_iterations:
            if cur_iteration % self.ratio_increment_period == 0:
                self.pruner.ratio = self.ratios_stack.pop()
        elif len(
                self.ratios_stack
        ) == 0 or cur_iteration >= self.stable_iterations + self.pruning_iterations:
            self.pruner.ratio = self.ratio

        if ori_ratio != self.pruner.ratio and cur_iteration >= self.stable_iterations:
            self.pruner.step()
