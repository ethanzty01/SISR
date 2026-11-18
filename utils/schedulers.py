import math


class WarmupCosineScheduler:
    """Linear warmup to optimizer LRs, then cosine decay; bounds may be per-group."""
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, eta_min=0.0, warmup_start_lr=1e-7):
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.start_lrs = self._group_values(warmup_start_lr)
        self.min_lrs = self._group_values(eta_min)

        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.warmup_epochs >= self.total_epochs:
            raise ValueError("warmup_epochs must be smaller than total_epochs")
        for start, peak, minimum in zip(self.start_lrs, self.base_lrs, self.min_lrs):
            if not all(math.isfinite(v) for v in (start, peak, minimum)):
                raise ValueError('learning rates must be finite')
            if not (0 <= start <= peak and 0 <= minimum <= peak):
                raise ValueError('initial and minimum learning rates must be between zero and peak LR')

        self._apply_lr()

    def _group_values(self, value):
        values = list(value) if isinstance(value, (list, tuple)) else [value] * len(self.base_lrs)
        if len(values) != len(self.base_lrs):
            raise ValueError('learning rate bounds must match optimizer parameter groups')
        return [float(v) for v in values]

    def step(self):
        self.current_epoch += 1
        self._apply_lr()

    def _apply_lr(self):
        if self.warmup_epochs > 0 and self.current_epoch <= self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.start_lrs[i] + (self.base_lrs[i] - self.start_lrs[i]) * progress
            return

        cosine_epochs = self.total_epochs - self.warmup_epochs
        cosine_step = self.current_epoch - self.warmup_epochs
        cosine_ratio = min(max(cosine_step / cosine_epochs, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_ratio))
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.min_lrs[i] + (self.base_lrs[i] - self.min_lrs[i]) * cosine_factor

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        state = {
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'eta_min': self.eta_min,
            'warmup_start_lr': self.warmup_start_lr,
            'base_lrs': self.base_lrs,
        }
        return state

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.total_epochs = state_dict['total_epochs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.eta_min = state_dict['eta_min']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.base_lrs = state_dict['base_lrs']
        self.start_lrs = self._group_values(self.warmup_start_lr)
        self.min_lrs = self._group_values(self.eta_min)
        self._apply_lr()
