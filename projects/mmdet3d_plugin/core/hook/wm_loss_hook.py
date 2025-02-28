from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class EnableWmLossHookIter(Hook):

    def __init__(self,
                 enable_after_iter=10000,
                 wm_loss_weight=0.2,
                 ):
        self.enable_after_iter = enable_after_iter
        self.wm_loss_weight = wm_loss_weight

    def before_train_iter(self, runner):
        cur_iter = runner.iter # begin from 0
        if cur_iter == self.enable_after_iter:
            runner.logger.info(f'Enable wm loss from now.')
        if cur_iter >= self.enable_after_iter: # keep the sanity when resuming model
            runner.model.module.wm_loss_weight = self.wm_loss_weight