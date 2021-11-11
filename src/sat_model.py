import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from src.utils.common import merge_list_dicts, merge_list_2d
from src.utils.decorators import safe_interruption
from src.utils.timer import CodeTimer
from src.transforms.transforms import Compose 

class ModelState():
  """
    State which DS_Model exposes to the outer world. 
  """
  def push(self, key, value):
    if key not in self.__dict__.keys(): self.__dict__[key] = []
    self.__dict__[key].append(value)

  def clear(self):
    for key in self.__dict__.keys():
      self.__dict__[key] = []

  def get(self):
      state = self.__dict__
      return state

def linear_combination(x, y, alpha):
  return alpha*x + (1-alpha)*y

class Sat_Model(nn.Module):
    def __init__(self):
      super(Sat_Model, self).__init__()
      # Model components - he needs them for propper working
      self._device = "cpu"
      self._train_loader = []
      self._valid_loader = []
      self.optimizer = None
      self.scheduler = None
      self.loss_function = None
      self.postprocess = Compose([])
      
      self.observers = {
        "after_epoch" : [], 
        "before_epoch": [],
        "after_step"  : [], 
        "before_step" : []
      }

      # State which model exposes to the environment = (observers, user)
      self.outer_state = ModelState()

      # Working state -> TRAIN, TEST
      self._use_amp = False
      self.mixup_factor = -1
      self.activate_mixup = False
      self._step_timer  = CodeTimer(False)
      self._epoch_timer = CodeTimer(False)

    ############ MIXUP ###############

    def use_mixup(self):
      return self.activate_mixup and self.mixup_factor >= 0 and self.mixup_factor <= 1

    ############ AMP ###############

    @property
    def use_amp(self) -> bool:
      return self._use_amp
    
    @use_amp.setter
    def use_amp(self, value: bool):
      if value:
        assert self.optimizer, "Optimizer should be defined"
        assert self.loss_function, "Loss function should be defined"
      self._use_amp = value
      self._scaler = GradScaler(enabled=self.use_amp)

    ############ TIMER ###############

    @property
    def step_timer(self) -> CodeTimer:
      return self._step_timer
    
    @step_timer.setter
    def step_timer(self, value: bool):
      self._step_timer = CodeTimer(value)

    @property
    def epoch_timer(self) -> CodeTimer:
      return self._epoch_timer
    
    @epoch_timer.setter
    def epoch_timer(self, value: bool):
      self._epoch_timer = CodeTimer(value)

    def activate_timing(self, activate: bool):
      self.step_timer  = activate
      self.epoch_timer = activate

    ############ WORKING STATE ###############

    def train_state(self):
      self.activate_mixup = True
      self.train()
      self.current_iterator = iter(self.train_loader)
      self.current_iter_size = len(self.train_loader)
      self.outer_state.iteration = 0
      self.outer_state.state = "TRAIN"
    
    def eval_state(self):
      self.activate_mixup = False
      self.eval()
      self.current_iterator  = iter(self.valid_loader)
      self.current_iter_size = len(self.valid_loader)
      self.outer_state.iteration = 0
      self.outer_state.state = "TEST"

    ############ DEVICE ###############

    @property
    def device(self):
      return self._device
    
    @device.setter
    def device(self, device):
      devices = ["cpu", "cuda"]
      if device not in devices:
          raise ValueError("device must be one of the following types:" + devices)
      self._device = device
      self = self.to(device)

    def to_device(self, *xs):
      return [x.to(self.device) for x in xs]

    ############ LOADERS ###############
    
    @property
    def valid_loader(self):
      return self._valid_loader

    @property
    def train_loader(self):
      return self._train_loader

    @valid_loader.setter
    def valid_loader(self, valid_loader):
      self._valid_loader = valid_loader

    @train_loader.setter
    def train_loader(self, train_loader):
      self._train_loader = train_loader
      dataset = train_loader.dataset
      self.outer_state.norm_mean, self.outer_state.norm_std = dataset.mean, dataset.std
    
    ################### HOOKS & CALLBACKS #############################
    
    def before_step_hook(self):
      pass

    def after_step_hook(self):
      pass

    def notify_observers(self, key=None):
      if key in self.observers.keys() and not self.observers[key]: return
      state = self.outer_state.get()
      metrics = self.observer_results()
      state["metrics"] = metrics
      state["model_state_dict"] = self.state_dict()
      observers = merge_list_2d(self.observers.values()) if not key else self.observers[key]
      for obs in observers:
        if self.outer_state.state == obs.when or not obs.when:
          obs.update(**state)

    def reset_observers(self, key=None):
      observers = merge_list_2d(self.observers.values()) if not key else self.observers[key]
      for obs in observers:
        if self.outer_state.state == obs.when or not obs.when:
          obs.reset_state()

    def observer_results(self):
      observers = merge_list_2d(self.observers.values())
      results = [obs.get() for obs in observers] 
      results = [m for m in results if m]
      results = merge_list_dicts(results)
      return results
      
    ################### MAIN #############################

    def forward_step(self):
      with autocast(enabled=self.use_amp):
        if self.use_mixup():
          input_batch1, target_batch1 = next(self.current_iterator)
          input_batch2, target_batch2 = next(self.current_iterator)
          input_batch  = linear_combination(input_batch1, input_batch2, self.mixup_factor)
          input_batch, target_batch1, target_batch2 = self.to_device(input_batch, target_batch1, target_batch2)
          logits_batch = self.forward(input_batch)
          loss1, loss2 = self.loss_function(logits_batch, target_batch1), self.loss_function(logits_batch, target_batch2)
          batch_loss = linear_combination(loss1, loss2, self.mixup_factor)
          ###
          self.outer_state.prediction = self.postprocess(logits_batch) if self.postprocess else logits_batch.argmax(1)
          self.outer_state.input = input_batch
          self.outer_state.target = linear_combination(target_batch1, target_batch2, self.mixup_factor)
          self.outer_state.logits = logits_batch
          self.outer_state.loss = batch_loss
        else:
          input_batch, target_batch = next(self.current_iterator)
          input_batch, target_batch = self.to_device(input_batch, target_batch)
          logits_batch = self.forward(input_batch)
          self.outer_state.prediction = self.postprocess(logits_batch) if self.postprocess.transforms else logits_batch.argmax(1)
          ###
          self.outer_state.loss = self.loss_function(logits_batch, target_batch)
          self.outer_state.input = input_batch
          self.outer_state.target = target_batch
          self.outer_state.logits = logits_batch

    def backward_step(self):
      scaled_batch_loss = self._scaler.scale(self.outer_state.loss) if self.use_amp else self.outer_state.loss
      scaled_batch_loss.backward()
      if self.use_amp:
        self._scaler.step(self.optimizer) 
        self._scaler.update()
      else:
        self.optimizer.step()
      self.zero_grad(set_to_none=True)

    def step(self):
      with self.step_timer as step_clock:
        self.forward_step()
        if self.outer_state.state == "TRAIN": self.backward_step()
        self.outer_state.iteration += 1
      if step_clock.active: self.outer_state.push("step_time", step_clock.time)
      
    def one_epoch(self):
      self.notify_observers("before_epoch")
      pbar = tqdm(total=self.current_iter_size, position=0, leave=True)
      with self.epoch_timer as epoch_clock:         
        try:
          while True:
            self.before_step_hook()
            self.notify_observers("before_step")
            self.step()
            self.notify_observers("after_step")
            self.after_step_hook()
            pbar.update()
        except StopIteration:
          pbar.close()
      if epoch_clock.active: self.outer_state.push("epoch_time", epoch_clock.time)
      self.notify_observers("after_epoch")

    def train_epoch(self):
      self.train_state()
      self.one_epoch()

    @torch.no_grad()
    def evaluate(self):
      self.eval_state()  
      self.one_epoch()
    
    def before_epoch_hook(self):
      pass

    def after_epoch_hook(self):
      pass

    @safe_interruption
    def fit(self, epochs):
      for self.outer_state.epoch in range(1, epochs+1):
        self.before_epoch_hook()
        self.train_epoch()
        self.reset_observers()
        if self.valid_loader: self.evaluate()
        if self.scheduler: self.scheduler.step()
        self.after_epoch_hook()
        self.reset_observers() 
