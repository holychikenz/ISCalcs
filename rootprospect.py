import numpy as np
import matplotlib.pyplot as plt

class GemGatherData:
  def __init__(self, **kwargs):
    self.actions = 1
    self.gems = np.array([1, 0, 0, 0, 0])
    self.enchant = kwargs.get("enchant", 5)
    self.bkg = None
    self._update_()
    
  def _update_(self):
    self.totalGems = sum(self.gems)
    self.apg = self.actions / self.totalGems
    
  def setActionsFromDeltaTime(self, tstart, tstop, timer):
    self.setActions( (tstop - tstart)/timer )
    
  def setActionsFromDeltaEssence(self, estart, estop, **kwargs):
    elvl = kwargs.get("elvl", 5)
    self.setActions( (estop - estart)/elvl )
    
  def setGems(self, gems):
    self.gems = np.array(gems)
    self._update_()
    
  def setActions(self, actions):
    self.actions = actions
    self._update_()
    
  def __add__(self, other):
    combo = GemGatherData()
    combo.actions = self.actions + other.actions
    combo.gems = self.gems + other.gems
    if self.enchant != other.enchant:
      raise ValueError("Cannot add data with different enchant levels")
    combo.enchant = self.enchant
    combo._update_()
    return combo
  
  def getActionProbability(self, **kwargs):
    verbose = kwargs.get("verbose", False)
    tg = self.totalGems
    uncert = tg**0.5
    if self.bkg is not None:
      bgcount = self.bkg.totalGems * self.actions / self.bkg.actions
      bguncert = (self.bkg.totalGems**0.5) * self.actions / self.bkg.actions
      tg = tg - bgcount
      if verbose:
        print(f'Signal Uncertainty: {uncert:0.2f}')
        print(f'Bkg Uncertainty: {bguncert:0.2f}')
      uncert = (uncert**2 + bguncert**2)**0.5
    pval = tg / self.actions / self.enchant
    pvhi = uncert/self.actions/self.enchant
    return pval, pvhi
  
  def report(self):
    pval, pvhi = self.getActionProbability()
    apg = 1/pval
    apghi = 1/(pval-pvhi) - apg
    apglow = apg - 1/(pval+pvhi)
    print(f'Gem Probability: {pval:0.5f} +- {pvhi:0.5f}')
    print(f'Action-enchant per gem: {apg:0.2f} + {apghi:0.2f} - {apglow:0.2f}')
    
  def setBackgroundDataset(self, bkg, **kwargs):
    self.bkg = bkg 
    verbose = kwargs.get("verbose", False)
    if verbose:
      p, h = self.getActionProbability(verbose=verbose)
      
    
def BackgroundSubtractedGGD(signal, background):
  newggd = GemGatherData(enchant=signal.enchant)
  actionFraction = signal.actions / background.actions
  newggd.setGems( signal.gems - background.gems*actionFraction )
  newggd.actions = signal.actions
  return newggd
    
actionTimer = lambda base, lvl, haste: base * 100 / (99 + lvl) * (1 - 0.04*haste)
    
def plotConfidence(ggd, **kwargs):
  f = lambda x, μ, σ: np.exp(-(x-μ)**2/2/σ**2)/(2*np.pi*σ**2)**0.5
  x = kwargs.pop("x", np.linspace(0, 0.5e-3, 1000))
  bw = x[1] - x[0]
  mu, sigma = ggd.getActionProbability()
  plt.plot(x, f(x, mu, sigma)*bw, **kwargs)
  plt.ylim(bottom=1e-5)
  return mu, sigma

def baseTimerHypothesis(ggdset, baseset, nameset):
  f = lambda x, μ, σ: np.exp(-(x-μ)**2/2/σ**2)/(2*np.pi*σ**2)**0.5
  fig = plt.figure(figsize=(10,8))
  trials = len(ggdset)
  baseMatrix = np.array([np.array(p.getActionProbability())*b for (p,b) in zip(ggdset, baseset)])
  delta = np.sum(baseMatrix, axis=0)[0] - trials*baseMatrix[0][0]
  uncert = np.sum(baseMatrix**2, axis=0)[1]**0.5
  tension = abs(delta / uncert)
  x = np.linspace(0, 1e-2, 1000)
  bw = x[1] - x[0]
  for (bm, name) in zip(baseMatrix, nameset):
    plt.plot(x, f(x, *bm)*bw, label=name)
  plt.ylim(bottom=1e-5)
  plt.title(f'Base timer hypothesis: {tension:0.2f}σ tension')
  plt.xlabel("Base time adjusted probability")
  plt.legend()
  
def gemChi2(ggd, **kwargs):
  plot = kwargs.get("plot", False)
  hypothesis = np.array([0.4, 0.3, 0.2, 0.07, 0.03])*ggd.totalGems
  chi2 = np.sum( (ggd.gems - hypothesis)**2/hypothesis )/(4)
  if plot:
    fig = plt.figure(figsize=(10, 8))
    x = np.array([1, 2, 3, 4, 5])
    axis = ["Sapphire", "Emerald", "Ruby", "Diamond", "Opal"]
    plt.errorbar(x, hypothesis, xerr=(x[2]-x[1])/2, linestyle="None", label="Expected")
    plt.errorbar(x, ggd.gems, yerr=ggd.gems**0.5, xerr=(x[2]-x[1])/2, linestyle="None", label="Data")
    plt.xticks(x, axis)
    plt.ylabel("Total Count")
    plt.ylim(bottom=0)
    plt.legend()
  return chi2
