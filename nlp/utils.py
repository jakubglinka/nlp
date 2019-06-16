import tqdm

class tqdm_wget_pbar:
  """ Progress bar for wget. """

  def __enter__(self):
    self.elapsed = None
    self.tqdm = None
    return self

  def update(self, elapsed, total, done):
    if self.tqdm is None:
      self.tqdm = tqdm.tqdm(total=round(total / 1e6), unit="MB", unit_scale=True)
      self.tqdm.update(round(elapsed / 1e6, 2))
      self.elapsed = elapsed
    else:
      self.tqdm.update(round((elapsed - self.elapsed) / 1e6, 5))
      self.elapsed = elapsed

  def __exit__(self, b, c, d):
    self.tqdm.close()

