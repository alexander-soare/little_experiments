In some of my experiments I compute accuracy/success scores. Experiments have a binary outcome: success or failure, and the score is the ratio `num_success / num_experiments`.

In an attempt to put error bars on this metric I assume a uniform prior over success probability and compute the beta posterior, then calculate the mean and lower/upper confidence bounds (with a 68.2% confidence interval centered on the mean).

_note: maybe that was a bit cryptic for some readers. If anyone asks, I can update this document with further elaboration_

Here's a Python snippet for calculating this:

```python
import scipy.stats

num_episodes = 500
success_rate = 0.548

alpha = num_episodes * success_rate + 1
beta = num_episodes * (1 - success_rate) + 1

confidence_interval = 0.682
lower_percentile = (1 - confidence_interval) / 2
upper_percentile = 1 - lower_percentile
print(lower_percentile, upper_percentile)
mean = scipy.stats.beta.mean(alpha, beta)
lower = scipy.stats.beta.ppf(lower_percentile, alpha, beta)
upper = scipy.stats.beta.ppf(upper_percentile, alpha, beta)

print(f"{lower * 100:.1f} / {mean * 100:.1f} / {upper * 100:.1f}")
```
