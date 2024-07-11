In this little experiment I use the [LeRobot library](https://huggingface.co/lerobot), with some tweaks, to really get a feel for the multi-modality (in the sense of probability distributions) of generative imitation learning policies like [Diffusion Policy (DP)](https://arxiv.org/abs/2303.04137) and [Action Chunking Transformers (ACT)](https://arxiv.org/abs/2304.13705).

**TLDR**: At least in the PushT environment. These models do not produce multi-modal action trajectory distributions! Furthermore, the distributions seem to be much sharper than one might expect. Adding small gaussian noise to the observations is a better means of producing multimodality, and can be used to the same effect with discriminative models like ACT trained without the variational objective.

### Quick background

I'm assuming you know about DP and ACT so I'll just very quickly jog your memory:

**Generative modeling** - DP and ACT both do generative modelling. This means they attempt to model an arbitrary distribution `p(action_trajectory | observation)`. 
- On the other hand, "discriminative modeling" tries to make a choice about which `action_trajectory` is best. At its core, it still models `p(action_trajectory | observation)`, but restricted to uni-modal distributions like Gaussians (with MSE loss during training) or Lorentzians (with L1 loss during training), or multinomial distributions in the case of categorical prediction tasks.  
- The supposed advantage of generative modeling over discriminative modeling is that we can represent multi-modal distributions in action trajectories, which naturally arise when an embodied agent can accomplish the same task in more than one way.

**ACT** - ACT uses a conditional variational auto-encoder during training and just the decoder during inference. It also has the option of doing discriminative modeling by not using the VAE during training, but instead training with just an L1 loss.

**Diffusion Policy** - Diffusion Policy uses a diffusion model.

**PushT** - PushT is a nice simulation environment for quick prototyping. The goal is to use a cylindrical end-effector to prod a T-shaped block around and try to get it onto a T-shaped target. I used it because it's 2D, and therefore easy to visualize.

### Visualizing multi-modality

The following GIF shows a trained diffusion policy being rolled out on a PushT environment. It also visualizes the action trajectory returned by the policy with a red->blue gradient to indicate the flow of the trajectory. Inference is rerun when the trajectory is depleted.

![](.images/multimodal_m1.gif)

This next GIF does the same thing, but ~100 action trajectories are generated and visualized at each inference step (where one of them is picked randomly for the actual rollout, and 99 are not used but just visualized). This is to get a monte-carlo view of the "distribution".

![](.images/multimodal_0.gif)

We do see some spread in the distribution but it mostly looks uni-modal and sharp (sharper than I would expect when thinking about all the paths I might take to accomplish the task).

To take things further, I wanted full control over where I'll run inference rather than being at the whim of the where the policy wants to go. So I rigged things up so that I could control the robot with mouse input. Then I ran inference on _every_ step. That way I could explore, and purposefully manufacture scenarios where multi-modality _should_ arise. In the following set of GIFs I do just that, with DP and ACT side-by-side.

*Note: The official DP implementation uses 2 steps worth of observation history. Here I just take the current step and copy it back to the previous step, effectively just using one step of history.*

*Note: The official ACT implementation uses a zero-vector for the latent at inference time to get deterministic rollouts. Here I do sampling from the standard normal distribution instead.*

Here's one where I just try to do the task of pushing the T onto the target. Here we do see some wider distributions produced by DP.

![](.images/multimodal_1.gif)

Next, I set up a scenario where we would expect bi-modal distributions. There are a few interesting observations here:
1. The DP distribution seems to be much smoother (or spread out) than the ACT distribution.
2. ACT seems to switch modes sooner than DP, indicating some form of relative bias between the two models.
3. Both distributions actually look uni-modal all the way, whereas I would have expected a transition from uni-modal, to bi-modal, back to uni-modal. Also related to this is that the trajectories that seem to hedge between the two modes look like they are bad trajectories!
4. For both models, there seems to be a bias to going around the T in a counter-clockwise direction over a clockwise direction. This might say something about the way the demonstration data was collected (I believe the demonstration data was made by human teleoperators).

![](.images/multimodal_2.gif)

I also observed that the policies can be quite stubborn about committing to a mode when there is clearly more than one way to solve the problem. In this GIF, again, we see a heavy bias towards counter-clockwise motion.

![](.images/multimodal_3.gif)

Finally, what might be interesting, is to add noise to the proprioceptive state and T keypoints (I didn't mentions so far, but these policies use 8 keypoints from the T-block rather than raw input images). In the GIF below we do see truly multi-modal looking distributions. But this would be true with a policy trained on a vanilla MSE objective (_it was just interesting to try anyway_).

![](.images/multimodal_7.gif)

Just to drive that last point home, here's an ACT policy trained without a VAE objective but with noised state vectors. It looks much like the GIF above. One interesting observation I'll make here is that perhaps it's sufficient to have noisy inputs to get the sort of multi-modality we are looking for.

![](.images/multimodal_8.gif)

_Maybe_ generative modelling is not actually necessary (it's also worth noting I've heard anecdotal evidence of this from other researchers in the field).

### What to make of all this

Generative modeling is supposed to overcome a problem that discriminative modeling would have in multi-modal scenarios: hedging between modes and ultimately following an out-of-distribution path. But:
1. We observed above that the generative models still output bad/hedging paths.
    - I'm not sure why this is the case. I'd really want to explore a lot more (with various environments, and dataset sizes / collection methods) before making a solid conclusion here.
2. We also observed that observation noise is enough to induce "multi-modal" outputs, even in the case of a discriminative model.


Overall, I'm leaning towards thinking that generative modeling is not necessary, at least for small scale experiments. This is especially the case in real world robots where the observations are bound to be noisy anyway. I think if anything could prove me wrong: it would be doing much larger scale experiments with complex environments and a lot of training data. Perhaps in such a regime, the model has more of a chance of learning nuanced multi-modal behavior. The geek in me certainly hopes that this would be the case 🤓.

### Bonus visualizations

One thing that I briefly noted above is that DP is actually set up to take the current observation, and one from the preceding timestep (two consecutive observations all up). I'd been actually just cloning the current observation ninto the previous timestep. Watch what happens if I actually use the previous observation. We see that the distribution doesn't just depend on the current position, but also on the velocity of the robot end-effector. Observe, as I make sudden movements that cause the distribution to switch modes momentarily before settling back to its default state (with a zero-velocity prior). Clearly, using an observation history makes a difference!

![](.images/multimodal_4.gif)

With this setup, I wanted to double check a point I made earlier: "trajectories that seem to hedge between the two modes look like they are bad trajectories". Here I set up some "bad" trajectories and try to follow them. It looks like I can't go far down the "bad" path before the policy switches to a more sensible plan. I tried this a few times and always found that eventually the policy would get back on track.

![](.images/multimodal_5.gif)

Finally, I also tried an experiment with [VQ-BeT](https://sjlee.cc/vq-bet/simulated_index.html). This is a GPT-style transformer that predicts action tokens. The "multi-modality" comes from the fact that it predicts a categorical distribution over possible tokens and we use multinomial sampling to select one. We can definitely see the multinomial nature of the multimodality in the GIF below! 

![](.images/vqbet_multimodal.gif)

What's very cool about VQ-BeT is that the action tokens are learned with a VQ-VAE, so we are looking at purely learned modes in the GIF above 🤓.

---

## Appendix: Reproducibility

All visualizations were run with [this branch of LeRobot](https://github.com/alexander-soare/lerobot/tree/experiment_multimodal_actions). You can run the setup as according the LeRobot's main README.

You can play around with the visualization tool by running `python lerobot/scripts/interactive_multimodality`. I've left a docstring up the top about how to use it. The pretrained policies I've used are all up in the HuggingFace hub and referenced in that script.