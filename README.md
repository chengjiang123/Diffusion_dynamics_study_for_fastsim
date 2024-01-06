# diffusion_dynamics_study_for_fastsim

Studies about efficient sampler, scheduler and training strategy for diffusion model on CaloChallenge and JetNet dataset.

The diffusion model has demonstrated promising results in image generation, recently becoming mainstream and representing a notable advancement for many generative modeling tasks. The prior applications of the diffusion model in both fast event and detector simulation in high energy physics have shown exceptional performance, providing a viable solution to generate sufficient statistics within a constrained computational budget in preparation for the High Luminosity LHC. However, many of these applications suffer from slow generation with large sampling steps. In this study, we focus on the latest benchmark developments in efficient ODE/SDE-based samplers, schedulers, and fast convergence training techniques. With the designs implemented on the existing architecture, the generated classes surpass the performance of previous models, achieving significant speedup via various evaluation metrics.



Using existing network, code modifed based on [CaloDiffu](https://github.com/OzAmram/CaloDiffusion) and [PC-Jedi](https://github.com/DebajyotiS/PC-JeDi/tree/EPiC-JeDi) repo.


