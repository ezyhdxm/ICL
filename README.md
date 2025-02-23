# Instructions

To run the current ongoing experiments, see this [notebook](./TriggerMarkov.ipynb) for random triggers and this [notebook](./LatentModel.ipynb) for latent markov models.  

The structure of this repositary is roughly summarized in the following file tree:

ICL/
├── checkpoints/          # checkpoints for models
├── tasks/
│   ├── markov.py         # classes for different task samplers. 
│   ├── causal_graph.py   # legacy class for causal graph samplers.
│   ├── test_markov.py    # classes for the latent markov task sampler. 
│   └── old_sampler.py    # legacy samplers.
│ 
│   
├── models/
│   ├── ngram_learner.py  # empircal n-gram learners, used to provide baselines.
│   ├── attention.py      # implementation of multi-head attention. 
│   ├── base_models.py    # implementation of the base transformer model.
│   ├── pos_encoder.py    # implementation of different positional encoding classes.
│   └── sae.py            # sparse autoencoder, not in use currently.
├── figures/
│   ├── head_view.py      # Bertviz-type attention visualization.
│   ├── view_util.py      # Bertviz-type attention visualization utility file.
│   ├── head_view.js      # Bertviz-type attention visualization javascript.
│   ├── test_head_view.js # failed attempt to transpose Bertviz-type attention visualization.
│   └── plot.py           # create plots for training loss/ memory probes/ average attention pattern/ etc.
│ 
├── TriggerMarkov.ipynb   # Entry point to run random trigger experiments.
├── LatentModel.ipynb     # Entry point to run latent markov experiments.
├── train.py              # trains the model, generates training statistics and creates plots.
├── config.py             # configuration classes for sampler and tasks.
├── train_utils.py        # some utilities to get training loss and memory probe statistics. 
├── util.py               # implementation of different probes.
└── README.md



## Legacy

We are interested in understanding how transformers develop the in-context-learning ability through gradient descent. This repo reproduces implemetations of a few ICL experiments. Our implemetation is more efficient compared to some of the original implementations (see the [notebook](./Legacy/SpeedTest.ipynb) for more details). For demostrations of experiments in the previous literature, see the [notebook](./Legacy/LiterReview.ipynb). 

### 📖 Citations

```bibtex
@article{edelman2024evolution,
  title={The evolution of statistical induction heads: In-context learning markov chains},
  author={Edelman, Benjamin L and Edelman, Ezra and Goel, Surbhi and Malach, Eran and Tsilivis, Nikolaos},
  journal={arXiv preprint arXiv:2402.11004},
  year={2024}
}
```
```bibtex
@article{bietti2024birth,
  title={Birth of a transformer: A memory viewpoint},
  author={Bietti, Alberto and Cabannes, Vivien and Bouchacourt, Diane and Jegou, Herve and Bottou, Leon},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
```bibtex
@article{guo2024active,
  title={Active-dormant attention heads: Mechanistically demystifying extreme-token phenomena in llms},
  author={Guo, Tianyu and Pai, Druv and Bai, Yu and Jiao, Jiantao and Jordan, Michael I and Mei, Song},
  journal={arXiv preprint arXiv:2410.13835},
  year={2024}
}
```

```bibtex
@article{nichani2024transformers,
  title={How transformers learn causal structure with gradient descent},
  author={Nichani, Eshaan and Damian, Alex and Lee, Jason D},
  journal={arXiv preprint arXiv:2402.14735},
  year={2024}
}
```

```bibtex
@article{rajaraman2024transformers,
  title={Transformers on markov data: Constant depth suffices},
  author={Rajaraman, Nived and Bondaschi, Marco and Ramchandran, Kannan and Gastpar, Michael and Makkuva, Ashok Vardhan},
  journal={arXiv preprint arXiv:2407.17686},
  year={2024}
}
```

