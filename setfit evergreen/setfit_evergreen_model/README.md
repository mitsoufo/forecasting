---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 'Watch: South Park roasts Trump over White House renovation '
- text: 'Free Halloween fun for families at The Littleton Arms '
- text: 'Super Furry Animals add three shows to UK tour '
- text: '15 spooky photos of Teesside children''s Halloween costumes in the 1970s
    to 1990s '
- text: 'Where to find GTA Online Peyote Plants and become an animal '
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
model-index:
- name: SetFit
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Unknown
      type: unknown
      split: test
    metrics:
    - type: accuracy
      value: 0.94
      name: Accuracy
---

# SetFit

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
<!-- - **Sentence Transformer:** [Unknown](https://huggingface.co/unknown) -->
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 128 tokens
- **Number of Classes:** 2 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                                                                                                                                            |
|:------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | <ul><li>'15 photos as Stan’s annual spooky Halloween display returns to South Shields for 2025 '</li><li>'Tesco Clubcard update as shoppers eligible for £2. 50 offer '</li><li>'Met Office issues a weather warning that could cause travel disruption for the UK '</li></ul>                      |
| 0     | <ul><li>"Best PS5 early Black Friday deals — here's 15 deals on must-play games and accessories I'd buy now "</li><li>'Thank You For Your Service! '</li><li>'TV Spy — Mayor of Kingstown, Tales of the Walking Dead, and all the US dramas you can watch on UK services: October 25-31 '</li></ul> |

## Evaluation

### Metrics
| Label   | Accuracy |
|:--------|:---------|
| **all** | 0.94     |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Super Furry Animals add three shows to UK tour ")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 3   | 12.975 | 40  |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 203                   |
| 1     | 197                   |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 2e-05
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch | Step | Training Loss | Validation Loss |
|:-----:|:----:|:-------------:|:---------------:|
| 0.001 | 1    | 0.3348        | -               |
| 0.05  | 50   | 0.3572        | -               |
| 0.1   | 100  | 0.2362        | -               |
| 0.15  | 150  | 0.131         | -               |
| 0.2   | 200  | 0.0582        | -               |
| 0.25  | 250  | 0.0101        | -               |
| 0.3   | 300  | 0.003         | -               |
| 0.35  | 350  | 0.0013        | -               |
| 0.4   | 400  | 0.001         | -               |
| 0.45  | 450  | 0.0007        | -               |
| 0.5   | 500  | 0.0006        | -               |
| 0.55  | 550  | 0.0005        | -               |
| 0.6   | 600  | 0.0005        | -               |
| 0.65  | 650  | 0.0005        | -               |
| 0.7   | 700  | 0.0005        | -               |
| 0.75  | 750  | 0.0004        | -               |
| 0.8   | 800  | 0.0004        | -               |
| 0.85  | 850  | 0.0003        | -               |
| 0.9   | 900  | 0.0003        | -               |
| 0.95  | 950  | 0.0003        | -               |
| 1.0   | 1000 | 0.0003        | -               |

### Framework Versions
- Python: 3.9.6
- SetFit: 1.1.3
- Sentence Transformers: 5.1.2
- Transformers: 4.57.6
- PyTorch: 2.8.0
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->