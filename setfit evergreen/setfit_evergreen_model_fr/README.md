---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 'Meurtre de Lola : le visage de Dahbia B. dévoilé sur les réseaux sociaux
    et dans TPMP '
- text: '10 films à regarder en famille sur Netflix pendant les vacances - Netflix
    News '
- text: 'Origami d''une Chauve souris en papier '
- text: 'Vincent Lagaf revient sur sa lourde opération au cœur : "J’étais à deux doigts
    de l’accident vasculaire" | Télé 7 Jours '
- text: 'Sigean : derniers préparatifs avant l’ouverture du bazar '
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
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
| Label | Examples                                                                                                                                                                                                                                     |
|:------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | <ul><li>'Halloween Party '</li><li>'Sorties Animations, f&#xEA;tes &#xFFFD; Tournon-sur-Rh&#xF4;ne et 30km autour - du 27/10/2025 au 03/11/2025 avec planetekiosque, page 2 '</li><li>'Loisirs enfants : Une nouvelle directrice '</li></ul> |
| 0     | <ul><li>"Recette d'Halloween : balais de sorcière au fromage "</li><li>'Créez un thermomètre personnalisé pour les enfants '</li><li>'30 idées déco Halloween à faire soi-même '</li></ul>                                                   |

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
preds = model("Origami d'une Chauve souris en papier ")
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
| Word count   | 2   | 12.11  | 25  |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 236                   |
| 1     | 164                   |

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
| 0.001 | 1    | 0.0164        | -               |
| 0.05  | 50   | 0.0642        | -               |
| 0.1   | 100  | 0.0365        | -               |
| 0.15  | 150  | 0.005         | -               |
| 0.2   | 200  | 0.0002        | -               |
| 0.25  | 250  | 0.0001        | -               |
| 0.3   | 300  | 0.0001        | -               |
| 0.35  | 350  | 0.0001        | -               |
| 0.4   | 400  | 0.0001        | -               |
| 0.45  | 450  | 0.0001        | -               |
| 0.5   | 500  | 0.0           | -               |
| 0.55  | 550  | 0.0           | -               |
| 0.6   | 600  | 0.0           | -               |
| 0.65  | 650  | 0.0           | -               |
| 0.7   | 700  | 0.0           | -               |
| 0.75  | 750  | 0.0           | -               |
| 0.8   | 800  | 0.0           | -               |
| 0.85  | 850  | 0.0           | -               |
| 0.9   | 900  | 0.0           | -               |
| 0.95  | 950  | 0.0           | -               |
| 1.0   | 1000 | 0.0           | -               |

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