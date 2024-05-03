

Directory structure


├── LICENSE

├── Makefile           &lt;- Makefile with commands like `make data` or `make train`

├── README.md          &lt;- The top-level README for developers using this project.

├── data


│&nbsp;&nbsp; ├── external       &lt;- Data from third party sources.

│&nbsp;&nbsp; ├── interim        &lt;- Intermediate data that has been transformed.

│&nbsp;&nbsp; ├── processed      &lt;- The final, canonical data sets for modeling.

│&nbsp;&nbsp; └── raw            &lt;- The original, immutable data dump.

│
├── docs               &lt;- A default Sphinx project; see sphinx-doc.org for details

│
├── models             &lt;- Trained and serialized models, model predictions, or model summaries

│
├── notebooks          &lt;- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.

│
├── references         &lt;- Data dictionaries, manuals, and all other explanatory materials.

│
├── reports            &lt;- Generated analysis as HTML, PDF, LaTeX, etc.

│&nbsp;&nbsp; └── figures        &lt;- Generated graphics and figures to be used in reporting

│
├── requirements.txt   &lt;- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze &gt; requirements.txt`

│
├── setup.py           &lt;- Make this project pip installable with `pip install -e`

├── src                &lt;- Source code for use in this project.

│&nbsp;&nbsp; ├── __init__.py    &lt;- Makes src a Python module
│    │

│&nbsp;&nbsp; ├── data           &lt;- Scripts to download or generate data

│&nbsp;&nbsp; │&nbsp;&nbsp; └── make_dataset.py
│   │

│&nbsp;&nbsp; ├── features       &lt;- Scripts to turn raw data into features for modeling

│&nbsp;&nbsp; │&nbsp;&nbsp; └── build_features.py

│   │
│&nbsp;&nbsp; ├── models         &lt;- Scripts to train models and then use trained models to make
│   │   │                 predictions

│&nbsp;&nbsp; │&nbsp;&nbsp; ├── predict_model.py

│&nbsp;&nbsp; │&nbsp;&nbsp; └── train_model.py
│   │

│&nbsp;&nbsp; └── visualization  &lt;- Scripts to create exploratory and results oriented visualizations

│&nbsp;&nbsp;     └── visualize.py
│

└── tox.ini            &lt;- tox file with settings for running tox; see tox.readthedocs.io
