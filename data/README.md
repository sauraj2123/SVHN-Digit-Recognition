# Data

The dataset is **not included** in this repository (~490 MB). It was provided to participants of the MIT Applied AI & Data Science Program as a preprocessed subset of the [SVHN dataset](http://ufldl.stanford.edu/housenumbers/).

To run the notebook locally, place the file here:

```
data/
└── SVHN_single_grey1.h5
```

The file contains four arrays:

| Key | Shape | Description |
| --- | --- | --- |
| `X_train` | (42000, 32, 32) | grayscale training images |
| `y_train` | (42000,) | digit labels 0–9 |
| `X_test` | (18000, 32, 32) | grayscale test images |
| `y_test` | (18000,) | digit labels 0–9 |
