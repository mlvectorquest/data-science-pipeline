# Fashion Forward Forecasting

A machine learning pipeline for StyleSense, an online women's clothing retailer. Many customer reviews are missing the "would you recommend this product" answer. This project trains a model that predicts it from the rest of the review.

## Summary

- **Task:** predict `Recommended IND` (1 = recommended, 0 = not recommended) for a review.
- **Data:** 18,442 reviews with numerical features (`Age`, `Positive Feedback Count`), categorical features (`Division Name`, `Department Name`, `Class Name`, `Clothing ID`) and text (`Title`, `Review Text`). The target is imbalanced: about 82% recommended and 18% not.
- **Approach:** a single scikit-learn `Pipeline` that combines TF-IDF on the review text, a review word-count feature, scaled numbers and a one-hot encoded department (with imputation for missing values), followed by a logistic regression with balanced class weights.
- **Evaluation:** stratified 90/10 train/test split and 5-fold cross-validation, scored on precision, recall and F1 for the "not recommended" class, because accuracy alone is misleading with imbalanced classes.
- **Result:** on the held-out test set the final model has an F1 of 0.70 for the "not recommended" class (recall 0.82, precision 0.62), close to the cross-validated estimate of 0.74.

## Files

| Path | Description |
|---|---|
| `starter/starter.ipynb` | The project notebook: data exploration, pipeline building, training, fine-tuning and the final test-set evaluation. |
| `starter/data/reviews.csv` | The review data used to train and test the model. |
| `requirements.txt` | Python packages needed to run the notebook. |
| `LICENSE.txt` | The project license. |
| `CODEOWNERS` | Repository ownership file. |

## Getting Started

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the notebook

```bash
jupyter notebook starter/starter.ipynb
```

Run the cells from top to bottom. The notebook reads `data/reviews.csv` relative to its own folder, so keep it in `starter/`. Cross-validation and the model comparison take a minute or two, mostly for the gradient boosting model.

## Built With

* [scikit-learn](https://scikit-learn.org/) - pipeline, models and evaluation
* [pandas](https://pandas.pydata.org/) - data loading and exploration
* [Jupyter Notebook](https://jupyter.org/) - the notebook environment

## License

[License](LICENSE.txt)
