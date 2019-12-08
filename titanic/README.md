# Data Download

This sample uses the Titanic data set on kaggle.

If you have a kaggle account and kaggle-API is ready,
you can obtain the data from the command below:

```
kaggle competitions download -c titanic
unzip titanic.zip
```

# Usage Guide
## Model Training
```
python titanic_train.py run
```

## Prediction
```
python titanic_predict.py run
```

You can also designate the model by designating the run ID
of the TitanicModeling.

```
python titanic_predict.py run --id <ID of the TitanicModeling run>
```

If `--id` is not designated, the latest successful run is chosen.