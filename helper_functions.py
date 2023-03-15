import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();


def save_model_to_drive(project_name, model):
  path = "/content/drive/MyDrive/Kaggle/" + project_name + "/" + model.name
  model.save(path)


def na_values_table_for_dataframe(df, count=20):
  """
  For the dataframe output the top {count} values of features with NA values
  """
  total = df.isnull().sum().sort_values(ascending=False)
  percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
  missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

  return missing_data.head(count)


def calculate_model_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.


  Args:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.

  Based on https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/helper_functions.py#L270
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100

  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

  return {
      "accuracy": model_accuracy,
      "precision": model_precision,
      "recall": model_recall,
      "f1": model_f1
  }


def compare_model_metrics(models, val_features, val_labels):
  results = {}

  for model in models:
    pred_probs = model.predict(val_features)
    preds = tf.squeeze(tf.round(pred_probs))

    results[model.name] = calculate_model_results(y_true = val_labels, y_pred = preds)

  return pd.DataFrame.from_dict(results).transpose().sort_values(by=['accuracy'])


def create_checkpoint_callback(filepath, save_weights_only=True, save_best_only=True, monitor='val_accuracy', verbose=1):
  """
  Return a create checkpoin callback
  """
  return tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    save_weights_only=save_weights_only,
    save_best_only=save_best_only,
    monitor=monitor,
    verbose=verbose,
  )


def create_early_stopping_callback(monitor='val_loss', patience=10, mode='min'):
  """
  Return an early stopping callback
  """
  return tf.keras.callbacks.EarlyStopping(
    monitor=monitor,
    patience=patience,
    mode=mode,
  )