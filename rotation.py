import tensorflow as tf

from data_preprocessing import (
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test
)

from model import build_model
from evaluation import evaluate_model

rotation_augmentation = tf.keras.layers.RandomRotation(
    factor = 10 / 360,
    fill_mode="constant",
    fill_value=0.0
)

model = build_model(
    augmentation=rotation_augmentation
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=128,
    verbose=1
)

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

results = evaluate_model(
    model,
    x_test,
    y_test
)

print("\nRotation Augmentation Test Results")
print("---------------------")
print(f"Training Accuracy:   {final_train_accuracy:.4f}")
print(f"Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Test Loss:           {results['loss']:.4f}")
print(f"Test Accuracy:       {results['accuracy']:.4f}")
print(f"Precision:            {results['precision']:.4f}")
print(f"Recall:               {results['recall']:.4f}")
print(f"F1-score:             {results['f1']:.4f}")

print("\nConfusion Matrix:")
print(results["confusion_matrix"])