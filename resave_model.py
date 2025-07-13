from tensorflow.keras.models import load_model

# Load your uploaded model
model = load_model("trained_modelrev.h5", compile=False)

# Save a new, compatible version
model.save("converted_model.h5", include_optimizer=False)

print(" Model saved as 'converted_model.h5'")
