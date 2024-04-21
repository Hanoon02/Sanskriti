from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define the model name
model_name = "facebook/m2m100_418M"

# Download the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to disk
model.save_pretrained("saved_models/m2m100_418M")
tokenizer.save_pretrained("saved_models/m2m100_418M")
