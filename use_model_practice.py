"""Import and Translation"""


text_en_ru = "translate English to Russian: Legumes share resources with nitrogen-fixing bacteria."
text_ru_en = "translate Russian to English: Бобовые растения делят ресурсы с азотфиксирующими бактериями."

from transformers import pipeline

# Change `xx` to the language of the input and `yy` to the language of the desired output.
# Examples: "en" for English, "fr" for French, "de" for German, "es" for Spanish, "zh" for Chinese, etc; translation_en_to_fr translates English to French
# You can view all the lists of languages here - https://huggingface.co/languages
translator = pipeline("translation_ru_to_en", model="Goshective/my_awesome_opus_books_model")
translator(text_ru_en)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Goshective/my_awesome_opus_books_model")
inputs = tokenizer(text_ru_en, return_tensors="pt").input_ids


from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("Goshective/my_awesome_opus_books_model")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

tokenizer.decode(outputs[0], skip_special_tokens=True)