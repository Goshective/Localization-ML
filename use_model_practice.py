from csv import reader
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

"""Import and Translation"""

CUR_MODEL_VERSION = 1
CUR_MODEL_NAME = ""

with open('model_names.csv', 'r') as f:
    csvreader = reader(f, delimiter=';')
    next(csvreader, None)
    for model_id, model_name in csvreader:
        if int(model_id) == CUR_MODEL_VERSION:
            CUR_MODEL_NAME = model_name
            break

if not CUR_MODEL_NAME:
    exit(0)

text_en_ru = "translate English to Russian: Legumes share resources with nitrogen-fixing bacteria."
text_ru_en = "translate Russian to English: Бобовые растения делят ресурсы с азотфиксирующими бактериями."

# Change `xx` to the language of the input and `yy` to the language of the desired output.
# Examples: "en" for English, "fr" for French, "de" for German, "es" for Spanish, "zh" for Chinese, etc; translation_en_to_fr translates English to French
# You can view all the lists of languages here - https://huggingface.co/languages
translator = pipeline("translation_ru_to_en", model=CUR_MODEL_NAME)
translator(text_ru_en)


tokenizer = AutoTokenizer.from_pretrained(CUR_MODEL_NAME)
inputs = tokenizer(text_ru_en, return_tensors="pt").input_ids


model = AutoModelForSeq2SeqLM.from_pretrained(CUR_MODEL_NAME)
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

tokenizer.decode(outputs[0], skip_special_tokens=True)