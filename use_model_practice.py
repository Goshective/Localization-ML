from csv import reader
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

"""Import and Translation"""

CUR_MODEL_VERSION = 3
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

simple_text = "translate English to Russian: What a wonderful day!"
simple_text = "translate Russian to English: Какой чудесный день!"

repo_name = "Goshective/opus_books_model_1"

# Change `xx` to the language of the input and `yy` to the language of the desired output.
# Examples: "en" for English, "fr" for French, "de" for German, "es" for Spanish, "zh" for Chinese, etc; translation_en_to_fr translates English to French
# You can view all the lists of languages here - https://huggingface.co/languages

"""AUTO"""

translator = pipeline("translation_ru_to_en", model=repo_name, device='cuda')
print(translator(simple_text, max_length=400)[0]['translation_text'])


"""SETTING UP"""

tokenizer = AutoTokenizer.from_pretrained(repo_name)
inputs = tokenizer(simple_text, return_tensors="pt").input_ids

model = AutoModelForSeq2SeqLM.from_pretrained(repo_name)
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))