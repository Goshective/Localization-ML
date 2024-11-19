from csv import reader
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

"""Import and Translation"""

TRANSLATION_MODE = "en-fr"
CUR_MODEL_NAME = ""
DECODE = {
    'ru': "Russian",
    'en': "English",
    'fr': "French"
    }

with open('model_names.csv', 'r') as f:
    csvreader = reader(f, delimiter=';')
    next(csvreader, None)
    for model_id, model_name, translation_mode in csvreader: # finds last trained version
        if translation_mode == TRANSLATION_MODE:
            CUR_MODEL_NAME = model_name

if not CUR_MODEL_NAME:
    exit(0)

source_lang = TRANSLATION_MODE[:2]
target_lang = TRANSLATION_MODE[3:]
prefix = f"translate {DECODE[source_lang]} to {DECODE[target_lang]}: "

texts_pack = []
with open('sample_texts.csv', 'r', encoding='utf-8') as f:
    csvreader = reader(f, delimiter=';')
    next(csvreader, None)
    for lang, text in csvreader: # finds last trained version
        if lang == source_lang:
            texts_pack.append(text)


# Change `xx` to the language of the input and `yy` to the language of the desired output.
# Examples: "en" for English, "fr" for French, "de" for German, "es" for Spanish, "zh" for Chinese, etc; translation_en_to_fr translates English to French
# You can view all the lists of languages here - https://huggingface.co/languages

def pipeline_translation(text):
    """AUTO"""

    translator = pipeline(f"translation_{source_lang}_to_{target_lang}", model=CUR_MODEL_NAME)
    return translator(text, max_length=400)[0]['translation_text']

def precise_translation(text):
    """SETTING UP"""

    tokenizer = AutoTokenizer.from_pretrained(CUR_MODEL_NAME)
    inputs = tokenizer(text, return_tensors="pt").input_ids

    model = AutoModelForSeq2SeqLM.from_pretrained(CUR_MODEL_NAME)
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


for i, text in enumerate(texts_pack):
    print("TEXT", i + 1)
    print(text)
    print()

    print("Auto-translation:")
    print(pipeline_translation(text))
    print()

    print("My translation:")
    print(precise_translation(text))
    print()