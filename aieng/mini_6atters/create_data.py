# built-in libraries
from datetime import datetime
import gc
import json
from pathlib import Path
import re
from typing import List

# 3rd party libraries
import datasets
import pandas as pd
import requests
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

## to create a pipeline for text normalization
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import spacy

# custom libraries

def load_data() -> pd.DataFrame:
    raw_dfs: List[pd.DataFrame] = []
    with open(Path('./.data_sets/twitter_misinformation/train.json'), 'r') as file:
        raw_dfs.append(pd.DataFrame(json.load(file)))

    with open(Path('./.data_sets/twitter_misinformation/test.json'), 'r') as file:
        raw_dfs.append(pd.DataFrame(json.load(file)))

    return pd.concat(raw_dfs)

def create_text_processor():
    return TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'date', 'number'],
        annotate={"hashtag", "allcaps", "elongated", "repeated"},
        fix_html=True,  # fix HTML tokens
        segmenter="twitter", 
        corrector="twitter", 
        unpack_hashtags=True,  # split hashtag words
        spell_correct_elong=False,  # keep it fast
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]
    )

def preprocess_tweet(text: str, text_processor, nlp):
    # Step 1: Social media normalization
    cleaned_text = text_processor.pre_process_doc(text)
    # print(cleaned_text)
    
    # Step 2: Simple sentence splitting (for long tweets)
    if len(cleaned_text) > 30:  # Only split very long text
        doc = nlp(" ".join(cleaned_text))
        sentences = [sent.text.strip() for sent in doc.sents]
        # possibility to split further for better processing...
        return " ".join(sentences)
    
    return " ".join(cleaned_text)

def extract_5w1h(clean_tweet: str):
    prompt = f"""
    Please extract the 5W1H elements (who, what, where, when, why, and how) from this tweet. Return your result as JSON:
    
    Tweet: "{clean_tweet}"
    
    Extract:
    - who: The main person/entity being discussed
    - what: The main action or claim
    - where: Location or null
    - when: Datetime or null
    - why: Reason or null
    - how: Method or null
    
    Return only valid JSON without additional formatting. Use null for missing elements. Please be concise.
    """
    
    # Call Ollama API
    # response = requests.post('http://172.17.0.1:11434/api/generate',
    # response = requests.post('http://172.26.112.1:11434/api/generate',
    response = requests.post('http://localhost:11434/api/generate',
        json={'model': 'deepseek-llm:7b', 'prompt': prompt, "stream": False})
    
    return response
    
def serialize_ollama(response_json: str):
    serialize = None
    try:
        tmp = response_json.get('response')
        tmp = re.sub(r'\n', ' ', tmp).strip()
        assert tmp is not None
        serialize = json.loads(tmp)
    except:
        print("Unable to serialize")

    # Logging:
    return serialize

# Probably better as a class
def create_synth_data(data: pd.DataFrame, entries: int, text_processor, nlp):
    counter = 0
    index = 0
    max_tries = len(data)
    synth_data_list = []
    while counter < entries and index < max_tries:
        print(f"Index: {index}, Counter: {counter}")
        try:
            dirty_tweet = data.iloc[index]['text']

            # increment index - failure after this point moves to next case
            index += 1

            clean_tweet = preprocess_tweet(dirty_tweet, text_processor, nlp)
            
            # returns full response
            response = extract_5w1h(clean_tweet)
            response_json = response.json()

            # try serialize response
            serialize = serialize_ollama(response_json)

            # Logging?
            with open('./test.json.log', 'a') as file:
                entry = json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'original': dirty_tweet,
                    'cleaned': clean_tweet,
                    'result': response_json,
                    'json': serialize,
                }, indent=2)
                file.write(f"{entry}\n")

            assert serialize is not None
            
            # First w/out cleaning input I suppose...
            # to see if transformer can make the leap
            synth_data_list.append({
                'input': dirty_tweet,
                'cleaned_input': clean_tweet,
                'output': serialize,
            })
            print(serialize)
            counter += 1
        except Exception as err:
            print(err)
    
    return synth_data_list



def main():
    # Setting entries here to be obvious
    entries = 200

    print('loading data')
    df = load_data()
    print('data loaded')
    print('Configuring Ekphrasis for Social Media posts')
    text_processor = create_text_processor()
    print("Text Processor Created")
    print('Loading NLP')
    nlp = spacy.load("en_core_web_sm")
    print('NLP loaded!')
    print('Processing...')
    synth_data_list = create_synth_data(
        df, entries, text_processor, nlp
    )
    print('processing complete...')
    print('Saving data')

    # Save Data
    with open(f'./synthdata{datetime.now().strftime("%Y%m%d%H%M%S")}.json', 'w') as file:
        json.dump(synth_data_list, file, indent=2)
            
    print("All Done")

if __name__ == "__main__":
    main()