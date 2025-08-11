from collections import Counter
import json
import sqlite3
import pandas as pd
from datasets import Dataset
from pathlib import Path
import logging
import re
import time
import shutil

class FeverDataProcessor:
    def __init__(self, dbpath: Path):
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.here = Path(__file__)
        
        # This is particular to my situation
        self.wikidbpath = dbpath
        self.challenge_types = {
            "other": True,
            "multi-hop reasoning": True,
            "combining tables and text": True,
            "numerical reasoning": True,
            "entity disambiguation": True,
            "search terms not in claim": True,
        }
    
    def database_health_check(self):
        with sqlite3.connect(self.wikidbpath) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            self.logger.info(f"Tables: {tables}")

            try:
                table_name = "wiki"
                cursor = conn.execute(f"PRAGMA table_info({table_name});")
                schema = cursor.fetchall()
                self.logger.info(f"Wiki Table Schema: {schema}")
            except:
                self.logger.warning(f"Could not find {table_name} table.")

    @staticmethod
    def clean_wikipedia_links(text):
        """
        Replace [[link|display_text]] with display_text
        Replace [[link]] with link
        """
        # Pattern 1: [[link|display_text]] -> display_text
        text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
        
        # Pattern 2: [[simple_link]] -> simple_link  
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        
        return text

    def process_data(self, tag: str,  jsonl_file_path: Path, output_dir_path: Path, chunk_size: int = 1000):
        processed_data = []
        chunk_num = 0
        total_processed = 0
        output_dir_path.mkdir(parents=True, exist_ok=True)
        assert jsonl_file_path.exists()
        with sqlite3.connect(self.wikidbpath) as conn: # Database Context Manager
            line_count = sum(1 for _ in open(jsonl_file_path, 'rb'))
            with open(jsonl_file_path, 'r', encoding='utf-8') as file: # File Context Manager
                cn = 0
                size = shutil.get_terminal_size()
                loading_bar = size.columns > 20
                bar_len = min(40, size.columns-10)
                loading_text = None
                backspace = 0
                
                # READING FILE
                for line in file.readlines():

                    # progress bar
                    progress = int(round((cn/line_count) * bar_len, 0))
                    if loading_text is not None: 
                        backspace = len(loading_text)
                        print("\b"*backspace, sep="", end="") # Removing Old 
                        print(" "*backspace, sep="", end="")
                        print("\b"*backspace, sep="", end="")
                    if loading_bar:
                        loading_text = f'|{"=" * (progress)}{" " * (bar_len - progress)}| {cn/line_count:.2%}'
                    else:
                        loading_text = f'{1/line_count:.2%}'
                    print(loading_text, sep="", end="", flush=True)
                    cn += 1
                    # PROGRESS BAR is OVER

                    cache = {}
                    if len(line.strip()) == 0:
                        continue
                    datum = json.loads(line)
                    challenge = datum.get("challenge", '').strip().casefold()
                    
                    # skipping numerical reasoning challenges
                    if challenge == '' or challenge == "numerical reasoning":
                        continue

                    claim_id = datum.get('id')
                    claim = datum.get('claim')
                    label = datum.get('label')
                    support = []

                    evidence = datum.get("evidence") # a list of { content, context }
                    for pair in evidence:
                        content = pair.get("content") # list of strings
                        for line in content:
                            parts = line.split("_") # [title, sentence, number]
                            if len(parts) > 2 and parts[1] == "sentence":

                                ## Check Cache First
                                page = cache.get(parts[0])
                                if page is None:
                                    cursor = conn.execute("SELECT data FROM wiki WHERE id=?", (parts[0],))
                                    result = cursor.fetchone()
                                    stringy = json.loads(result[0])
                                    # Cache System maybe
                                    cache[parts[0]] = stringy
                                    page = stringy
                                
                                ## We have the page here
                                sentence = page.get(f"{parts[1]}_{parts[2]}")
                                if sentence is not None:
                                    support.append(
                                        self.clean_wikipedia_links(sentence)
                                    )
                    # We now have our pieces of data
                    if len(support) > 0:
                        processed_data.append({
                            'id': claim_id,
                            'claim': claim,
                            'evidence': support,
                            'label': label,
                            'challenge': challenge,
                        })
                        total_processed += 1
                    
                    if len(processed_data) >= chunk_size:
                        # SAVE CHUNK into Parquet
                        df = pd.DataFrame(processed_data)
                        chunk_file = output_dir_path / f"fever_{tag}_chunk_{chunk_num:04d}.parquet"
                        df.to_parquet(chunk_file, index=False)
                        ## increment chunk
                        chunk_num += 1
                # End of Loop
                
                # Last Chunk
                if len(processed_data) > 0:
                    # SAVE CHUNK into Parquet
                    df = pd.DataFrame(processed_data)
                    chunk_file = output_dir_path / f"fever_{tag}_chunk_{chunk_num:04d}.parquet"
                    df.to_parquet(chunk_file, index=False)
                    ## increment chunk
                    chunk_num += 1
                
                # Finish Loading
                if loading_text is not None: 
                    backspace = len(loading_text)
                    print("\b"*backspace, sep="", end="") # Removing Old 
                    print(" "*backspace, sep="", end="")
                    print("\b"*backspace, sep="", end="")
                if loading_bar:
                    loading_text = f'|{"=" * (bar_len)}| {1:.2%}'
                else:
                    loading_text = f'{1/line_count:.2%}'
                print(loading_text, sep="", end="\n", flush=True) # New Line

if __name__ == '__main__':
    this_file = Path(__file__).resolve()
    p = this_file.parts
    dbpath = Path(p[0]) / p[1] / p[2] / 'Downloads' / 'feverous-wiki-pages-db' / 'feverous_wikiv1.db'

    this_dir = this_file.parent
    datasets_dir = this_dir / '.datasets'
    original_data_dir = datasets_dir / 'original'

    traindatapath = original_data_dir / 'feverous_train_challenges.jsonl'
    devdatapath = original_data_dir / 'feverous_dev_challenges.jsonl'

    outputpath = datasets_dir / 'processed'
    processor = FeverDataProcessor(dbpath)
    processor.database_health_check()
    print("Creating Training File")
    processor.process_data(
        "train",
        traindatapath,
        outputpath,
        100000
    )
    print("Creating Dev File")
    processor.process_data(
        "dev",
        devdatapath,
        outputpath,
        100000
    )