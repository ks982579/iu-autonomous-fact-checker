# 1st party
import re
from collections import Counter
from typing import List
from pathlib import Path

# 3rd party
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import pandas as pd
import spacy
from spacy_experimental.coref.coref_component import DEFAULT_COREF_MODEL, CoreferenceResolver
from spacy_experimental.coref.coref_util import DEFAULT_CLUSTER_PREFIX
# import neuralcoref...

"""
https://spacy.io/usage/linguistic-features

Has a list of the Parts Of Speech (pos). 
https://spacy.io/usage/linguistic-features#named-entities
There are also entities, similar to the POS proper nouns

https://spacy.io/universe/project/neuralcoref
What I am looking for I think
https://medium.com/nlplanet/two-minutes-nlp-quick-intro-to-coreference-resolution-with-neuralcoref-7fa2be2c4284
To install
https://github.com/huggingface/neuralcoref
This hasn't been updated since 2019 and issues reveal it's not working for python>=3.9

"""

# https://github.com/cbaziotis/ekphrasis
# website includes a spell checker too!

# Mine
# ...

class MyTextPreProcessor():
    """
    This class was used for preprocessing and getting search queries.
    Results were not great - it's out of commission currently
    """
    def __init__(self):
        self.text_preprocessor = self._create_text_processor()
        self.nlp = spacy.load("en_core_web_sm") # sm is efficient but using _trf could be more accurate
        self.cache = None # remember last...

    def get_sentences(self, text: str, cache: bool = True) -> List[str]:
        doc = self.nlp(text)
        if cache:
            self.cache = doc
        sentences = [sentence.text.strip() for sentence in doc.sents]
        return sentences
    
    # Run after determining IF text was a claim.
    def create_search_queries(self, text):
        doc = self.nlp(text) # should be one sentence

        entities = []
        for ent in doc.ents:
            # print(dir(ent))
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]:
                entities.append(ent.text)
        
        # Get important keywords (nouns, verbs)
        keywords = []
        for token in doc:
            if token.pos_ in ["VERB"] and len(token.text) > 3:
            # if token.pos_ in ["NOUN", "VERB"] and len(token.text) > 3:
                keywords.append(token.lemma_.lower())
        
        # Create multiple simple queries
        queries = []
        
        if entities:
            # Query 1: Just the main person/org
            queries.append(entities)
            
            # Query 2: Person + one keyword  
            # if keywords:
            #     queries.append(f"{entities[0]} {keywords[0]}")
            #     print(keywords)
        if keywords:
            queries.append(keywords)
        
        # Query 3: Top 2-3 keywords only
        # if len(keywords) >= 2:
        #     queries.append(" ".join(keywords[:2]))
        
        # flatten
        flattened = [thing for that in queries for thing in that]
        
        return flattened

    
    def preprocess_text(self, dirty_text: str):
        """
        This is quite useless at the moment...

        Args:
            dirty_text (str): _description_

        Returns:
            _type_: _description_
        """
        # I believe this creates tokens -> so a big list of words and stuff
        cleaned_text = self.text_preprocessor.pre_process_doc(dirty_text)
        return cleaned_text

    @staticmethod
    def _create_text_processor():
        return TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                'time', 'date', 'number'],
            annotate={"hashtag", "allcaps", "elongated", "repeated", "emphasis", "censored"},
            fix_html=True,  # fix HTML tokens
            segmenter="twitter", 
            corrector="twitter", 
            unpack_hashtags=True,  # split hashtag words
            spell_correct_elong=False,  # keep it fast
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )


class KeywordExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Comprehensive Exclusion List of words that hurt news search
        # Most found with manual testing - they don't add meaningful context to API search queries
        self.exclusion_list = {
            # Time/Date words (too generic)
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 
            'september', 'october', 'november', 'december',
            'today', 'yesterday', 'tomorrow', 'morning', 'afternoon', 'evening', 'night',
            'week', 'month', 'year', 'day', 'time', 'date', 'recently', 'latest',
            
            # Communication/Reporting verbs
            'said', 'says', 'saying', 'told', 'tells', 'telling', 'reported', 'reports', 
            'reporting', 'announced', 'announces', 'announcing', 'stated', 'states', 
            'stating', 'claimed', 'claims', 'claiming', 'declared', 'declares', 
            'according', 'spokesperson', 'statement', 'comments', 'commented',
            
            # Generic people/groups
            'people', 'person', 'individuals', 'citizens', 'residents', 'officials',
            'authorities', 'sources', 'experts', 'analysts', 'observers', 'witnesses',
            'group', 'groups', 'team', 'teams', 'organization', 'organizations',
            'committee', 'committees', 'board', 'boards', 'members', 'member',
            
            # Generic business/institutional words
            'company', 'companies', 'business', 'businesses', 'corporation', 'corporations',
            'firm', 'firms', 'agency', 'agencies', 'department', 'departments',
            'office', 'offices', 'administration', 'government', 'federal', 'state',
            'local', 'public', 'private', 'sector', 'industry', 'market', 'markets',
            
            # Media/Information words
            'news', 'media', 'press', 'article', 'articles', 'story', 'stories',
            'report', 'reports', 'information', 'data', 'details', 'facts',
            'coverage', 'interview', 'interviews', 'documentary', 'broadcast',
            
            # Generic action words
            'happened', 'occurs', 'occurred', 'occurring', 'takes', 'took', 'taking',
            'makes', 'made', 'making', 'does', 'done', 'doing', 'goes', 'went', 'going',
            'comes', 'came', 'coming', 'gets', 'got', 'getting', 'gives', 'gave', 'giving',
            'shows', 'showed', 'showing', 'seems', 'appeared', 'appears', 'looks',
            'becomes', 'became', 'becoming', 'remains', 'stayed', 'continues', 'continued',
            
            # Generic descriptors
            'things', 'thing', 'stuff', 'matter', 'matters', 'issue', 'issues',
            'situation', 'situations', 'case', 'cases', 'instance', 'instances',
            'example', 'examples', 'way', 'ways', 'means', 'method', 'methods',
            'process', 'processes', 'system', 'systems', 'program', 'programs',
            
            # Generic adjectives
            'good', 'bad', 'great', 'small', 'large', 'big', 'little', 'old', 'new',
            'long', 'short', 'high', 'low', 'important', 'significant', 'major', 'minor',
            'serious', 'critical', 'potential', 'possible', 'likely', 'unlikely',
            'various', 'different', 'similar', 'same', 'other', 'another', 'additional',
            
            # Generic numbers/quantities
            'number', 'numbers', 'amount', 'amounts', 'total', 'totals', 'count',
            'many', 'much', 'more', 'most', 'less', 'least', 'few', 'several',
            'some', 'any', 'all', 'every', 'each', 'both', 'either', 'neither',
            
            # Generic locations (keep specific ones like "Washington" or "California")
            'place', 'places', 'location', 'locations', 'area', 'areas', 'region',
            'regions', 'zone', 'zones', 'site', 'sites', 'building', 'buildings',
            'facility', 'facilities', 'center', 'centers', 'headquarters',
            
            # Legal/procedural (often too generic)
            'investigation', 'investigations', 'trial', 'trials', 'hearing', 'hearings',
            'meeting', 'meetings', 'conference', 'conferences', 'session', 'sessions',
            'decision', 'decisions', 'ruling', 'rulings', 'verdict', 'verdicts',
            
            # Generic emotions/reactions
            'concerns', 'worried', 'worried', 'angry', 'upset', 'disappointed',
            'pleased', 'happy', 'satisfied', 'frustrated', 'concerned', 'interested',
            
            # Common conjunctions/prepositions that sometimes get picked up
            'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
            'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
            'during', 'except', 'inside', 'outside', 'through', 'throughout', 'under',
            'until', 'within', 'without', 'regarding', 'concerning', 'involving',
            
            # Generic outcomes/results
            'result', 'results', 'outcome', 'outcomes', 'effect', 'effects', 'impact',
            'impacts', 'consequence', 'consequences', 'response', 'responses', 'reaction',
            'reactions', 'change', 'changes', 'development', 'developments'
        }

    def get_sentences(self, text: str, cache: bool = True) -> List[str]:
        doc = self.nlp(text)
        if cache:
            self.cache = doc
        sentences = [sentence.text.strip() for sentence in doc.sents]
        return sentences
    
    def extract_keywords(self, text):
        cleantext = self._preprocess_text(text)
        doc = self.nlp(cleantext)
        
        # Get named entities first (highest priority)
        entities = []
        for ent in doc.ents:
            # print(f"ENTITY: {ent} | LABEL: {ent.label_}")
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "NORP"]: # DATE PERCENT CARDINAL
                clean_ent = ent.text.strip()
                if len(clean_ent) > 1 and clean_ent.lower() not in self.exclusion_list:
                    entities.append(clean_ent)
        
        # Get important nouns/adjectives as backup
        backup_words = []
        for token in doc:
            # print(f"TOKENS: {token.text} : {token.pos} {token.pos_} || {token.lemma} : {token.lemma_}")
            # print(f"TOKENS: {token.text} | POS: {token.pos_} | LEMMA: {token.lemma_}")
            # FYI: Emojis seem to fall into the "PROP" category. 
            if (
                token.pos_ in ["NOUN", "ADJ", "VERB"] and 
                len(token.text) > 3 and 
                not token.is_stop and 
                token.text.lower() not in self.exclusion_list and
                token.is_alpha
            ):
                if token.lemma_.casefold() not in [y.casefold() for y in backup_words]:
                    backup_words.append(token.lemma_)
        
        # Filter backup_words to exclude items already in entities
        backup_words = [x for x in backup_words if x.casefold() not in [y.casefold() for y in entities]]
        
        # Prioritize entities, use backup if needed
        that = [entities, backup_words]
        keywords = [thing for this in that for thing in this]
        # keywords = entities[:4] if entities else backup_words[:3]
        
        return keywords

    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for better classification
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Completely Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Replace all @mentions with {{USERNAME}}
        text = re.sub(r'@\w+', '{{USERNAME}}', text)
        text = re.sub(r'\n+', ' ', text)
        
        # removing posessives for keywords
        text = re.sub(r"'s$", "", text)

        # How this DistilBERT model was trained
        return text

"""
['_', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__',
'__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__',
'__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__',
'__ne__', '__new__', '__pyx_vtable__', '__reduce__', '__reduce_ex__', '__repr__',
'__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_fix_dep_copy', '_vector',
'_vector_norm', 'as_doc', 'char_span', 'conjuncts', 'doc', 'end', 'end_char',
'ent_id', 'ent_id_', 'ents', 'get_extension', 'get_lca_matrix', 'has_extension',
'has_vector', 'id', 'id_', 'kb_id', 'kb_id_', 'label', 'label_', 'lefts', 'lemma_',
'n_lefts', 'n_rights', 'noun_chunks', 'orth_', 'remove_extension', 'rights',
'root', 'sent', 'sentiment', 'sents', 'set_extension', 'similarity', 'start',
'start_char', 'subtree', 'tensor', 'text', 'text_with_ws', 'to_array', 'vector', 'vector_norm', 'vocab']
"""
"""Tokens
[...,
'ancestors', 'check_flag',
'children', 'cluster', 'conjuncts', 'dep', 'dep_', 'doc', 'ent_id', 'ent_id_',
'ent_iob', 'ent_iob_', 'ent_kb_id', 'ent_kb_id_', 'ent_type', 'ent_type_',
'get_extension', 'has_dep', 'has_extension', 'has_head', 'has_morph', 'has_vector',
'head', 'i', 'idx', 'iob_strings', 'is_alpha', 'is_ancestor', 'is_ascii',
'is_bracket', 'is_currency', 'is_digit', 'is_left_punct', 'is_lower', 'is_oov',
'is_punct', 'is_quote', 'is_right_punct', 'is_sent_end', 'is_sent_start', 'is_space',
'is_stop', 'is_title', 'is_upper', 'lang', 'lang_', 'left_edge', 'lefts', 'lemma',
'lemma_', 'lex', 'lex_id', 'like_email', 'like_num', 'like_url', 'lower', 'lower_',
'morph', 'n_lefts', 'n_rights', 'nbor', 'norm', 'norm_', 'orth', 'orth_', 'pos',
'pos_', 'prefix', 'prefix_', 'prob', 'rank', 'remove_extension', 'right_edge', 'rights',
'sent', 'sent_start', 'sentiment', 'set_extension', 'set_morph', 'shape',
'shape_', 'similarity', 'subtree', 'suffix', 'suffix_', 'tag', 'tag_', 'tensor',
'text', 'text_with_ws', 'vector', 'vector_norm', 'vocab', 'whitespace_']
"""

class PronounReplacer:
    """
    The process is Coreference resolution - and will help with research in claim resolution.
    I was trying to avoid building another model.
    This solution is OK - has some gaps but works for now
    """
    def __init__(self, nlp_model="en_core_web_sm"):
        # https://spacy.io/api/coref
        # Follow the docs I think
        config = {
            "model": DEFAULT_COREF_MODEL,
            "span_cluster_prefix": DEFAULT_CLUSTER_PREFIX
        }
        self.nlp = spacy.load("en_core_web_sm")
        # self.coref = self.nlp.add_pipe("experimental_coref", config=config)
        self.coref = CoreferenceResolver(self.nlp.vocab, DEFAULT_COREF_MODEL)
        # self.coref.initialize(lambda: examples, nlp=self.nlp)
    
    def experimental_coreferencing(self, text):
        doc = self.nlp(text)
        # import pdb; pdb.set_trace()
        return self.coref.pipe(self.nlp(text))
        
    def replace_pronouns_rule_based(self, text):
        """
        Rule-based pronoun replacement using SpaCy's dependency parsing
        and named entity recognition.
        Model based approach didn't work - and I am not building one at the moment.
        """
        doc = self.nlp(text)
        
        # Extract entities with their positions
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start_token': ent.start,
                'end_token': ent.end,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
        
        # print("Entities")
        # print(entities)
        # print()
        
        # Build replacement mappings
        replacements = self._build_replacement_mappings(doc, entities)

        # print("Replacements")
        # print(replacements)
        # print()
        
        # Apply replacements
        modified_text = self._apply_replacements(text, doc, replacements)
        return modified_text
    
    def _build_replacement_mappings(self, doc, entities):
        """Build mappings from pronoun positions to replacement text"""
        replacements = {}
        
        # Group entities by type for easier lookup
        person_entities = [e for e in entities if e['label'] == 'PERSON']
        org_entities = [e for e in entities if e['label'] == 'ORG']
        norp_entities = [e for e in entities if e['label'] == 'NORP']
        gpe_entities = [e for e in entities if e['label'] == 'GPE']  # Countries, cities, states

        for token in doc:
            # print(f"{token.lemma_} | {token.pos_}")
            if token.pos_ == 'PRON':
                replacement = self._find_best_referent(
                    token, doc, person_entities, org_entities, norp_entities, gpe_entities
                )
                # print("Finding replacement")
                # print(replacement)
                if replacement:
                    replacements[token.i] = replacement
        
        return replacements
    
    def _find_best_referent(self, pronoun_token, doc, person_entities, org_entities, norp_entities, gpe_entities):
        """Find the best referent for a pronoun"""
        pronoun = pronoun_token.text.lower()
        pronoun_pos = pronoun_token.i
        
        # Helper function to clean possessive 's from entity text
        def clean_entity_text(text):
            return re.sub(r"'s$", "", text)
        
        # Only consider entities that appear before the pronoun
        def get_preceding_entities(entity_list):
            return [e for e in entity_list if e['end_token'] <= pronoun_pos]
        
        preceding_persons = get_preceding_entities(person_entities)
        preceding_orgs = get_preceding_entities(org_entities)
        preceding_norps = get_preceding_entities(norp_entities)
        preceding_gpes = get_preceding_entities(gpe_entities)
        
        # Apply pronoun-specific rules
        if pronoun in ['he', 'him', 'his', 'himself']:
            if preceding_persons:
                return clean_entity_text(preceding_persons[-1]['text'])  # Most recent person
                
        elif pronoun in ['she', 'her', 'hers', 'herself']:
            if preceding_persons:
                return clean_entity_text(preceding_persons[-1]['text'])  # Most recent person
                
        elif pronoun in ['it', 'its', 'itself']:
            # Prefer organizations, then fall back to other entities
            if preceding_orgs:
                return clean_entity_text(preceding_orgs[-1]['text'])
            elif preceding_gpes:
                return clean_entity_text(preceding_gpes[-1]['text'])
            elif preceding_persons:
                return clean_entity_text(preceding_persons[-1]['text'])
                
        elif pronoun in ['they', 'them', 'their', 'theirs', 'themselves']:
            # Prefer group entities (NORP), then fall back to persons
            # print(preceding_norps)
            # print(preceding_orgs)
            # print(preceding_persons)
            if preceding_norps:
                return clean_entity_text(preceding_norps[-1]['text'])
            elif preceding_gpes:
                return clean_entity_text(preceding_gpes[-1]['text'])
            elif preceding_orgs:
                return clean_entity_text(preceding_orgs[-1]['text'])
            elif preceding_persons:
                return clean_entity_text(preceding_persons[-1]['text'])
    
    def _apply_replacements(self, original_text, doc, replacements):
        """Apply the pronoun replacements to the original text"""
        if not replacements:
            return original_text
        
        # Work backwards through the text to avoid position shifts
        modified_text = original_text
        
        # Sort by token position in reverse order
        for token_pos in sorted(replacements.keys(), reverse=True):
            token = doc[token_pos]
            replacement_text = replacements[token_pos]

            # Handle possessive pronouns - add 's if needed (trying per testing)
            if token.text.lower() in ['his', 'her', 'its', 'their']:
                if not replacement_text.endswith("'s"):
                    replacement_text += "'s"
            
            # Handle capitalization
            if token.text[0].isupper():
                replacement_text = replacement_text[0].upper() + replacement_text[1:]
            
            # Replace the pronoun
            start_char = token.idx
            end_char = token.idx + len(token.text)
            
            modified_text = (
                modified_text[:start_char] + 
                replacement_text + 
                modified_text[end_char:]
            )
        
        return modified_text

    def replace_pronouns_dependency_enhanced(self, text):
        """
        Enhanced version with dependency parsing to better understand relationships.
        Currently - Same results as the rule based...
        """
        doc = self.nlp(text)
        
        # Find subject-verb-object relationships
        subjects = []
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass'] and token.pos_ in ['PROPN', 'NOUN']:
                subjects.append({
                    'token': token,
                    'head': token.head,
                    'position': token.i
                })
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start_token': ent.start,
                'end_token': ent.end,
                'tokens': [doc[i] for i in range(ent.start, ent.end)]
            })
        
        # Build enhanced mappings using syntactic information
        replacements = {}
        for token in doc:
            if token.pos_ == 'PRON':
                replacement = self._find_syntactic_referent(token, doc, entities, subjects)
                if replacement:
                    replacements[token.i] = replacement
        
        return self._apply_replacements(text, doc, replacements)
    
    def _find_syntactic_referent(self, pronoun_token, doc, entities, subjects):
        """Find referent using syntactic information"""
        pronoun = pronoun_token.text.lower()
        
        # Look for entities that are subjects in similar syntactic positions
        preceding_entities = [e for e in entities if e['end_token'] <= pronoun_token.i]
        
        if not preceding_entities:
            return None
        
        # Clean entity text (remove possessive "'s" per testing)
        def clean_entity_text(text):
            return re.sub(r"'s$", "", text)
        
        # Apply specific logic based on pronoun type and entity matching
        if pronoun in ['they', 'them', 'their', 'theirs', 'themselves']:
            # For plural pronouns, prioritize NORP (groups) over PERSON
            norp_entities = [e for e in preceding_entities if e['label'] == 'NORP']
            if norp_entities:
                return clean_entity_text(norp_entities[-1]['text'])  # Most recent group
            
            gpe_entities = [e for e in preceding_entities if e['label'] == 'GPE']
            if gpe_entities:
                return clean_entity_text(gpe_entities[-1]['text'])  # Most recent country/place
            
            org_entities = [e for e in preceding_entities if e['label'] == 'ORG']
            if org_entities:
                return clean_entity_text(org_entities[-1]['text'])  # Most recent organization
            
            # If no NORP, check if there are multiple persons (could be collective)
            person_entities = [e for e in preceding_entities if e['label'] == 'PERSON']
            if person_entities:
                return clean_entity_text(person_entities[-1]['text'])
        
        # For singular pronouns, find the closest appropriate entity
        best_entity = None
        min_distance = float('inf')
        
        for entity in preceding_entities:
            distance = pronoun_token.i - entity['end_token']
            
            # Check if entity type matches pronoun
            if self._entity_pronoun_match(entity, pronoun):
                if distance < min_distance:
                    min_distance = distance
                    best_entity = entity
        
        return clean_entity_text(best_entity['text']) if best_entity else None

    def _entity_pronoun_match(self, entity, pronoun):
        """Check if entity type matches pronoun"""
        if pronoun in ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself']:
            return entity['label'] == 'PERSON'
        elif pronoun in ['it', 'its', 'itself']:
            return entity['label'] in ['ORG', 'GPE' 'PERSON']  # Could be either
        elif pronoun in ['they', 'them', 'their', 'theirs', 'themselves']:
            return entity['label'] in ['NORP', 'GPE', 'ORG', 'PERSON']
        return False


## ----------------------------------------------------
## ----------------------------------------------------
## ----------------------------------------------------


# ------------- TESTING ---------------

test_easy = [
    "The president announced a new climate policy yesterday.",
    "Apple's stock price increased by 15% last week.", 
    "COVID-19 vaccines are 95% effective against severe illness.",
    "Tesla stock price dropped 20% this week.",
    "Microsoft announced a new AI partnership yesterday.",
    "Biden signed a climate change bill last month.",
    "Apple released a new iPhone model in September.",
    "Meta laid off 10,000 employees in 2024.",
    "Trump is going to deport Elon Musk!",
]

# -------------------------------------

if __name__ == "__main__":
    testing = False
    ke = KeywordExtractor()

    if testing:
        print("---" * 5 + "Testing" + "---" * 5)
        print("---" * 5 + "EASY" + "---" * 5)
        for claim in test_easy:
            kws = ke.extract_keywords(claim)
            search_query = " ".join(kws)
            print(f"Claim: {claim}")
            print(f"Search: {search_query}")
            print()


        print("---" * 5 + "Testing" + "---" * 5)
        print("---" * 5 + "HARD" + "---" * 5)
        p = Path(__file__).resolve().parent / '..' / 'claim_extractor' / '.data_sets' / 'Twitter.csv'
        p.resolve()
        assert p.exists()
        df =  pd.read_csv(p, encoding='utf-8')
        df = df.sample(n=25, replace=False, ignore_index=True)
        # print(df.head())

        for i, claim in df.iterrows():
            claim = claim['tweet_text']
            sents = ke.get_sentences(claim)
            for sent in sents:
                kws = ke.extract_keywords(sent)
                search_query = " ".join(kws)
                print(f"Claim: {sent}")
                print(f"Search: {search_query}")
    else:
        pr = PronounReplacer()
        tpp = MyTextPreProcessor()
        # What about "those" and "that"?
        
        test_sentences = [
            "Steven is a cool guy, he always shows up on time",
            "Apple is a greedy company, it is only after your money!",
            "Democrats hate Donald Trump, but they can't do anything about him.", # They must skip over DT and be Democrats
            "Donald Trump's Big Beautiful Bill is garbage, he doesn't care about anyone but himself.", # found the "'s" is attached
            "Donald Trump's Big Beautiful Bill is garbage, we mean nothing to him.",
            "Donald Trump's Big Beautiful Bill is garbage, his actions speak louder than his lies.",
            "The US bombs Iran, so they retaliate by launching missiles at military bases in Qatar, Afghanistan, Iraq, and Syria.", # Testing GPE
            "As per the article by Kevin Sullivan, the US bombed Iran so they retaliated by launching missiles at military bases in Qatar, Afghanistan, Iraq, and Syria." # Testing GPE extensive
        ]
        
        for sentence in test_sentences:
            print(f"Original:\n  {sentence}")
            
            result1 = pr.replace_pronouns_rule_based(sentence)
            print(f"Rule-based:\n  {result1}")
            
            result2 = pr.replace_pronouns_dependency_enhanced(sentence)
            print(f"Dependency-enhanced:\n  {result2}")

            kwds = tpp.create_search_queries(result1)
            print(f"Keywords: {kwds}")

            # This one performs slightly better currently
            kwds = ke.extract_keywords(result1)
            print(f"Keywords: {kwds}")

            
            ## Couldn't get working 
            # result3 = replacer.experimental_coreferencing(sentence)
            # that = [x for x in result3]
            # print(f"Experimental: {that}")
            
            print("-" * 80)