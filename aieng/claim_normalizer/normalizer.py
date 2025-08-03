# 1st party
import re
from collections import Counter
from typing import List

# 3rd party
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import spacy

# https://github.com/cbaziotis/ekphrasis
# website includes a spell checker too!

# Mine
# ...

class MyTextPreProcessor():
    def __init__(self):
        self.text_preprocessor = self._create_text_processor()
        self.nlp = spacy.load("en_core_web_sm")
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
        
        # Words that are almost never useful for news search
        # self.blacklist = {
        #     'said', 'says', 'according', 'reported', 'statement', 'wednesday', 
        #     'thursday', 'friday', 'people', 'group', 'company', 'officials',
        #     'sources', 'news', 'report', 'article', 'story', 'information'
        # }

        # Comprehensive blacklist of words that hurt news search
        self.blacklist = {
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
    
    def extract_keywords(self, text):
        doc = self.nlp(text)
        
        # Get named entities first (highest priority)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT"]:
                clean_ent = ent.text.strip()
                if len(clean_ent) > 1 and clean_ent.lower() not in self.blacklist:
                    entities.append(clean_ent)
        
        # Get important nouns/adjectives as backup
        backup_words = []
        for token in doc:
            if (token.pos_ in ["NOUN", "ADJ"] and 
                len(token.text) > 3 and 
                not token.is_stop and 
                token.text.lower() not in self.blacklist and
                token.is_alpha):
                backup_words.append(token.lemma_.lower())
        
        # Prioritize entities, use backup if needed
        that = [entities, backup_words]
        keywords = [thing for this in that for thing in this]
        # keywords = entities[:4] if entities else backup_words[:3]
        
        return keywords


## ----------------------------------------------------
## ----------------------------------------------------
## ----------------------------------------------------

def extract_search_keywords(claim, max_keywords=5):
    """Extract key terms for NewsAPI search"""
    
    # Remove common stop words and short words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'that', 'this', 'it', 'they', 'them', 'their'
    }
    
    # Clean and split
    words = re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower())
    keywords = [w for w in words if w not in stop_words]
    
    # Simple logic for getting important terms
    # TODO: Improve later!
    word_counts = Counter(keywords)
    top_keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return top_keywords



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
    print("---" * 5 + "Testing" + "---" * 5)
    print("---" * 5 + "EASY" + "---" * 5)
    for claim in test_easy:
        keywords = extract_search_keywords(claim)
        search_query = " ".join(keywords)
        print(f"Claim: {claim}")
        print(f"Search: {search_query}")
        print()