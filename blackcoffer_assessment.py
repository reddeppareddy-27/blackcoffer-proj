import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
from nltk.tokenize import word_tokenize, sent_tokenize

# --- Configuration and File Paths ---
INPUT_FILE = 'Input.xlsx - Sheet1.csv'
OUTPUT_FILE = 'Output.csv'
STOP_WORDS_DIR = 'StopWords'  # Directory containing StopWords files
MASTER_DICT_DIR = 'MasterDictionary' # Directory containing MasterDictionary files
EXTRACTED_ARTICLES_DIR = 'ExtractedArticles' # Directory to save extracted text files

# Create necessary directories
os.makedirs(EXTRACTED_ARTICLES_DIR, exist_ok=True)

# --- 1. Load Data and Dictionaries ---

def load_stop_words(stop_words_dir):
    """Loads all stop words from the files in the specified directory."""
    stop_words = set()
    for filename in os.listdir(stop_words_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(stop_words_dir, filename)
            with open(filepath, 'r', encoding='latin-1') as f:
                # Assuming stop words are separated by newlines or pipes (|)
                content = f.read().split()
                for word in content:
                    # Clean the word, removing potential pipes/separators
                    cleaned_word = re.sub(r'\|', '', word).strip()
                    if cleaned_word:
                        stop_words.add(cleaned_word.lower())
    return stop_words

def load_master_dictionary(master_dict_dir, stop_words):
    """Loads positive and negative words, excluding stop words."""
    positive_words = set()
    negative_words = set()

    # NOTE: MasterDictionary folder usually contains 'positive-words.txt' and 'negative-words.txt'
    # The files below are assumed based on standard NLP practice for this type of assignment.

    # Load Positive Words
    pos_path = os.path.join(master_dict_dir, 'positive-words.txt')
    if os.path.exists(pos_path):
        with open(pos_path, 'r', encoding='latin-1') as f:
            for line in f:
                word = line.strip().lower()
                if word and word not in stop_words:
                    positive_words.add(word)

    # Load Negative Words
    neg_path = os.path.join(master_dict_dir, 'negative-words.txt')
    if os.path.exists(neg_path):
        with open(neg_path, 'r', encoding='latin-1') as f:
            for line in f:
                word = line.strip().lower()
                if word and word not in stop_words:
                    negative_words.add(word)

    return positive_words, negative_words

# Pre-load dictionaries and stop words (These files must exist in the specified directories)
STOP_WORDS = load_stop_words(STOP_WORDS_DIR)
POSITIVE_DICT, NEGATIVE_DICT = load_master_dictionary(MASTER_DICT_DIR, STOP_WORDS)

# --- 2. Data Extraction (Web Scraper) ---

def extract_article_text(url):
    """
    Extracts the article title and content from a given URL.
    Attempts to target common article body elements.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Extract Title
        title_tag = soup.find('h1', class_='entry-title') or soup.find('title')
        title = title_tag.text.strip() if title_tag else ""

        # 2. Extract Article Text (Common selectors for Blackcoffer Insights articles)
        # Targeting the main content area where the article text resides
        content_element = soup.find('div', class_='td-post-content tagdiv-type')
        if not content_element:
            content_element = soup.find('div', class_='td-pb-padding-side')

        article_text = ""
        if content_element:
            # Extract text from p tags within the content element
            paragraphs = content_element.find_all('p')
            article_text = '\n'.join([p.text.strip() for p in paragraphs])
        
        # Combine title and body text
        full_text = f"{title}\n{article_text}"
        
        # Clean up any residual non-article text that might have been accidentally included
        # based on visual inspection of the target site's structure.
        return full_text

    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {url}: {e}")
        return None

def save_article(url_id, text):
    """Saves the extracted text to a file named after the URL_ID."""
    if text:
        filepath = os.path.join(EXTRACTED_ARTICLES_DIR, f"{url_id}.txt")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            return True
        except Exception as e:
            print(f"Error saving file {url_id}: {e}")
            return False
    return False

# --- 3. Text Analysis (NLP) Helper Functions ---

def clean_and_tokenize_words(text, stop_words):
    """
    Cleans the text by removing punctuation and stop words, then tokenizes it into words.
    Returns the list of cleaned words and the total raw word count (for AVG WORD LENGTH).
    """
    # 1. Tokenize into words, remove punctuation (using regex) and convert to lowercase
    raw_words = word_tokenize(text)
    
    # 2. Cleaned Words: remove punctuation and stop words [cite: 112, 113]
    cleaned_words = []
    # Total count of all words (for denominator in AVG WORD LENGTH, before cleaning)
    total_raw_words_for_avg_length = 0 
    
    for word in raw_words:
        # Remove punctuation [cite: 114]
        clean_word = re.sub(r'[^\w\s]', '', word).strip()
        
        if clean_word:
            total_raw_words_for_avg_length += 1
            if clean_word.lower() not in stop_words:
                cleaned_words.append(clean_word)
                
    # WORD COUNT is the count of *cleaned* words [cite: 112]
    word_count_final = len(cleaned_words)
    
    return cleaned_words, word_count_final, total_raw_words_for_avg_length

def calculate_readability(text, cleaned_words):
    """Calculates readability metrics (AVG SENTENCE LENGTH, FOG INDEX, etc.)"""
    
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    word_count = len(cleaned_words) # Using cleaned word count for Readability (as implied)
    
    if num_sentences == 0:
        return 0, 0, 0, 0, 0
    
    # 1. Complex Word Count (Words with > 2 syllables) [cite: 110]
    complex_word_count = 0
    total_syllables = 0
    
    for word in cleaned_words:
        syllable_count = calculate_syllable_count(word)
        total_syllables += syllable_count
        if syllable_count > 2:
            complex_word_count += 1
    
    # 2. AVG SENTENCE LENGTH [cite: 103]
    # NOTE: Using word_count (cleaned words) / num_sentences as per definition
    avg_sentence_length = word_count / num_sentences
    
    # 3. PERCENTAGE OF COMPLEX WORDS [cite: 104]
    percentage_complex_words = complex_word_count / word_count if word_count > 0 else 0
    
    # 4. FOG INDEX [cite: 105]
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    # 5. AVG NUMBER OF WORDS PER SENTENCE (same as AVG SENTENCE LENGTH) [cite: 108]
    avg_words_per_sentence = avg_sentence_length
    
    # 6. SYLLABLE PER WORD (Average) [cite: 50]
    syllable_per_word_avg = total_syllables / word_count if word_count > 0 else 0
    
    return avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, \
           complex_word_count, total_syllables, syllable_per_word_avg

def calculate_syllable_count(word):
    """
    Counts syllables based on vowels, handling 'es', 'ed' exceptions. [cite: 116, 117]
    A word is generally assumed to have at least 1 syllable if it contains a vowel.
    """
    word = word.lower()
    count = 0
    vowels = 'aeiou'
    
    # Handle 'es' and 'ed' exceptions: words ending with "es" or "ed" are not counted as a syllable 
    if word.endswith('es') or word.endswith('ed'):
        # Check if the remaining part is still a valid word part with vowels, 
        # but the rule says 'by not counting them as a syllable'
        # Simple implementation: subtract 1 from the total if it ends with 'es' or 'ed'
        pass
    else:
        # Standard vowel counting method
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
                
    # Adjust for the exceptions (subtract 1 from the final count if it ends with 'es' or 'ed')
    if word.endswith('es') and count > 1:
        count -= 1
    if word.endswith('ed') and count > 1:
        count -= 1

    return max(1, count) # Ensure a word has at least 1 syllable if it's not empty

def calculate_sentiment_scores(cleaned_words, pos_dict, neg_dict):
    """Calculates sentiment and subjectivity scores."""
    
    # Positive Score: +1 for each word in Positive Dictionary [cite: 90]
    positive_score = sum(1 for word in cleaned_words if word.lower() in pos_dict)
    
    # Negative Score: -1 for each word in Negative Dictionary, then multiplied by -1 [cite: 91, 92]
    negative_score_raw = sum(-1 for word in cleaned_words if word.lower() in neg_dict)
    negative_score = negative_score_raw * -1 
    
    word_count_clean = len(cleaned_words)
    
    # Polarity Score [cite: 95]
    denominator_pol = positive_score + negative_score + 0.000001
    polarity_score = (positive_score - negative_score) / denominator_pol
    
    # Subjectivity Score [cite: 99]
    denominator_sub = word_count_clean + 0.000001
    subjectivity_score = (positive_score + negative_score) / denominator_sub
    
    return positive_score, negative_score, polarity_score, subjectivity_score

def count_personal_pronouns(text):
    """
    Counts personal pronouns: I, we, my, ours, us, excluding 'US' (country). [cite: 119, 120]
    Using regex to find whole words.
    """
    # Regex for whole word match: I, we, my, ours, us (case-insensitive)
    pronoun_regex = r'\b(I|we|my|ours|us)\b'
    matches = re.findall(pronoun_regex, text, re.IGNORECASE)
    
    count = 0
    for match in matches:
        # Special check to exclude 'US' if it is ALL CAPS (to avoid country name) [cite: 120]
        # Although the regex \b(us)\b should prevent matching "plus", we'll check case for 'US'
        if match.upper() == 'US' and len(match) == 2:
            # If 'US' is found in all caps, skip it
            # The assignment specifies "Special care is taken so that the country name US is not included"
            # Since the text is case-insensitive, we'll assume a stricter rule: if the matched word is 'US' in all caps, it's excluded.
            continue
        count += 1
        
    return count

def calculate_avg_word_length(text, total_raw_words):
    """
    Calculates the average word length: Sum of characters in each word / Total number of words. [cite: 122, 123]
    NOTE: Using all words *before* cleaning (raw words) as implied by the definition.
    """
    if total_raw_words == 0:
        return 0
        
    # Get all words (tokens) including punctuation-bearing ones initially
    raw_tokens = word_tokenize(text)
    total_characters = 0
    actual_word_count = 0 # Count of tokens that are actual words (not just punctuation)

    for token in raw_tokens:
        # Clean the token: only keep letters and numbers (no punctuation)
        clean_token = re.sub(r'[^\w\s]', '', token).strip()
        
        if clean_token:
            total_characters += len(clean_token)
            actual_word_count += 1
    
    # Using the total number of words that were counted after removing stop words (for consistency with word_count_final)
    # The definition is ambiguous about whether "Total number of words" refers to raw words or cleaned words.
    # We will use the count of words that contain characters (i.e., not just punctuation tokens), which should align with the raw word count used in clean_and_tokenize_words.
    
    return total_characters / actual_word_count if actual_word_count > 0 else 0

# --- 4. Main Execution ---

def main_analysis():
    """Main function to run data extraction and analysis."""
    
    # Read Input data [cite: 11]
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    results = []

    for index, row in df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        
        print(f"Processing URL_ID: {url_id}")

        # --- Data Extraction ---
        article_text = extract_article_text(url)
        
        if article_text:
            save_article(url_id, article_text)
        
            # --- Text Analysis ---
            
            # 1. Clean and tokenize
            # total_raw_words_for_avg_length is a count of tokens that are actual words (not just punctuation)
            cleaned_words, word_count_final, total_raw_words_for_avg_length = clean_and_tokenize_words(article_text, STOP_WORDS)
            
            if word_count_final == 0:
                 # Handle case with no words
                analysis_results = {
                    'POSITIVE SCORE': 0, 'NEGATIVE SCORE': 0, 'POLARITY SCORE': 0, 
                    'SUBJECTIVITY SCORE': 0, 'AVG SENTENCE LENGTH': 0, 
                    'PERCENTAGE OF COMPLEX WORDS': 0, 'FOG INDEX': 0, 
                    'AVG NUMBER OF WORDS PER SENTENCE': 0, 'COMPLEX WORD COUNT': 0, 
                    'WORD COUNT': 0, 'SYLLABLE PER WORD': 0, 'PERSONAL PRONOUNS': 0, 
                    'AVG WORD LENGTH': 0
                }
            else:
                # 2. Sentimental Analysis
                pos_score, neg_score, polarity_score, subjectivity_score = calculate_sentiment_scores(
                    cleaned_words, POSITIVE_DICT, NEGATIVE_DICT
                )
                
                # 3. Readability Analysis
                avg_sen_len, perc_complex, fog_index, avg_words_per_sen, complex_count, total_syllables, avg_syllable_per_word = calculate_readability(
                    article_text, cleaned_words
                )
                
                # 4. Personal Pronouns
                personal_pronouns_count = count_personal_pronouns(article_text)
                
                # 5. Average Word Length
                avg_word_length = calculate_avg_word_length(article_text, total_raw_words_for_avg_length)

                analysis_results = {
                    'POSITIVE SCORE': pos_score,
                    'NEGATIVE SCORE': neg_score,
                    'POLARITY SCORE': polarity_score,
                    'SUBJECTIVITY SCORE': subjectivity_score,
                    'AVG SENTENCE LENGTH': avg_sen_len,
                    'PERCENTAGE OF COMPLEX WORDS': perc_complex,
                    'FOG INDEX': fog_index,
                    'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sen,
                    'COMPLEX WORD COUNT': complex_count,
                    'WORD COUNT': word_count_final,
                    'SYLLABLE PER WORD': avg_syllable_per_word, # The output structure asks for SYLLABLE PER WORD (average) [cite: 65]
                    'PERSONAL PRONOUNS': personal_pronouns_count,
                    'AVG WORD LENGTH': avg_word_length
                }
        else:
            # If extraction failed, fill with zeros
            print(f"Extraction failed for {url_id}. Filling with zeros.")
            analysis_results = {
                'POSITIVE SCORE': 0, 'NEGATIVE SCORE': 0, 'POLARITY SCORE': 0, 
                'SUBJECTIVITY SCORE': 0, 'AVG SENTENCE LENGTH': 0, 
                'PERCENTAGE OF COMPLEX WORDS': 0, 'FOG INDEX': 0, 
                'AVG NUMBER OF WORDS PER SENTENCE': 0, 'COMPLEX WORD COUNT': 0, 
                'WORD COUNT': 0, 'SYLLABLE PER WORD': 0, 'PERSONAL PRONOUNS': 0, 
                'AVG WORD LENGTH': 0
            }


        # Combine input variables with analysis results
        output_row = {
            'URL_ID': url_id, 
            'URL': url, 
            **analysis_results
        }
        results.append(output_row)
        
    # Create Output DataFrame and save to CSV
    output_df = pd.DataFrame(results)
    
    # Ensure columns are in the exact order as specified in the output structure 
    output_columns = [
        'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 
        'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 
        'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 
        'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
    ]
    
    output_df = output_df[output_columns]
    
    try:
        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Analysis complete. Results saved to '{OUTPUT_FILE}'.")
    except Exception as e:
        print(f"\nError saving output file: {e}")

if __name__ == '__main__':
    main_analysis()