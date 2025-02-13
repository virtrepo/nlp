


































































































# practical 1 - preprocessing methods for text data (tokenization and stop words)
# text_preprocessing packages
import nltk
nltk.download('punkt_tab', force=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Function for Tokenization
def tokenize(text):
    """Tokenizes the input text into words."""
    return word_tokenize(text)
# Function for Stop-words Removal
def remove_stopwords(tokens):
    """Removes stop words from the list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]
# Function for Text Normalization
def normalize(text):
    """Normalizes the text by converting it to lowercase and removing punctuation."""
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text
# Pre-processing Pipeline Function
def preprocess_text(text):
    """Combines all text preprocessing steps."""
    normalized_text = normalize(text) # this is goa it has beaches
    tokens = tokenize(normalized_text) # this, is,  goa, it, has, beaches
    #print(tokens)
    filtered_tokens = remove_stopwords(tokens) #goa beaches
    return ' '.join(filtered_tokens)  # Return as a single string
# preprocess_text("This is Goa . It has beaches")

input_file_path = '/content/spam.csv'
output_file_path = 'processed_output.txt'

with open(input_file_path, 'r', encoding='utf-8', errors="replace") as file:
    lines = file.readlines()

# Process each line and store results
processed_lines = [preprocess_text(line.strip()) for line in lines]

# Save the processed lines to a new text file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for line in processed_lines:
        file.write(line + '\n')

print(f"Text preprocessing completed and saved to '{output_file_path}'.")

# Print the first 5 lines of the input file
print("Provided Input (First 5 Lines):")
with open(input_file_path, 'r', encoding='ISO-8859-1') as file:
    for _ in range(5):
        line = file.readline()
        if line:  # Check if there's a line to read
            print(line.strip())
        else:
            break  # Exit if there are fewer than 5 lines

# Print the first 5 lines of the output file
print("Processed Output (First 5 Lines):")
with open('/content/processed_output.txt', 'r', encoding='utf-8') as file:
    for _ in range(5):
        line = file.readline()
        if line:  # Check if there's a line to read
            print(line.strip())
        else:
            break  # Exit if there are fewer than 5 lines


# practical 2 - Implementing Stemming in Text Pr-processing for NLP
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the dataset
data = {'messages': [
    "The cats are playing in the garden.",
    "He is running quickly to catch the bus.",
    "The boys are enjoying their game.",
    "She was reading a book.",
    "I love to eat apples and bananas."
]}
df = pd.DataFrame(data)

# Initialize the stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Stop-words Removal
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Text Normalization (Lowercasing and Removing punctuation)
    tokens = [word.lower() for word in tokens if word.isalnum()]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Apply the pre-processing pipeline
df['processed_messages'] = df['messages'].apply(preprocess_text)

# Display the result
print(df[['messages', 'processed_messages']])
print("\n \n")

def another_stem(val):
    """Applies stemming to a given text using PorterStemmer."""
    stemmer = PorterStemmer()
    tokens = word_tokenize(val)  # Tokenize input text
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(stemmed_tokens)  # Return stemmed text

# Example usage
sample_text = "The running cats are playing in the garden."
print(another_stem(sample_text))

# Install and import necessary packages
# !pip install nltk pandas

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize different stemmers
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

# Function for stemming
def another_stem(text, method="porter"):
    """Applies stemming using the specified method: 'porter', 'lancaster', or 'snowball'."""
    if method == "porter":
        stemmer = porter
    elif method == "lancaster":
        stemmer = lancaster
    elif method == "snowball":
        stemmer = snowball
    else:
        raise ValueError("Invalid stemming method. Choose 'porter', 'lancaster', or 'snowball'.")

    return ' '.join([stemmer.stem(word) for word in text.split()])

# Function for Tokenization
def tokenize(text):
    """Tokenizes the input text into words."""
    return word_tokenize(text)

# Function for Stop-words Removal
def remove_stopwords(tokens):
    """Removes stop words from the list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

# Function for Text Normalization
def normalize(text):
    """Normalizes the text by converting it to lowercase and removing punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Pre-processing Pipeline Function with Stemming
def preprocess_text(text, stem_method="porter"):
    """Combines all text preprocessing steps and applies stemming."""
    normalized_text = normalize(text)
    tokens = tokenize(normalized_text)
    filtered_tokens = remove_stopwords(tokens)
    stemmed_text = another_stem(' '.join(filtered_tokens), method=stem_method)
    return stemmed_text

# Load dataset (spam.csv)
input_file_path = '/content/spam.csv'
output_file_path = '/content/processed_output.txt'

df = pd.read_csv(input_file_path, encoding="ISO-8859-1")

# Assuming the text data is in the first column
df['processed_text'] = df.iloc[:, 0].apply(lambda text: preprocess_text(str(text), stem_method="porter"))

# Save processed text
df[['processed_text']].to_csv(output_file_path, index=False, header=False)

print(f"Text preprocessing completed and saved to '{output_file_path}'.")


# practical 3 - implementing morphology analysis in text pre-processing for NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def morphological_analysis(text):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Tokenization
    tokens = word_tokenize(text)

    # Stop-words removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Text normalization
    normalized_tokens = [word.lower().strip(string.punctuation) for word in filtered_tokens]

    # Morphological analysis using lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos='n') for word in normalized_tokens]

    return lemmatized_tokens

# Example usage
text_sample = "The cats are playing with the ball and they are enjoying happiness"
processed_text = morphological_analysis(text_sample)
print(processed_text)


import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def spacy_morphological_analysis(word):
    # Process the word
    doc = nlp(word)
    analyzed_forms = [(token.text, token.lemma_) for token in doc]
    return analyzed_forms

# Example usage
word_sample = "playing"
analyzed_word = spacy_morphological_analysis(word_sample)
print(analyzed_word)


# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Sample for a morphological parser function
# hello
#h=0, e=1,l=2, o = 4
#hello : o = -1, l = -2, h = -5
def morphological_parsing(word):
    morphemes = []
    if word.startswith('un') and len(word) > 8:  # Checks if it starts with 'un' and is long enough
        morphemes.append('un-')  # Add the prefix
        base_word = word[2:-4]  # Extract the root (removing 'un' and 'ness')
        morphemes.append(base_word)  # Add the root
        morphemes.append('-ness')  # Add the suffix
    elif word.endswith('ness') and len(word) > 4:  # Checks if it ends with 'ness' and is long enough
        base_word = word[:-4]  # Remove 'ness'
        morphemes.append(base_word)  # Add the root
        morphemes.append('-ness')  # Add the suffix
    elif word.endswith('ing') and len(word) > 3:
        morphemes.append(word[:-3])  # Remove 'ing'
        morphemes.append('ing')
    elif word.endswith('ed') and len(word) > 2:
        morphemes.append(word[:-2])  # Remove 'ed'
        morphemes.append('ed')
    else:
        morphemes.append(word)  # Return the word as is
    return morphemes

# Placeholder for Finite State Transducer (FST) example
def finite_state_transducer(word):
    if word.endswith('s'):
        return word[:-1]  # Remove plural 's'
    return word

def morphological_analysis(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Stop-words removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Text normalization
    normalized_tokens = [word.lower().strip(string.punctuation) for word in filtered_tokens]

    # Morphological analysis using lemmatization and other methods
    processed_tokens = []
    for word in normalized_tokens:
        # Apply finite state transducer
        base_form = finite_state_transducer(word)
        processed_tokens.append(base_form)  # Add the base form to the processed tokens
        # Apply morphological parsing
        morphemes = morphological_parsing(word)
        processed_tokens.extend(morphemes)  # Add morphemes to the processed tokens

    return list(set(processed_tokens))  # Return unique processed tokens

# Example usage
text_sample = "The cats are playing with the ball and they are enjoying No unhappiness reported"
processed_text = morphological_analysis(text_sample)
print(processed_text)


# practical - 4 Implementing N-gram model
import nltk
from collections import defaultdict, Counter

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

class UnigramModel:
    def __init__(self):
        self.word_freq = Counter()

    def preprocess(self, text):
        # Tokenization and cleaning
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]  # Removing punctuation
        return tokens

    def fit(self, text):
        tokens = self.preprocess(text)
        # Count word frequencies
        self.word_freq.update(tokens)

        # Debug: Print word frequencies
        print("Word frequencies:")
        for word, freq in self.word_freq.items():
            print(f"{word}: {freq}")

    def predict(self, word):
        # Return the top words based on frequency
        most_common = self.word_freq.most_common(3)
        return most_common

# Example usage
if __name__ == "__main__":
    # Sample text corpus
    corpus = """
    Natural language processing (NLP) is a subfield of artificial intelligence (AI).
    It enables computers to understand, interpret, and generate human language.
    With advances in machine learning and deep learning, NLP has made significant strides.
    Applications include sentiment analysis, machine translation, and chatbot development.
    """

    # Create and fit unigram model
    unigram_model = UnigramModel()
    unigram_model.fit(corpus)

    # Make predictions
    print("Top 3 predicted words based on frequency:")
    predictions = unigram_model.predict('any_word')
    print(predictions)  # Should show the top 3 words based on their frequencies


class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams_freq = defaultdict(Counter)

    def preprocess(self, text):
        # Tokenization and cleaning
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]  # Removing punctuation
        return tokens

    def fit_bigram(self, text):
        tokens = self.preprocess(text)
        # Generate n-grams and count frequencies
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            self.ngrams_freq[ngram[:-1]][ngram[-1]] += 1  # Count the frequency of the last word

            # Debug: Print the generated ngram
            print(f"Generated bigram: {ngram[:-1]} -> {ngram[-1]}")

    def predict_bigram(self, prefix):
        prefix = self.preprocess(prefix)
        if len(prefix) < self.n - 1:
            return []  # Not enough words to form a prediction
        prefix_tuple = tuple(prefix[-(self.n - 1):])  # Get the last (n-1) words
        # Debug: Print the prefix tuple being checked
        print(f"Checking prefix: {prefix_tuple}")

        if prefix_tuple in self.ngrams_freq:
            next_words = self.ngrams_freq[prefix_tuple]
            return next_words.most_common(3)  # Return top 3 predictions
        else:
            return []

# Example usage
if __name__ == "__main__":
    # Sample text corpus
    corpus = """
    Natural language processing (NLP) is a subfield of artificial intelligence (AI).
    It enables computers to understand, interpret, and generate human language.
    With advances in machine learning and deep learning, NLP has made significant strides.
    Applications include sentiment analysis, machine translation, and chatbot development.
    """

    # # Create and fit bigram model
    # bigram_model = NGramModel(n=2)
    # bigram_model.fit_bigram(corpus)
    # Create and fit bigram model
    bigram_model = NGramModel(n=3)
    bigram_model.fit_bigram(corpus)

    # Make predictions
    print("Bigram Predictions for 'natural language':")
    predictions = bigram_model.predict_bigram('natural language')
    print(predictions)  # Should only include 'processing'

    # print("Bigram Predictions for 'I live in India':")
    # predictions = bigram_model.predict_bigram('I live in India')
    # print(predictions)


# practical - 5 Part of speech tagging
import nltk
# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
def preprocess(text):
    """Tokenizes the input text into words."""
    tokens = nltk.word_tokenize(text)
    return tokens
def pos_tagging(text):
    """Performs Part of Speech tagging on the tokenized text."""
    tokens = preprocess(text)
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens
if __name__ == "__main__":
    # Sample text corpus
    text = """
    Natural language processing (NLP) is a fascinating field of artificial intelligence.
    It allows computers to understand and interpret human language.
    """
    # Perform POS tagging
    tagged_output = pos_tagging(text)
    # Display the tagged tokens
    print("Tagged Tokens:")
    for token, tag in tagged_output:
        print(f"{token}: {tag}")


# Practical - 6 text chunking for nlp
def chunk_list(data, chunk_size):
    """Yield successive chunk_size-sized chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Example usage:
data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
chunks = list(chunk_list(data, 3))
print(chunks)


def chunk_text(text, chunk_size):
    """Break text into chunks of specified character length."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

# Example usage: "cha , rac, ter"
text = "This is an example of chunking text into smaller pieces."
chunks = list(chunk_text(text, 10))
print(chunks)


def read_file_in_chunks(file_path, chunk_size=1024):
    """Read a file in chunks."""
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

for chunk in read_file_in_chunks('/content/spam.csv'):
    print(chunk)


# practical - 7 Text Summerization
# !pip install sumy

import nltk

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('punkt_tab')

# Import necessary libraries
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def get_text_input():
    """Function to get long text input from the user."""
    print("Please enter the text you want to summarize (end with a blank line):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

def summarize_text(text, sentence_count=3):
    """Function to summarize the text using Sumy."""
    # "this is a string" 'string valid'
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return ' '.join(str(sentence) for sentence in summary)
    except Exception as e:
        return f"Error in summarization: {str(e)}"

def main():
    """Main function to run the text summarization."""
    long_text = get_text_input()

    # Create a summarized.txt file and write original and summarized text to it
    with open('extractivesummarized.txt', 'w') as file:
        file.write("Original Text:\n")
        file.write(long_text + "\n\n")

        summary = summarize_text(long_text)
        file.write("Summarized Text:\n")
        file.write(summary)

    print("The original and summarized texts have been written to summarized.txt.")

if __name__ == "__main__":
    main()
# Global warming is the rise in Earth's average temperature caused by human activities, primarily the release of greenhouse gases like carbon dioxide. This phenomenon leads to severe environmental consequences, including rising sea levels, extreme weather events, and ecosystem disruptions. As temperatures increase, biodiversity is threatened, and agricultural systems face significant challenges. Immediate action is essential to reduce emissions and shift towards renewable energy sources. Addressing global warming is vital for a sustainable future and the well-being of future generations


# pip install transformers torch


from transformers import pipeline

# Ensure NLTK resources are available
nltk.download('punkt')

def get_text_input():
    """Function to get long text input from the user."""
    print("Please enter the text you want to summarize (end with a blank line):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

def summarize_text(text):
    """Function to summarize the text using Hugging Face Transformers."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    input_length = len(text.split())  # Approximate token count using words

    # Set max_length dynamically (e.g., half of input length)
    max_length = min(150, max(40, input_length // 2))
    summary = summarizer(text, max_length=max_length, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def main():
    """Main function to run the text summarization."""
    long_text = get_text_input()

    # Create a summarized.txt file and write original and summarized text to it
    with open('abstractivesummarized.txt', 'w') as file:
        file.write("Original Text:\n")
        file.write(long_text + "\n\n")

        summary = summarize_text(long_text)
        file.write("Summarized Text:\n")
        file.write(summary)

    print("The original and summarized texts have been written to summarized.txt.")

if __name__ == "__main__":
    main()


# practical - 8 name entity recognition
# pip install flair


from flair.models import SequenceTagger
from flair.data import Sentence

# I am Abhinav I live in Goa
# Abhinav = Person, Goa Place/Location
# Apple Inc. announced a new product launch in San Francisco

def get_user_input():
    # Step 2: Text Input
    user_input = input("Please enter a text: ")
    return user_input

def named_entity_recognition(text):
    # Step 3: NER Implementation
    # Load the pre-trained NER model
    tagger = SequenceTagger.load("ner")

    # Create a Sentence object
    sentence = Sentence(text)

    # Predict entities
    tagger.predict(sentence)

    return sentence

def print_named_entities(sentence):
    # Step 4: User-Friendly Output
    print("\nNamed Entities:")
    for entity in sentence.get_spans('ner'):
        print(f"- {entity.text} ({entity.tag})")

# Main execution
if __name__ == "__main__":
    text = get_user_input()  # Get text input from user
    sentence = named_entity_recognition(text)  # Perform NER
    print_named_entities(sentence)  # Print results
#Barack Obama was the 44th President of the United States.
#The Eiffel Tower is one of the most famous landmarks in Paris, France.
#Apple Inc. announced a new product launch in San Francisco.

# ORG ORGANISATION
# LOC LOCATION


# practical - 9 sentiment analysis
# pip install pandas nltk scikit-learn
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Download NLTK data files
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Load the dataset
df = pd.read_csv('/content/TestLarge.csv')

# Data Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and non-alphabetic words
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
df['Processed_Review'] = df['Review'].apply(preprocess_text)

# Features and labels
X = df['Processed_Review']
y = df['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predictions
y_pred = model.predict(X_test_vectorized)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Positive', average='binary')
recall = recall_score(y_test, y_pred, pos_label='Positive', average='binary')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Sample predictions
sample_reviews = [
    "This product is amazing!",
    "I'm really disappointed with my purchase.",
    "It does exactly what I needed.",
    "Not worth the price.",
    "I would buy this again in a heartbeat!"
]

# Preprocess and vectorize sample reviews
sample_reviews_processed = [preprocess_text(review) for review in sample_reviews]
sample_reviews_vectorized = vectorizer.transform(sample_reviews_processed)

# Predict sentiment
sample_predictions = model.predict(sample_reviews_vectorized)

# Display predictions
for review, sentiment in zip(sample_reviews, sample_predictions):
    print(f'Review: "{review}" - Predicted Sentiment: {sentiment}')


# practical - 10 real time application NLP application
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import string
import nltk

# Download stopwords from NLTK
nltk.download('stopwords')


# Load dataset
dataset = pd.read_csv('/content/emails.csv')

# Clean data: Remove punctuation, stopwords, and tokenize
def process(text):
    clean_text = ''.join([char for char in text if char not in string.punctuation])
    return [word for word in clean_text.split() if word.lower() not in stopwords.words('english')]

# Apply CountVectorizer
vectorizer = CountVectorizer(analyzer=process)
X = vectorizer.fit_transform(dataset['text'])
y = dataset['spam']

# Save the vectorizer
pickle.dump(vectorizer, open("/content/vectorizer.pkl", "wb"))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("/content/model.pkl", "wb"))

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(dpi=100)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# Load the saved vectorizer and model
vectorizer = pickle.load(open("/content/vectorizer.pkl", "rb"))
model = pickle.load(open("/content/model.pkl", "rb"))


new_email = "Congratulations! You've won a free gift card. Claim it now!"
processed_email = vectorizer.transform([new_email])  # Apply the same vectorizer


prediction = model.predict(processed_email)
if prediction == 1:
    print("This email is SPAM")
else:
    print("This email is NOT SPAM (Not-SPAM)")
