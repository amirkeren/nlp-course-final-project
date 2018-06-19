import re
import string
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pattern.en import suggest

stopwords_set = set(stopwords.words('english'))
letters_regex = re.compile('[^a-zA-Z ]')
length_regex = re.compile(r"(.)\1{2,}")
ps = PorterStemmer()

def load_preprocessed_data(train_path, test_path):
    return pickle.load(open(train_path, 'rb')), pickle.load(open(test_path, 'rb'))

def save_preprocessed_data(data, path):
    pickle.dump(data, open(path, 'wb'))

def subsampling(data, max_lines=100):
    data['subject'] = data['subject'][:max_lines]
    data['content'] = data['content'][:max_lines]
    if 'category' in data:
        data['category'] = data['category'][:max_lines]

def correct_spelling(data):
    data['subject'] = [[_suggest_aux(y) for y in x] for x in data['subject']]
    data['content'] = [[_suggest_aux(y) for y in x] for x in data['content']]

def reduce_lengthening(data):
    data['subject'] = [[length_regex.sub(r"\1\1", y) for y in x] for x in data['subject']]
    data['content'] = [[length_regex.sub(r"\1\1", y) for y in x] for x in data['content']]

def remove_non_letters(data):
    data['subject'] = [letters_regex.sub(' ', x) for x in data['subject']]
    data['content'] = [letters_regex.sub(' ', x) for x in data['content']]

def remove_stopwords(data):
    data['subject'] = [[y if y not in stopwords_set else '' for y in x] for x in data['subject']]
    data['subject'] = [filter(None, x) for x in data['subject']]
    data['content'] = [[y if y not in stopwords_set else '' for y in x] for x in data['content']]
    data['content'] = [filter(None, x) for x in data['content']]

def stemming(data):
    data['subject'] = [[str(ps.stem(y)) for y in x] for x in data['subject']]
    data['content'] = [[str(ps.stem(y)) for y in x] for x in data['content']]

def remove_punctuation(data):
    data['subject'] = [x.translate(None, string.punctuation) for x in data['subject']]
    data['content'] = [x.translate(None, string.punctuation) for x in data['content']]

def tokenize(data):
    data['subject'] = [word_tokenize(x) for x in data['subject']]
    data['content'] = [word_tokenize(x) for x in data['content']]

def lowercase(data):
    data['subject'] = [x.lower() for x in data['subject']]
    data['content'] = [x.lower() for x in data['content']]

def get_train_test(train_input, test_input):
    subject_regex = '\<subject\>(.+)\<\/subject\>'
    content_regex = '\<content\>(.+)\<\/content\>'
    maincat_regex = '\<maincat\>(.+)\<\/maincat\>'
    with open(train_input) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines] 
    subjects = []
    contents = []
    maincats = []
    for line in lines:
        m = re.search(subject_regex, line)
        if m:
            subjects.append(m.group(1))
        else:
            subjects.append('')
        m = re.search(content_regex, line)
        if m:
            contents.append(m.group(1))
        else:
            contents.append('')
        m = re.search(maincat_regex, line)
        if m:
            maincats.append(m.group(1))
        else:
            maincats.append('')
    train = {'subject': subjects, 'content': contents, 'category': maincats}
    with open(test_input) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines] 
    subjects = []
    contents = []
    for line in lines:
        m = re.search(subject_regex, line)
        if m:
            subjects.append(m.group(1))
        else:
            subjects.append('')
        m = re.search(content_regex, line)
        if m:
            contents.append(m.group(1))
        else:
            contents.append('')
        m = re.search(maincat_regex, line)
    test = {'subject': subjects, 'content': contents}
    return train, test

def _suggest_aux(text, threshold=0.9):
    suggestions = suggest(text)
    if suggestions[0][1] > threshold:
        return suggestions[0][0]
    return text