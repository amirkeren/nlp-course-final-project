import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def remove_non_letters(data):
    regex = re.compile('[^a-zA-Z ]')
    data['subject'] = [regex.sub(' ', x) for x in data['subject']]
    data['content'] = [regex.sub(' ', x) for x in data['content']]

def remove_stopwords(data):
    stopwords_set = set(stopwords.words('english'))
    data['subject'] = [[y if y not in stopwords_set else '' for y in x] for x in data['subject']]
    data['subject'] = [filter(None, x) for x in data['subject']]
    data['content'] = [[y if y not in stopwords_set else '' for y in x] for x in data['content']]
    data['content'] = [filter(None, x) for x in data['content']]

def stemming(data):
    ps = PorterStemmer()
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