import pandas as pd
import re

def get_train_test(train_input, test_input):
    with open(train_input) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines] 

    subject_regex = '\<subject\>(.+)\<\/subject\>'
    content_regex = '\<content\>(.+)\<\/content\>'
    maincat_regex = '\<maincat\>(.+)\<\/maincat\>'

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

    train = pd.DataFrame.from_dict({'subject': subjects, 'content': contents, 'category': maincats})

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

    test = pd.DataFrame.from_dict({'subject': subjects, 'content': contents})
    return train, test