#!/usr/bin/env python

import re
import numpy as np

import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

from collections import OrderedDict as OD

import uuid
from IPython.display import display_javascript, display_html, display
import json

'''A collection of classes and functions used for parsing resumes.'''


class Classifier:
    '''A Class of text classifers.'''
	
    def __init__(self, bags, cats, model):
    	'''Train a classifier on the data provided.
    	
    	INPUTS
    	bags (list):  training data in the form [[bag_1], [bag_2], ...] where each bag_i 
    	contains a list of strings bag_i = ['some string', 'another string'].
    	
    	cats (list): Categorical labels of each bag.  cats = ['CAT1', 'CAT2', ...].  The 
    	length of cats should match the length of bags.
    	
    	model (class):  Model to fit to the data.  e.g. model = MultinomialNB().
    	
    	'''
        
        # reshape data for training
        y_tr = flatten([[n] * len(bags[n]) for n in range(0,len(bags))])
        X_tr = flatten(bags)
        
        self.cats = cats
        
        # convert bags into tf_idf vectors
        self.count_vect = CountVectorizer()
        self.bow_matrix = self.count_vect.fit_transform(X_tr)
        self.tfidf_transformer = TfidfTransformer()
        self.normalized_matrix = self.tfidf_transformer.fit_transform(self.bow_matrix)
        
        # train model
        self.model = model.fit(self.normalized_matrix, y_tr)
    
    def text_class(self, text):
    	'''Use self.model to classify text.
    	
    	INPUTS
		text (list): a list of strings to be classified.  
    	
    	OUTPUT
    	predicted (tuple or array): 
    	
    	if len(list) == 1 the output is a tuple of the form 
    	(PREDICTED_CLASS, [probabilities]).  
    	
    	if len(list) > 1 the output is an array of the form 
    	[[probabilites_1], [probabilites_2], ...]
    	
    	where the elements of [probabilities] are the probabilities that the text belongs
    	to one of the classes in self.cats  
    	
    	'''
    	
    	# convert text to tf_idf vectors
        text = map(lambda x: unicode(x,"utf-8"),text)
        X_new_counts = self.count_vect.transform(text)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        
        # classify text
        if X_new_counts.shape[0] == 1 and np.sum(X_new_counts) == 0:
            predicted = ('UNKN', np.array([0,0,0,0]))
        elif X_new_counts.shape[0] == 1:
            pred = self.model.predict_proba(X_new_tfidf)
            predicted = (self.cats[np.argmax(pred)], pred)
        else:
            predicted = self.model.predict_proba(X_new_tfidf)
            
        return predicted
    
    def score_report(self, bags):
    	
    	'''
    	Test performance of self.text_class.
    	
    	INPUTS
    	bags (list of strings):  test strings to be classified in the form 
    	[[bag_1], [bag_2], ...]
    	
    	OUTPUTS
    	returns classification_report for test data
    		
    	'''
    	
    	# generate y_true
        y_te = flatten([[n] * len(bags[n]) for n in range(0,len(bags))])
        
        return classification_report(y_te, map(np.argmax, self.text_class(flatten(bags))))


def find_date(string):
    '''Look for *date intervals* based a few different regex patterns
    
    INPUTS
    string (string):  a string that potentially contains date intervals
    
    OUTPUTS
    matches (tuple):  ((x0, x1), 'matching string')  where x0 and x1 are the start and end
    positions in string of the matching string.
    
    '''
    
    # a bunch of ways that people write months 
    dates_string = '''(Jan|Jan.|January|Feb|Feb.|February|Mar|Mar.|March|Apr|Apr.|April|\
May|June|Jun|Jun.|Jul|Jul.|July|Aug|Aug.|August|Sept|Sept.|Sep|Sep.|September|Oct|Oct.|\
October|Nov|Nov.|November|Dec|Dec.|December)'''
        
    # regex patterns for date intervals    
    date_regex = '('+dates_string+'(.{0,2}|\s{,2})\d{4}.{0,5}'+dates_string+'(.{0,2}|\s\
{0,2})\d{4}'+')|('+ dates_string + '.{0,2}\d{4}.{0,5}(Current|current|Present|present)'\
+')|('+ '\d{4}.{0,5}\d{4}' +')|('+ '\d{2}/(\d{4}|\d{2}).{0,5}\d{2}/(\d{4}|\d{2})' +')|(\
'+ '(\d{2}/\d{4}|\d{4}).{0,5}(present|Present|Current|current)' +')|('+ '(20|19)\d{2}'+\
')'
    
    matches = []
    for match in re.finditer(date_regex, string, re.IGNORECASE):
        matches.append((match.span(), match.group()))
    return matches


def kill_space(list):
    '''Get rid of anything in [ '', [], [''] ]'''
    new_list = []
    for ele in list:
        if ele != '' and ele != [] and ele != ['']:
            new_list.append(ele)
    return new_list  


def flatten(list_of_lists):
    '''Flatten a list of lists.'''
    flattened_list = []
    for x in list_of_lists:
        for y in x:
            flattened_list.append(y)
    return flattened_list


def sup_info(text):
    '''
	Extract unwanted information from text.
	
	INPUTS
	text (string):  text to be cleaned
	
	OUTPUTS:
	to_remove (list of strings):  parts of text to remove
	
	'''
    
    # Break up text
    f5l = ' '.join(re.split('\n',text)[0:1])
    
    # Look for email address and phone number
    to_remove = re.findall(r'[\w\.-]+@[\w\.-]+|\d{3}[\.-]\d{3}[\.-]\d{4}', text)
    
    # Look for name
    nlp = spacy.load('en_core_web_sm')
    doc = nlp( unicode(f5l, "utf-8"))
    for token in doc:
        if (token.pos_ == u'PROPN')|(token.pos_ == u'NOUN'):
            name = token.text.encode("utf-8", 'ignore')
            to_remove.append(name)
                
    return to_remove  


def extract_lines(resume):
    
    '''
    Breaks the resume string into lines and removes data from sup_info
    
    INPUTS
    resume (string):  full resume text as output from pdf_to_text
    
    OUTPUTS
    resume_lines_raw (list of lists):  [['line 1 string'],['line 2 string'],...]
    
    '''
    
    # remove sup_info
    new_res = resume
    for rx in sup_info(resume):
        new_res = new_res.replace(rx, '')
    
    # full text ==> list containing text lines ==> sublists containing groups of text 
    resume_lines_raw = map(lambda x: re.split(r'\s{5,}', x), kill_space(re.split('\n',new_res)))
    return resume_lines_raw


def clean_resume(resume):
    '''
    Some basic cleaning of resume lines.
    
    INPUTS
    resume (string):  full resume text as output from pdf_to_text
    
    OUTPUTS
    word_bags_cleaned (list of strings):  
    ['cleaned resume line 1', 'cleaned resume line 2', ...]
    
    '''
    
    # extract out Name/Phone/Email
    new_res = resume[:]
    for rx in sup_info(resume):
        new_res = new_res.replace(rx, '')

    # throw away lines that do not contain any letters or numbers
    new_lines = []
    for line in flatten(extract_lines(new_res)):
        if re.search('[a-zA-Z0-9]', line):
            new_lines.append(line)
    new_lines = new_lines[2:]

    # join lines starting with lower case to previous lines
    word_bags = []
    for line in new_lines:
        if not str.islower(line[0]):
            word_bags.append(line)
        elif len(word_bags)>0:
            word_bags[-1] = word_bags[-1] + line
   
    # isolate and tag dates
    word_bags_cleaned = []
    for string in word_bags:

        new_string = ''
        date_string = ''
        date_info = find_date(string)

        if date_info == []:
            new_string = new_string + string
        else:
            date_loc = date_info[0][0]
            date = date_info[0][1]
            new_string = new_string + (string[0:date_loc[0]] + string[date_loc[1]:]).replace('()','').replace('( )','')
            date_string = date_string + '~DATE(S)~: ' + date
        if new_string:
            word_bags_cleaned.append(new_string)
        if date_string:
            word_bags_cleaned.append(date_string)   
            
    return word_bags_cleaned


def extract_headers(word_bags_cleaned, text_class, head_class):
    '''
    Find and classify headers.
    
    INPUTS
    word_bags_cleaned (list of strings):  List of cleaned resume lines.  Typically the 
    output of the clean_resume function.
    
    text_class (function belonging to Classifer class):  The function that will perform 
    the first classification of the line of the resume. 
    
    head_class (function belonging to Classifer class):  The function that will perform 
    the classification of line identified as headers by text_class
    
    OUTPUTS
    header_data (list of lists):  
    [['HEADER CLASS 1', line_index_1], ['HEADER CLASS 2', line_index_2], ...] 
    
    '''
    header_data = []
    for n in range(len(word_bags_cleaned)):
        bag = word_bags_cleaned[n]
        bag_class = text_class([bag])
        if bag_class[0] == 'HEAD':
            header_pred = head_class([bag])
            header_data.append([header_pred[0], n])
    return header_data


def extract_section(clfd_bags, head_pos, sec_string):
    
    '''
    Extract a given section from a resume.
    
    INPUTS
    clfd_bags (list of strings):  List of cleaned resume lines.  Typically the output of 
    the clean_resume function.
    
    head_pos (list of lists):  The type and position of each header.  Typically the output
    of the extract_headers function.
    
    sec_string (string):  The type of section to extract.  Currently supported types are
    those in ['HEAD_WORK', 'HEAD_EDUC', 'HEAD_OTHR']
    
    OUTPUTS
    sec (list of strings):  list containing the lines of the desired section.
    
    '''
    
    # find where each section starts/ends
    for n in range(len(head_pos)-1):
        head_pos[n][-1] = (head_pos[n][-1], head_pos[n+1][-1]) 
    head_pos[-1][-1] = (head_pos[-1][-1], len(clfd_bags))
  
    # combine all sections of the desired type
    sec_locs = []
    for lis in head_pos:
        if lis[0] == sec_string:
            sec_locs.append(lis[-1])
    sec_locs

	# extract lines
    sec = []
    for loc in sec_locs:
        sec = sec + clfd_bags[loc[0]:loc[1]]

    return sec
    
    
def clf_intra_sec(resume, clf_lines, clf_headers, clf_wrk):
    
    ''' 
    Classify lines inside the work section of a resume.
    
    INPUTS
    resume (string):  Full text of the resume.
    
	clf_lines (function):  (function belonging to Classifer class):  The function that 
	will perform the first classification of the line of the resume. 
    
    head_class (function belonging to Classifer class):  The function that will perform 
    the classification of line identified as headers by text_class.
    
    clf_wrf (function):  (function belonging to Classifer class):  The function that will 
    perform the classification of lines within the work section. 
     
    OUTPUT
    clfd_wrk_lines (list of tuples):  
    [('CLASSIFICATION', prob, line_loc, original string ), ... ]
     
    '''
    
    cleaned_resume = clean_resume(resume)
    extracted_headers = extract_headers(cleaned_resume, clf_lines.text_class, clf_headers.text_class)
    wrk_bags = extract_section(cleaned_resume, extracted_headers ,'HEAD_WORK')

    clfd_wrk_lines = []
    for n in range(len(wrk_bags)):
        wrk_line = wrk_bags[n]
        wrk_line_data = clf_wrk.text_class([wrk_line])
        
        # remove lines that have already been tagged as dates
        if wrk_line[0:len('~DATE(S)~:')] == '~DATE(S)~:':
            clfd_wrk_lines.append(('DATE', 1.0, n, wrk_line[len('~DATE(S)~:')+1:]))
        
        # classify work lines
        else:
            clfd_wrk_lines.append((wrk_line_data[0], round(np.max(wrk_line_data[1]),2), n, wrk_line ))
            
    return clfd_wrk_lines
        
    
def cluster_lines(section, sec_head):
    '''Group associated lines of the work section.
 
    INPUT
    section (list of tuples):  List of classified lines.  Typically the output of the   
    clf_intra_sec function.
    
    sec_head (sting):  The type of section.  e.g. 'HEAD_WORK'.
    
    OUTPUT
    clusters (list of dicts):  A list of dictionaries, each containing associated info.
    
    '''
    clusters = []
    emp_locs = []
    if sec_head == 'HEAD_WORK':
        for line in section:
            if line[0] == 'EMPL':
                clusters.append(OD([(line[0], line[-1])]))
                emp_locs.append(line[-2])

        for n in range(len(section)):
            line = section[n]
            if line[0] == 'TITL':
                temp = np.arange(len(emp_locs))[(line[-2] > np.array(emp_locs))]
                
                if len(temp) > 0:
                    emp_above = np.max(temp)
                    if 'TITLS' not in clusters[emp_above].keys():
                        clusters[emp_above]['TITLS'] = [OD([(line[0], line[-1])])]
                    else: 
                        clusters[emp_above]['TITLS'].append(OD([(line[0], line[-1])]))
                    if section[n-1][0] == 'DATE':
                        clusters[emp_above]['TITLS'][-1]['DATE'] = section[n-1][-1]
                    if n < len(section)-1 and section[n+1][0] == 'DATE':
                        clusters[emp_above]['TITLS'][-1]['DATE'] = section[n+1][-1]
                    if 'DATE' not in clusters[emp_above]['TITLS'][-1].keys():
                        emp_line = emp_locs[emp_above]
                        if section[emp_line+1][0] == 'DATE':
                            clusters[emp_above]['TITLS'][-1]['DATE'] = section[emp_line+1][-1]
                        elif section[emp_line+1][0] == 'UNKN' and section[emp_line+2][0] == 'DATE':
                            clusters[emp_above]['TITLS'][-1]['DATE'] = section[emp_line+2][-1]
    return clusters  

class RenderJSON(object):
    '''Render JSON in formattable html.  From https://github.com/caldwell/renderjson'''

    def __init__(self, json_data):
    
        if isinstance(json_data, dict):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json
        self.uuid = str(uuid.uuid4())
        
    def _ipython_display_(self):
        display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid),
            raw=True
        )
        display_javascript("""
        require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
          document.getElementById('%s').appendChild(renderjson(%s))
        });
        """ % (self.uuid, self.json_str), raw=True)
