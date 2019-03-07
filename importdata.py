#!/usr/bin/python

import pandas as pd
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from cStringIO import StringIO

'''
Some functions for extracting training data.
'''


def extract_data(filepath):
    '''
    Extract training and testing data from the json file
    
    INPUT
    filepath (string):  path to the csv containing the outout json file.
    
    OUTPUT
    list of tuples:
    
    [(emp_name_tr, pos_name_tr, edu_lines_tr, descrips_tr, headers_tr), \
            (emp_name_te, pos_name_te, edu_lines_te, descrips_te, headers_te)]
    
    '''

    df_resumes = pd.read_csv(filepath)

    edu_lines_tr = []
    emp_name_tr = []
    pos_name_tr = []
    descrips_tr = []
    headers_tr = []

    for n in range(len(df_resumes)-100):
        temp = (df_resumes.api_response_json[n]).replace("null","0")
        exec('test = ' + temp)

        if 'EducationHistory' in test['Resume']['StructuredXMLResume'].keys():    
            for dic in test['Resume']['StructuredXMLResume']['EducationHistory']['SchoolOrInstitution']:
                if 'School' in dic.keys():
                    edu_lines_tr.append(dic['School']['SchoolName'])
                if 'DegreeName' in dic['Degree'].keys():
                    edu_lines_tr.append(dic['Degree']['DegreeName'])
                if 'DegreeMajor' in dic['Degree'].keys():
                    for name in dic['Degree']['DegreeMajor']['Name']:
                        edu_lines_tr.append(name)

        if 'EmploymentHistory' in test['Resume']['StructuredXMLResume'].keys():
            for dic in test['Resume']['StructuredXMLResume']['EmploymentHistory']['EmployerOrg']:
                emp_name_tr.append(dic['EmployerOrgName'])

        if 'EmploymentHistory' in test['Resume']['StructuredXMLResume'].keys():
            for dic in test['Resume']['StructuredXMLResume']['EmploymentHistory']['EmployerOrg']:
                if 'Title' in dic['PositionHistory'][0].keys(): 
                    pos_name_tr.append(dic['PositionHistory'][0]['Title'])

        if 'EmploymentHistory' in test['Resume']['StructuredXMLResume'].keys():           
            for dic in test['Resume']['StructuredXMLResume']['EmploymentHistory']['EmployerOrg']:
                descrips_tr.append(dic['PositionHistory'][0]['Description'])

        for text in test['Resume']['UserArea']['sov:ResumeUserArea']['sov:Sections']['sov:Section']:
            if isinstance(text, dict) and '#text' in text.keys():
                headers_tr.append([text['#text'], text['@sectionType']])
                
        
    edu_lines_te = []
    emp_name_te = []
    pos_name_te = []
    descrips_te = []
    headers_te = []

    for n in range(len(df_resumes)-100,len(df_resumes)):
        temp = (df_resumes.api_response_json[n]).replace("null","0")
        exec('test = '+temp)

        if 'EducationHistory' in test['Resume']['StructuredXMLResume'].keys():    
            for dic in test['Resume']['StructuredXMLResume']['EducationHistory']['SchoolOrInstitution']:
                if 'School' in dic.keys():
                    edu_lines_te.append(dic['School']['SchoolName'])
                if 'DegreeName' in dic['Degree'].keys():
                    edu_lines_te.append(dic['Degree']['DegreeName'])
                if 'DegreeMajor' in dic['Degree'].keys():
                    for name in dic['Degree']['DegreeMajor']['Name']:
                        edu_lines_te.append(name)

        if 'EmploymentHistory' in test['Resume']['StructuredXMLResume'].keys():
            for dic in test['Resume']['StructuredXMLResume']['EmploymentHistory']['EmployerOrg']:
                emp_name_te.append(dic['EmployerOrgName'])

        if 'EmploymentHistory' in test['Resume']['StructuredXMLResume'].keys():
            for dic in test['Resume']['StructuredXMLResume']['EmploymentHistory']['EmployerOrg']:
                if 'Title' in dic['PositionHistory'][0].keys(): 
                    pos_name_te.append(dic['PositionHistory'][0]['Title'])

        if 'EmploymentHistory' in test['Resume']['StructuredXMLResume'].keys():           
            for dic in test['Resume']['StructuredXMLResume']['EmploymentHistory']['EmployerOrg']:
                descrips_te.append(dic['PositionHistory'][0]['Description'])

        for text in test['Resume']['UserArea']['sov:ResumeUserArea']['sov:Sections']['sov:Section']:
            if isinstance(text, dict):
                if '#text' in text.keys():
                    headers_te.append([text['#text'], text['@sectionType']])
            
                
    return [(emp_name_tr, pos_name_tr, edu_lines_tr, descrips_tr, headers_tr), \
            (emp_name_te, pos_name_te, edu_lines_te, descrips_te, headers_te)]
 

def pdf_to_text(pdfname):
    '''Extract text data from pdf.
	
	INPUTS
	pdfname (string): filepath to odf

    OUTPUTS
    text (string):  text extracted from the pdf
	
	'''	
	
    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Extract text
    fp = file(pdfname, 'rb')
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()
    # Get text from StringIO
    text = sio.getvalue()
    # Cleanup
    device.close()
    sio.close()
    return text

