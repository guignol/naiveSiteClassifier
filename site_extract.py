import os
import math
import sys
import csv
import urllib2
import random
import re
import pickle
import socket
import datetime


from bs4 import BeautifulSoup
from collections import Counter


import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer

from textblob.classifiers import NaiveBayesClassifier

import contextlib



print datetime.datetime.now().time()
arguments = sys.argv
NB_SITES = int(arguments[1])

LOAD_FROM_CLASSIFIER = True if arguments[2] == 'true' else False
SAVE_CLASSIFIER = True if arguments[3] == 'true' else False

LOAD_FROM_WEBSITE_DATA = True if arguments[4] == 'true' else False
SAVE_WEBSITE_DATA = True if arguments[5] == 'true' else False

DATA_FOLDER = "/Users/Jean-Louis/spark/data/"
DATA_FOLDER = "/home/neo_f/ML/data/"

CATEGORIES_FILE = DATA_FOLDER + "categories/categories"
CLASSIFIED_SITES_FILE = DATA_FOLDER + "categories/classified_shuffle_sites"
CLASSIFIER_MODEL_FILE = DATA_FOLDER + "categories/classifier/" + str(NB_SITES) + "/classifier_model.pickle"
WEBSITE_DATA_FILE = DATA_FOLDER + "categories/classifier/" + str(NB_SITES) + "/website_data.pickle"

BENCHMARK_FILE = DATA_FOLDER + "results/benchmark.csv"

BENCHMARK_DATA = dict()
BENCHMARK_DATA['dataset_size'] = NB_SITES


URL_LIMITS = [0, NB_SITES]


DELIMITER = "\t"
DELIMITER_URL = ","
URL_LIST = []

SITE_CATEGORIES = dict()
SITE_SUBCATEGORIES = dict()
SUBCATEGORY_REVERSE_DICT = dict()
CLASSIFIED_SITES = dict()
SUBCATEGORY_CATEGORY_DICT = dict()
SITES_CONTENT = dict()

IGNORED_CATEGORIES = ['247', '248', '249']

STOP = stopwords.words('english')
TOKENIZER = RegexpTokenizer(r'\w+')

IP_PATTERN = re.compile(ur'(\d*\.\d*\.\d*)')
PAGE_COUNTER_PATTERN = re.compile(ur'\d*')
TAG_PATTERN = re.compile(ur'>(.*)<', re.IGNORECASE)
ENCODING_CLEANER_PATTERN = re.compile(ur'\\x.{2}', re.MULTILINE | re.IGNORECASE)

STEMMER = LancasterStemmer()

REQUEST_HEADER={'User-Agent': "(Mozilla/5.0 (Windows; U; Windows NT 6.0;en-US; rv:1.9.2) Gecko/20100115 Firefox/3.6" }


urlopener=urllib2.build_opener()

web_stopwords = [
	'webmaster',
	'www',
	'com'
]


def load_classified_sites():
	with open(CLASSIFIED_SITES_FILE, 'rb') as sites_categories_file:
		sites_categories_file_reader = csv.reader(sites_categories_file, delimiter=',')
		for row in sites_categories_file_reader:
			if row[3] not in IGNORED_CATEGORIES:
				CLASSIFIED_SITES[row[0]] = {'domain': row[0],'category': row[1], 'subcategory': row[2], 'subcategory_id':row[3]}
	return CLASSIFIED_SITES



def load_url_list(URL_LIST):
	for filename in os.listdir(DATA_FOLDER + "url"):
		data_file = DATA_FOLDER + "url/" + filename
		rank = 0
		if filename != ".DS_Store":
			with open(data_file, 'rb') as url_file:
				url_reader = csv.reader(url_file, delimiter=DELIMITER_URL)
				for row in url_reader:
					rank = rank + 1
					if rank >= URL_LIMITS[0] and rank <= URL_LIMITS[1]:
						try:
							URL_LIST.append(row[0])
						except:
							pass
	URL_LIST = list(set(URL_LIST))

	random.shuffle(URL_LIST)

	return URL_LIST


def load_categories(SITE_CATEGORIES,SITE_SUBCATEGORIES,SUBCATEGORY_CATEGORY_DICT, SUBCATEGORY_REVERSE_DICT):
	with open(CATEGORIES_FILE, 'rU') as category_file:
		category_reader = csv.reader(category_file, delimiter=',')
		for row in category_reader:
			SITE_CATEGORIES[row[2]] = row[1]
			SITE_SUBCATEGORIES[row[2]] = row[0]
			SUBCATEGORY_CATEGORY_DICT[row[0]] = {'id': row[0], 'category': row[1], 'subcategory': row[2]}
			SUBCATEGORY_REVERSE_DICT[row[2]] = {'id': row[0], 'category': row[1], 'subcategory': row[2]}
	SITE_CATEGORIES = list(set(SITE_CATEGORIES))
	return SUBCATEGORY_CATEGORY_DICT


def subcategory_to_dictionnary(domain, subcategory):
	output = dict()
	if subcategory != "":
		output['domain'] = domain
		output['category'] = SUBCATEGORY_REVERSE_DICT[subcategory]['category']
		output['subcategory'] = subcategory
		output['subcategory_id'] = SUBCATEGORY_REVERSE_DICT[subcategory]['id']
		CLASSIFIED_SITES[domain] = output
	return output



load_classified_sites()



load_url_list(URL_LIST)



URL_LIST


load_categories(SITE_CATEGORIES,SITE_SUBCATEGORIES,SUBCATEGORY_CATEGORY_DICT,SUBCATEGORY_REVERSE_DICT)


class site_extractor:
	def is_visible_text(self, element):
		if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
			return False
		elif re.match('<!--.*-->', str(element.encode('utf-8'))):
			return False
		else:
			return True

	def clean_sentence(self, sentence, encode = True, encode_type = 'ascii'):
		output = []
		sentence = sentence.lower()

		if len(sentence) > 0 and not sentence is None:
			if encode:
				sentence = sentence.encode(encode_type, 'replace').strip()
			sentence = re.sub(r'[^\x00-\x7F]+',' ', sentence)
			words = TOKENIZER.tokenize(sentence)

			for word in words:
				try:
					if word not in STOP and not word.isdigit() and word not in web_stopwords and len(word) > 2:
						output.append(STEMMER.stem(word))
				except:
					pass
			return output

	def extract_hyperlink_text(self, soup):
		output = []
		for links in soup.findAll('a', text=True):
			for link in links:
				if len(link) > 1:
					sentence = link.string
					output += self.clean_sentence(sentence, True, 'utf-8') 
		return output

	def extract_meta_description(self, soup):
		desc = soup.findAll(attrs={"name":"description"})
		output = []
		if len(desc) >0:
			sentence = desc[0]['content']
			output += self.clean_sentence(sentence)

		return output

	def extract_title(self, soup):
		output = []
		try:
			title = soup.html.head.title.string
			if len(title) > 0 :
				output += self.clean_sentence(title)
		except:
			pass
		return output

	def extract_visible_text(self, soup):
		text = ""
		try:
			text = soup.find('h1').getText()
		except:
			pass
		try:
			text += " " + soup.find('h2').getText()
		except:
			pass
		try:
			text += " " + soup.find('h3').getText()
		except:
			pass
		try:
			text += " " + soup.find('h4').getText()
		except:
			pass
		try:
			text += " " +soup.find('h5').getText()
		except:
			pass
		try:
			text += " " + soup.find('p').getText()
		except:
			pass
		# sentences = filter(self.is_visible_text, text)
		output = []
		if len(text) > 0:
			output += self.clean_sentence(text)			

		return output

def extract_website_content(url):
	url = 'http://www.' + url
	request = urllib2.Request(url, None, REQUEST_HEADER)
	words = []
	try:
		reponse = urlopener.open(request, None, 2)
		content = reponse.read(1000000)
		soup = BeautifulSoup(content, 'html.parser')

		se = site_extractor()
		words = se.extract_hyperlink_text(soup)
		words += se.extract_meta_description(soup) 
		words += se.extract_title(soup) 
		words += se.extract_visible_text(soup)
	except:
		pass
	
	words = ' '.join(words)

	return words


def extract_all_url(URL_LIST):
	list_size = len(URL_LIST)
	item = 0
	for url in URL_LIST:
		item += 1
		print str(item) + "/" + str(list_size)
		if url != "" and is_ip(url) == False and url not in SITES_CONTENT:
			try:
				SITES_CONTENT[url] = extract_website_content(url)
			except:
				pass
			
	return SITES_CONTENT

def is_ip(domain):
	return not re.search(IP_PATTERN, domain) is None


def build_data_set(set_type, size):
	data_set = []
	i = 0
	for domain, categories in set_type.iteritems():
		if domain != "" and is_ip(domain) == False and i <= size:
			i += 1
			sys.stdout.write("\r extracting: %d%%" % math.trunc(i*100/size))
			sys.stdout.flush()
			data_set.append((extract_website_content(domain), categories['category']))

	return data_set

def save_to_file(objct, filename):
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	f = open(filename, 'wb')
	out = pickle.dump(objct, f)
	f.close()
	return out

def load_from_file(filename):
	f = open(filename, 'rb')
	objct = pickle.load(f)
	f.close()
	return objct

def extract_website(domain):
	output = ''
	if domain != "" and is_ip(domain) == False:
		output += extract_website_content(domain)
	return output

def classify_website(domain):
	return classifier.classify(extract_website(domain))
		

def save_benchmark(d):
	with open(BENCHMARK_FILE, 'a') as fp:
		bench_file = csv.writer(fp, delimiter=',')
		data = [[
			d['dataset_size'], d['extract_start_time'], d['extract_end_time'], d['train_start_time'], d['train_end_time'], d['test_start_time'], d['test_end_time'], d['accuracy'] 
			]]
		bench_file.writerows(data)

data_set_split = math.trunc(round(NB_SITES*2/3,0))

if LOAD_FROM_WEBSITE_DATA:
	print "loading website data"
	featuresets = load_from_file(WEBSITE_DATA_FILE)
else:
	print "extracting website data"
	BENCHMARK_DATA['extract_start_time'] = datetime.datetime.now().time()
	print BENCHMARK_DATA['extract_start_time']
	featuresets = build_data_set(CLASSIFIED_SITES, NB_SITES)
	BENCHMARK_DATA['extract_end_time'] = datetime.datetime.now().time()
	print BENCHMARK_DATA['extract_end_time']

if SAVE_WEBSITE_DATA:
	print "saving website data"
	save_to_file(featuresets, WEBSITE_DATA_FILE)

random.shuffle(featuresets)

train_set, test_set = featuresets[0:], featuresets[:data_set_split]


print "train and test set ready"

if LOAD_FROM_CLASSIFIER:
	print "loading classifier"
	classifier = load_from_file(CLASSIFIER_MODEL_FILE)
else:
	print "building classifier"
	BENCHMARK_DATA['train_start_time'] = datetime.datetime.now().time()
	print BENCHMARK_DATA['train_start_time']
	classifier = NaiveBayesClassifier(train_set)
	BENCHMARK_DATA['train_end_time'] = datetime.datetime.now().time()
	print BENCHMARK_DATA['train_end_time']

if SAVE_CLASSIFIER:
	print "saving classifier"
	save_to_file(classifier, CLASSIFIER_MODEL_FILE)


print "testing classifier accuracy"
BENCHMARK_DATA['test_start_time'] = datetime.datetime.now().time()
print BENCHMARK_DATA['test_start_time']
BENCHMARK_DATA['accuracy'] = classifier.accuracy(test_set)
print BENCHMARK_DATA['accuracy']
BENCHMARK_DATA['test_end_time'] = datetime.datetime.now().time()
print BENCHMARK_DATA['test_end_time']

save_benchmark(BENCHMARK_DATA)

# random.shuffle(featuresets)
# train_set, test_set = featuresets[0:], featuresets[:data_set_split]
# classifier = NaiveBayesClassifier(train_set)
# print classifier.accuracy(test_set)

# random.shuffle(featuresets)
# train_set, test_set = featuresets[0:], featuresets[:data_set_split]
# classifier = NaiveBayesClassifier(train_set)
# print classifier.accuracy(test_set)

# random.shuffle(featuresets)
# train_set, test_set = featuresets[0:], featuresets[:data_set_split]
# classifier = NaiveBayesClassifier(train_set)
# print classifier.accuracy(test_set)

print "classify youporn.com"
print datetime.datetime.now().time()
print classify_website("youporn.com")
print datetime.datetime.now().time()
print "classify facebook.com"
print classify_website("facebook.com")
print "classify wajam.com"
print classify_website("wajam.com")
print "classify cnn.com"
print classify_website("cnn.com")
print "classify weather.com"
print classify_website("weather.com")
print "classify ebay.com"
print classify_website("ebay.com")
print datetime.datetime.now().time()
	