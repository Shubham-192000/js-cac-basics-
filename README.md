# Importing libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# Importing NLTK stopwords and ensuring they are downloaded
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Importing WordCloud for text visualization
from wordcloud import WordCloud

# Importing libraries for model building and evaluation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importing libraries for model evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Ensure TensorFlow version compatibility (optional)
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

     
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
TensorFlow version: 2.17.0
New section

!pip install --upgrade tensorflow



     
Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (2.17.0)
Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.6.0)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: h5py>=3.10.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.11.0)
Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (18.1.1)
Requirement already satisfied: ml-dtypes<0.5.0,>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.4.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.3.0)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.1)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.20.3)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.32.3)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow) (71.0.4)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (4.12.2)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.64.1)
Requirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.17.0)
Requirement already satisfied: keras>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.4.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.37.1)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.26.4)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)
Requirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (13.7.1)
Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.0.8)
Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.12.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2024.7.4)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.6)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.0.3)
Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.18,>=2.17->tensorflow) (2.1.5)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (2.16.1)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.2.0->tensorflow) (0.1.2)


     

import pandas as pd
import json

def read_json_lines(filename):
    data_list = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip lines that cause errors
    return data_list

# Read and parse both JSON files
data1_list = read_json_lines('Sarcasm_Headlines_Dataset.json')
data2_list = read_json_lines('Sarcasm_Headlines_Dataset_v2.json')

# Convert lists to DataFrames
data1 = pd.DataFrame(data1_list)
data2 = pd.DataFrame(data2_list)

# Concatenate the DataFrames
data = pd.concat([data1, data2], ignore_index=True)

# Display the concatenated data
print(data.head())

     
                                        article_link  \
0  https://www.huffingtonpost.com/entry/versace-b...   
1  https://www.huffingtonpost.com/entry/roseanne-...   
2  https://local.theonion.com/mom-starting-to-fea...   
3  https://politics.theonion.com/boehner-just-wan...   
4  https://www.huffingtonpost.com/entry/jk-rowlin...   

                                            headline  is_sarcastic  
0  former versace store clerk sues over secret 'b...             0  
1  the 'roseanne' revival catches up to our thorn...             0  
2  mom starting to fear son's web series closest ...             1  
3  boehner just wants wife to listen, not come up...             1  
4  j.k. rowling wishes snape happy birthday in th...             0  

data.info()

     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 55328 entries, 0 to 55327
Data columns (total 3 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   article_link  55328 non-null  object
 1   headline      55328 non-null  object
 2   is_sarcastic  55328 non-null  int64 
dtypes: int64(1), object(2)
memory usage: 1.3+ MB

# Check the dataset label balance or not

# Check number of headlines by is_sarcastics
plt.figure(figsize=(10, 4))
sns.countplot(x='is_sarcastic', data=data, palette="Set1").set_title(
	"Count and plot of headlines")
plt.show()

     
<ipython-input-12-ff142ce33935>:5: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x='is_sarcastic', data=data, palette="Set1").set_title(


#downloading the stopwords corpus list
nltk.download('stopwords')
stopwords_list = stopwords.words('english')

     
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!

def clean_text(sentences):
	# convert text to lowercase
	text = sentences.lower()
	# remove text in square brackets
	text = re.sub('
', '', text)
	# removing punctuations
	text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
	# removing words containing digits
	text = re.sub('\w*\d\w*', '', text)
	# Join the words
	text = ' '.join([word for word in text.split()
					if word not in stopwords_list])
	return text


print(data['headline'].iloc[1])
clean_text(data['headline'].iloc[1])

     
the 'roseanne' revival catches up to our thorny political mood, for better and worse
'roseanne revival catches thorny political mood better worse'

#new column to store cleaned text
data['cleaned_headline']=data['headline'].map(clean_text)

     

# Combine all sarcastic cleaned headlines into a single text
import matplotlib.pyplot as plt
from wordcloud import WordCloud
Sarcastic_text = ' '.join(
	data['cleaned_headline'][data['is_sarcastic'] == 1].tolist())

# Import the necessary libraries

# Create a WordCloud object with specified width, height, and background color
wordcloud = WordCloud(width=800, height=400,
					background_color='white').generate(Sarcastic_text)

# Display the WordCloud without axes
plt.figure(figsize=(10, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Sarcastic')
plt.show()

     


# Combine all non-sarcastic cleaned headlines into a single text
Non_Sarcastic_text = ' '.join(
	data['cleaned_headline'][data['is_sarcastic'] == 0].tolist())

# Create a WordCloud object with specified width, height, and background color
wordcloud = WordCloud(width=800, height=400,
					background_color='maroon').generate(Non_Sarcastic_text)

# Display the WordCloud without axes
plt.figure(figsize=(10, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Not Sarcastic')
plt.show()

     



     

#coverting the columns into lists
text = data['cleaned_headline'].tolist()
label = data['is_sarcastic'].tolist()

     

# train :test : validation = 80:10:10
train_portion = .8

# Set the train size using training_portion arg
train_size = int(len(text) * train_portion)

# Training dataset
train_text = text[:train_size]
train_label = label[:train_size]
# Validations dataset
valid_size = train_size+int((len(text)-train_size)/2)
val_text = text[train_size:valid_size]
val_label = label[train_size:valid_size]
# Testing dataset
test_text = text[valid_size:]
test_label = label[valid_size:]

# Check
print('Training data :', len(train_text), len(train_label))
print('Validations data :', len(val_text), len(val_label))
print('Testing data :', len(test_text), len(test_label))

     
Training data : 44262 44262
Validations data : 5533 5533
Testing data : 5533 5533

# Set parameters
# Max len of unique words
vocab_size = 10000

# Embedding dimension value
embedding_dim = 200

# Max length of sentence
max_length = 60

# pad_sequences arg
padding_type = 'post'

# Unknow words = 
oov_tok = ''

# Tokenizing and padding
# Create a tokenizer with a specified vocabulary size and out-of-vocabulary token
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# Fit the tokenizer on the training text data to create word-to-index mapping
tokenizer.fit_on_texts(train_text)

     

# Get the word index from the tokenizer
word_index = tokenizer.word_index

#Printing the word_index
word_index

     
{'<OOV>': 1,
 'new': 2,
 'man': 3,
 'trump': 4,
 'us': 5,
 'report': 6,
 'one': 7,
 'area': 8,
 'woman': 9,
 'donald': 10,
 'says': 11,
 'day': 12,
 'like': 13,
 'get': 14,
 'first': 15,
 'time': 16,
 'people': 17,
 'trumps': 18,
 'obama': 19,
 'house': 20,
 'life': 21,
 'still': 22,
 'make': 23,
 'white': 24,
 'women': 25,
 'back': 26,
 'clinton': 27,
 'world': 28,
 'could': 29,
 'years': 30,
 'family': 31,
 'americans': 32,
 'way': 33,
 'study': 34,
 'black': 35,
 'gop': 36,
 'bill': 37,
 'would': 38,
 'best': 39,
 'cant': 40,
 'really': 41,
 'police': 42,
 'american': 43,
 'watch': 44,
 'show': 45,
 'school': 46,
 'know': 47,
 'home': 48,
 'good': 49,
 'nation': 50,
 'going': 51,
 'finds': 52,
 'say': 53,
 'things': 54,
 'president': 55,
 'death': 56,
 'video': 57,
 'last': 58,
 'love': 59,
 'parents': 60,
 'year': 61,
 'mom': 62,
 'big': 63,
 'state': 64,
 'health': 65,
 'hillary': 66,
 'every': 67,
 'kids': 68,
 'need': 69,
 'getting': 70,
 'may': 71,
 'gets': 72,
 'campaign': 73,
 'party': 74,
 'right': 75,
 'little': 76,
 'change': 77,
 'work': 78,
 'john': 79,
 'dead': 80,
 'dont': 81,
 'makes': 82,
 'never': 83,
 'take': 84,
 'court': 85,
 'america': 86,
 'child': 87,
 'calls': 88,
 'news': 89,
 'doesnt': 90,
 'next': 91,
 'heres': 92,
 'hes': 93,
 'takes': 94,
 'go': 95,
 'see': 96,
 'look': 97,
 'want': 98,
 'even': 99,
 'stop': 100,
 'real': 101,
 'nations': 102,
 'local': 103,
 'war': 104,
 'gay': 105,
 'guy': 106,
 'bush': 107,
 'got': 108,
 'election': 109,
 'office': 110,
 'dad': 111,
 'star': 112,
 'around': 113,
 'college': 114,
 'help': 115,
 'thing': 116,
 'plan': 117,
 'game': 118,
 'another': 119,
 'million': 120,
 'made': 121,
 'dog': 122,
 'baby': 123,
 'job': 124,
 'week': 125,
 'live': 126,
 'finally': 127,
 'gun': 128,
 'debate': 129,
 'wants': 130,
 'actually': 131,
 'mans': 132,
 'congress': 133,
 'much': 134,
 'high': 135,
 'couple': 136,
 'students': 137,
 'money': 138,
 'care': 139,
 'north': 140,
 'national': 141,
 'bad': 142,
 'announces': 143,
 'two': 144,
 'ever': 145,
 'friends': 146,
 'sex': 147,
 'shows': 148,
 'season': 149,
 'ways': 150,
 'better': 151,
 'reveals': 152,
 'shooting': 153,
 'god': 154,
 'story': 155,
 'trying': 156,
 'men': 157,
 'enough': 158,
 'climate': 159,
 'night': 160,
 'give': 161,
 'teen': 162,
 'top': 163,
 'without': 164,
 'wont': 165,
 'sexual': 166,
 'history': 167,
 'senate': 168,
 'everyone': 169,
 'away': 170,
 'fight': 171,
 'media': 172,
 'business': 173,
 'making': 174,
 'facebook': 175,
 'paul': 176,
 'friend': 177,
 'supreme': 178,
 'old': 179,
 'city': 180,
 'face': 181,
 'introduces': 182,
 'deal': 183,
 'children': 184,
 'pope': 185,
 'sanders': 186,
 'food': 187,
 'attack': 188,
 'part': 189,
 'law': 190,
 'entire': 191,
 'wedding': 192,
 'york': 193,
 'end': 194,
 'single': 195,
 'girl': 196,
 'find': 197,
 'son': 198,
 'fire': 199,
 'think': 200,
 'support': 201,
 'pretty': 202,
 'tv': 203,
 'former': 204,
 'found': 205,
 'movie': 206,
 'tell': 207,
 'book': 208,
 'great': 209,
 'already': 210,
 'morning': 211,
 'email': 212,
 'call': 213,
 'didnt': 214,
 'car': 215,
 'republican': 216,
 'body': 217,
 'film': 218,
 'come': 219,
 'james': 220,
 'power': 221,
 'run': 222,
 'future': 223,
 'presidential': 224,
 'government': 225,
 'fans': 226,
 'public': 227,
 'releases': 228,
 'company': 229,
 'keep': 230,
 'coming': 231,
 'self': 232,
 'name': 233,
 'violence': 234,
 'republicans': 235,
 'use': 236,
 'behind': 237,
 'photos': 238,
 'speech': 239,
 'asks': 240,
 'christmas': 241,
 'times': 242,
 'rights': 243,
 'line': 244,
 'secret': 245,
 'free': 246,
 'case': 247,
 'thinks': 248,
 'might': 249,
 'killed': 250,
 'girls': 251,
 'talk': 252,
 'room': 253,
 'human': 254,
 'days': 255,
 'long': 256,
 'something': 257,
 'student': 258,
 'worlds': 259,
 'unveils': 260,
 'voters': 261,
 'im': 262,
 'country': 263,
 'tax': 264,
 'used': 265,
 'security': 266,
 'scientists': 267,
 'states': 268,
 'must': 269,
 'looking': 270,
 'boy': 271,
 'vote': 272,
 'win': 273,
 'bernie': 274,
 'social': 275,
 'group': 276,
 'poll': 277,
 'music': 278,
 'always': 279,
 'race': 280,
 'many': 281,
 'ad': 282,
 'team': 283,
 'ryan': 284,
 'sure': 285,
 'perfect': 286,
 'goes': 287,
 'democrats': 288,
 'political': 289,
 'marriage': 290,
 'admits': 291,
 'super': 292,
 'middle': 293,
 'department': 294,
 'claims': 295,
 'dies': 296,
 'save': 297,
 'missing': 298,
 'mother': 299,
 'teacher': 300,
 'open': 301,
 'twitter': 302,
 'forced': 303,
 'person': 304,
 'lets': 305,
 'ban': 306,
 'candidate': 307,
 'inside': 308,
 'living': 309,
 'michael': 310,
 'second': 311,
 'judge': 312,
 'running': 313,
 'full': 314,
 'minutes': 315,
 'everything': 316,
 'texas': 317,
 'father': 318,
 'plans': 319,
 'taking': 320,
 'youre': 321,
 'meet': 322,
 'art': 323,
 'let': 324,
 'california': 325,
 'thousands': 326,
 'wall': 327,
 'summer': 328,
 'red': 329,
 'control': 330,
 'comes': 331,
 'lives': 332,
 'past': 333,
 'obamacare': 334,
 'gives': 335,
 'warns': 336,
 'wife': 337,
 'pay': 338,
 'fucking': 339,
 'looks': 340,
 'secretary': 341,
 'wrong': 342,
 'put': 343,
 'female': 344,
 'today': 345,
 'talks': 346,
 'ceo': 347,
 'reports': 348,
 'list': 349,
 'thought': 350,
 'head': 351,
 'left': 352,
 'photo': 353,
 'hours': 354,
 'cancer': 355,
 'idea': 356,
 'ready': 357,
 'water': 358,
 'tells': 359,
 'george': 360,
 'mike': 361,
 'together': 362,
 'place': 363,
 'employee': 364,
 'shot': 365,
 'needs': 366,
 'letter': 367,
 'probably': 368,
 'hot': 369,
 'street': 370,
 'working': 371,
 'cruz': 372,
 'stars': 373,
 'service': 374,
 'romney': 375,
 'town': 376,
 'someone': 377,
 'kill': 378,
 'crisis': 379,
 'dream': 380,
 'daughter': 381,
 'justice': 382,
 'three': 383,
 'tips': 384,
 'young': 385,
 'moms': 386,
 'lost': 387,
 'kim': 388,
 'set': 389,
 'start': 390,
 'order': 391,
 'record': 392,
 'wins': 393,
 'yet': 394,
 'shes': 395,
 'officials': 396,
 'feel': 397,
 'cat': 398,
 'breaking': 399,
 'questions': 400,
 'isis': 401,
 'washington': 402,
 'russia': 403,
 'believe': 404,
 'biden': 405,
 'korea': 406,
 'phone': 407,
 'womens': 408,
 'age': 409,
 'restaurant': 410,
 'eating': 411,
 'obamas': 412,
 'meeting': 413,
 'latest': 414,
 'rock': 415,
 'heart': 416,
 'prison': 417,
 'shit': 418,
 'internet': 419,
 'attacks': 420,
 'chief': 421,
 'percent': 422,
 'march': 423,
 'administration': 424,
 'south': 425,
 'months': 426,
 'nuclear': 427,
 'earth': 428,
 'democratic': 429,
 'king': 430,
 'owner': 431,
 'talking': 432,
 'less': 433,
 'francis': 434,
 'giving': 435,
 'class': 436,
 'move': 437,
 'small': 438,
 'nothing': 439,
 'assault': 440,
 'education': 441,
 'majority': 442,
 'federal': 443,
 'leaves': 444,
 'military': 445,
 'problem': 446,
 'florida': 447,
 'guide': 448,
 'chris': 449,
 'fan': 450,
 'director': 451,
 'reason': 452,
 'online': 453,
 'happy': 454,
 'gift': 455,
 'congressman': 456,
 'thinking': 457,
 'kind': 458,
 'whats': 459,
 'moment': 460,
 'buy': 461,
 'personal': 462,
 'hell': 463,
 'tweets': 464,
 'iran': 465,
 'sleep': 466,
 'stephen': 467,
 'ted': 468,
 'box': 469,
 'following': 470,
 'told': 471,
 'drug': 472,
 'outside': 473,
 'community': 474,
 'using': 475,
 'ask': 476,
 'senator': 477,
 'since': 478,
 'system': 479,
 'reasons': 480,
 'hard': 481,
 'air': 482,
 'fox': 483,
 'lot': 484,
 'travel': 485,
 'kid': 486,
 'knows': 487,
 'ice': 488,
 'fbi': 489,
 'birthday': 490,
 'rules': 491,
 'response': 492,
 'politics': 493,
 'watching': 494,
 'leaders': 495,
 'series': 496,
 'americas': 497,
 'mark': 498,
 'hollywood': 499,
 'word': 500,
 'rise': 501,
 'celebrates': 502,
 'mothers': 503,
 'beautiful': 504,
 'month': 505,
 'protest': 506,
 'play': 507,
 'leave': 508,
 'issues': 509,
 'hit': 510,
 'read': 511,
 'isnt': 512,
 'hair': 513,
 'union': 514,
 'anything': 515,
 'huge': 516,
 'bar': 517,
 'different': 518,
 'favorite': 519,
 'excited': 520,
 'millions': 521,
 'scott': 522,
 'cops': 523,
 'special': 524,
 'chinese': 525,
 'immigration': 526,
 'visit': 527,
 'straight': 528,
 'holiday': 529,
 'powerful': 530,
 'david': 531,
 'drunk': 532,
 'taylor': 533,
 'message': 534,
 'trailer': 535,
 'relationship': 536,
 'jimmy': 537,
 'victims': 538,
 'fun': 539,
 'offers': 540,
 'feels': 541,
 'girlfriend': 542,
 'russian': 543,
 'candidates': 544,
 'muslim': 545,
 'hope': 546,
 'worried': 547,
 'bring': 548,
 'leader': 549,
 'tom': 550,
 'china': 551,
 'weekend': 552,
 'lessons': 553,
 'sick': 554,
 'mass': 555,
 'well': 556,
 'career': 557,
 'across': 558,
 'interview': 559,
 'birth': 560,
 'huffpost': 561,
 'waiting': 562,
 'accused': 563,
 'crash': 564,
 'schools': 565,
 'investigation': 566,
 'opens': 567,
 'trip': 568,
 'third': 569,
 'words': 570,
 'die': 571,
 'totally': 572,
 'discover': 573,
 'billion': 574,
 'become': 575,
 'fall': 576,
 'theres': 577,
 'break': 578,
 'cover': 579,
 'massive': 580,
 'adds': 581,
 'hands': 582,
 'united': 583,
 'front': 584,
 'west': 585,
 'adorable': 586,
 'spends': 587,
 'least': 588,
 'hate': 589,
 'iraq': 590,
 'whole': 591,
 'weeks': 592,
 'employees': 593,
 'teens': 594,
 'called': 595,
 'returns': 596,
 'starting': 597,
 'sports': 598,
 'syria': 599,
 'fashion': 600,
 'late': 601,
 'theyre': 602,
 'point': 603,
 'struggling': 604,
 'boys': 605,
 'names': 606,
 'dance': 607,
 'moving': 608,
 'conversation': 609,
 'signs': 610,
 'early': 611,
 'center': 612,
 'reality': 613,
 'song': 614,
 'dating': 615,
 'puts': 616,
 'stage': 617,
 'cop': 618,
 'joe': 619,
 'killing': 620,
 'almost': 621,
 'un': 622,
 'abortion': 623,
 'final': 624,
 'weird': 625,
 'wearing': 626,
 'nfl': 627,
 'true': 628,
 'learned': 629,
 'pence': 630,
 'prince': 631,
 'users': 632,
 'light': 633,
 'experience': 634,
 'lose': 635,
 'apple': 636,
 'queer': 637,
 'breaks': 638,
 'date': 639,
 'host': 640,
 'vows': 641,
 'fuck': 642,
 'hurricane': 643,
 'policy': 644,
 'hits': 645,
 'wars': 646,
 'turn': 647,
 'workers': 648,
 'worst': 649,
 'road': 650,
 'walking': 651,
 'test': 652,
 'wait': 653,
 'keeps': 654,
 'syrian': 655,
 'bus': 656,
 'surprise': 657,
 'rubio': 658,
 'act': 659,
 'halloween': 660,
 'lead': 661,
 'audience': 662,
 'jobs': 663,
 'sign': 664,
 'murder': 665,
 'seen': 666,
 'official': 667,
 'decision': 668,
 'williams': 669,
 'kills': 670,
 'brings': 671,
 'global': 672,
 'anniversary': 673,
 'hoping': 674,
 'university': 675,
 'coworker': 676,
 'stand': 677,
 'apparently': 678,
 'wishes': 679,
 'awards': 680,
 'whether': 681,
 'return': 682,
 'governor': 683,
 'iowa': 684,
 'suspect': 685,
 'feeling': 686,
 'trans': 687,
 'suicide': 688,
 'program': 689,
 'robert': 690,
 'michelle': 691,
 'learn': 692,
 'post': 693,
 'risk': 694,
 'allegations': 695,
 'apartment': 696,
 'turns': 697,
 'playing': 698,
 'cool': 699,
 'mental': 700,
 'mueller': 701,
 'hall': 702,
 'ideas': 703,
 'problems': 704,
 'role': 705,
 'longer': 706,
 'paris': 707,
 'members': 708,
 'advice': 709,
 'worth': 710,
 'peace': 711,
 'transgender': 712,
 'planned': 713,
 'key': 714,
 'completely': 715,
 'important': 716,
 'reform': 717,
 'biggest': 718,
 'fear': 719,
 'reportedly': 720,
 'artist': 721,
 'coffee': 722,
 'abuse': 723,
 'clearly': 724,
 'eat': 725,
 'supporters': 726,
 'remember': 727,
 'oscar': 728,
 'kardashian': 729,
 'park': 730,
 'hand': 731,
 'steve': 732,
 'football': 733,
 'success': 734,
 'question': 735,
 'chance': 736,
 'anyone': 737,
 'demands': 738,
 'dinner': 739,
 'store': 740,
 'queen': 741,
 'fighting': 742,
 'voice': 743,
 'church': 744,
 'begins': 745,
 'poor': 746,
 'general': 747,
 'vacation': 748,
 'tour': 749,
 'press': 750,
 'urges': 751,
 'simple': 752,
 'asking': 753,
 'force': 754,
 'beauty': 755,
 'boss': 756,
 'rest': 757,
 'album': 758,
 'chicago': 759,
 'eyes': 760,
 'throws': 761,
 'lgbt': 762,
 'ferguson': 763,
 'died': 764,
 'apologizes': 765,
 'push': 766,
 'space': 767,
 'oscars': 768,
 'far': 769,
 'homeless': 770,
 'harry': 771,
 'mind': 772,
 'band': 773,
 'oil': 774,
 'reveal': 775,
 'major': 776,
 'prevent': 777,
 'blood': 778,
 'five': 779,
 'executive': 780,
 'quietly': 781,
 'voter': 782,
 'cut': 783,
 'experts': 784,
 'happens': 785,
 'industry': 786,
 'amazon': 787,
 'rally': 788,
 'side': 789,
 'hour': 790,
 'weight': 791,
 'bathroom': 792,
 'finding': 793,
 'deadly': 794,
 'officer': 795,
 'uses': 796,
 'kerry': 797,
 'also': 798,
 'marijuana': 799,
 'demand': 800,
 'proud': 801,
 'pick': 802,
 'grandma': 803,
 'doctor': 804,
 'explains': 805,
 'given': 806,
 'magazine': 807,
 'table': 808,
 'hilarious': 809,
 'dying': 810,
 'evidence': 811,
 'suggests': 812,
 'protesters': 813,
 'nyc': 814,
 'mean': 815,
 'culture': 816,
 'celebrate': 817,
 'responds': 818,
 'google': 819,
 'private': 820,
 'happened': 821,
 'families': 822,
 'humans': 823,
 'soon': 824,
 'colbert': 825,
 'avoid': 826,
 'amazing': 827,
 'guys': 828,
 'dogs': 829,
 'christian': 830,
 'reminds': 831,
 'check': 832,
 'elizabeth': 833,
 'building': 834,
 'matter': 835,
 'near': 836,
 'carolina': 837,
 'receives': 838,
 'netflix': 839,
 'shop': 840,
 'possible': 841,
 'plane': 842,
 'hero': 843,
 'likely': 844,
 'sean': 845,
 'data': 846,
 'leading': 847,
 'ben': 848,
 'shares': 849,
 'pregnant': 850,
 'sales': 851,
 'dads': 852,
 'worse': 853,
 'toward': 854,
 'economy': 855,
 'announce': 856,
 'address': 857,
 'door': 858,
 'movies': 859,
 'planet': 860,
 'picture': 861,
 'jr': 862,
 'try': 863,
 'spend': 864,
 'older': 865,
 'olympic': 866,
 'green': 867,
 'teachers': 868,
 'driving': 869,
 'healthy': 870,
 'arent': 871,
 'swift': 872,
 'sea': 873,
 'voting': 874,
 'perfectly': 875,
 'amid': 876,
 'epa': 877,
 'bowl': 878,
 'reading': 879,
 'manager': 880,
 'thats': 881,
 'card': 882,
 'dark': 883,
 'east': 884,
 'fails': 885,
 'jeb': 886,
 'hopes': 887,
 'jones': 888,
 'st': 889,
 'leads': 890,
 'hospital': 891,
 'close': 892,
 'train': 893,
 'results': 894,
 'ebola': 895,
 'arrested': 896,
 'number': 897,
 'religious': 898,
 'spring': 899,
 'hasnt': 900,
 'san': 901,
 'activists': 902,
 'energy': 903,
 'slams': 904,
 'battle': 905,
 'went': 906,
 'short': 907,
 'chicken': 908,
 'attempt': 909,
 'refugees': 910,
 'legal': 911,
 'opening': 912,
 'doctors': 913,
 'emotional': 914,
 'loss': 915,
 'fake': 916,
 'moore': 917,
 'performance': 918,
 'loses': 919,
 'learning': 920,
 'crime': 921,
 'israel': 922,
 'accidentally': 923,
 'website': 924,
 'winter': 925,
 'historical': 926,
 'hear': 927,
 'crowd': 928,
 'episode': 929,
 'stay': 930,
 'forward': 931,
 'airport': 932,
 'giant': 933,
 'jenner': 934,
 'done': 935,
 'male': 936,
 'coworkers': 937,
 'steps': 938,
 'desperate': 939,
 'telling': 940,
 'cnn': 941,
 'onion': 942,
 'martin': 943,
 'residents': 944,
 'mayor': 945,
 'daily': 946,
 'truth': 947,
 'beer': 948,
 'actor': 949,
 'driver': 950,
 'kelly': 951,
 'senators': 952,
 'color': 953,
 'despite': 954,
 'womans': 955,
 'easy': 956,
 'staff': 957,
 'promises': 958,
 'caught': 959,
 'jeff': 960,
 'killer': 961,
 'eye': 962,
 'sad': 963,
 'al': 964,
 'youth': 965,
 'reporter': 966,
 'defense': 967,
 'recalls': 968,
 'skin': 969,
 'guns': 970,
 'thanksgiving': 971,
 'rape': 972,
 'flight': 973,
 'said': 974,
 'dreams': 975,
 'scandal': 976,
 'couples': 977,
 'dnc': 978,
 'protect': 979,
 'financial': 980,
 'realizes': 981,
 'harassment': 982,
 'somehow': 983,
 'helping': 984,
 'commercial': 985,
 'faces': 986,
 'gave': 987,
 'fathers': 988,
 'board': 989,
 'fda': 990,
 'bob': 991,
 'pizza': 992,
 'committee': 993,
 'foreign': 994,
 'brown': 995,
 'threat': 996,
 'ago': 997,
 'came': 998,
 'player': 999,
 'documentary': 1000,
 ...}

# Convert training text to sequences of word indices
tokenizer.texts_to_sequences(train_text[:5])

     
[[204, 1, 740, 2953, 2235, 245, 35, 1958, 2515, 8487],
 [8488, 3234, 2422, 8489, 289, 3081, 151, 853],
 [62, 597, 719, 1063, 1840, 496, 4199, 116, 9676],
 [1320, 130, 337, 1439, 219, 3385, 1, 703],
 [4779, 5143, 679, 1, 454, 490, 1137, 33]]

# Tokenize and pad the training text data
# Convert training text to sequences of word indices
train_indices = tokenizer.texts_to_sequences(train_text)
# Pad sequences to a fixed length
train_padded = pad_sequences(train_indices,
							padding=padding_type,
							maxlen=max_length)

     

# Convert validation text to sequences of word indices
val_indices = tokenizer.texts_to_sequences(val_text)
# Pad sequences to a fixed length
validation_padded = pad_sequences(val_indices,
								padding=padding_type,
								maxlen=max_length)

# Convert test text to sequences of word indices
test_indices = tokenizer.texts_to_sequences(test_text)
# Pad sequences to a fixed length
test_padded = pad_sequences(test_indices,
							padding=padding_type,
							maxlen=max_length)

# Check
print('Training vector :', train_padded.shape)
print('Validations vector :', validation_padded.shape)
print('Testing vector :', test_padded.shape)

     
Training vector : (44262, 60)
Validations vector : (5533, 60)
Testing vector : (5533, 60)

# Decode the sample training vector
tokenizer.sequences_to_texts([train_padded[0].tolist()])

     
['<OOV> new man']

# Prepare labels for model
training_labels_final = np.array(train_label)
validation_labels_final = np.array(val_label)
testing_labels_final = np.array(test_label)


# Check shapes
print('Training vector :', training_labels_final.shape)
print('Validations vector :', validation_labels_final.shape)
print('Testing vector :', testing_labels_final.shape)

     
Training vector : (44262,)
Validations vector : (5533,)
Testing vector : (5533,)

	# Compile the model with specified loss function, optimizer, and evaluation metrics
model.compile(loss='binary_crossentropy',
			optimizer='adam', metrics=['accuracy'])

     

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Now the model will know the input shape
model.build(input_shape=(None, max_length))
model.summary()

     
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding_1 (Embedding)              │ (None, 60, 200)             │       2,000,000 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_max_pooling1d                 │ (None, 200)                 │               0 │
│ (GlobalMaxPooling1D)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 40)                  │           8,040 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 40)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 20)                  │             820 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 20)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 10)                  │             210 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 10)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (Dense)                      │ (None, 1)                   │              11 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 2,009,081 (7.66 MB)
 Trainable params: 2,009,081 (7.66 MB)
 Non-trainable params: 0 (0.00 B)

import numpy as np

max_index = np.max(train_padded)
print(f"Maximum index in train_padded: {max_index}")

     
Maximum index in train_padded: 9

vocab_size = max_index + 1  # Set vocab_size to cover all indices
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    # Add the rest of your layers
])

     

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Example values (replace with your actual data)
vocabulary_size = 5000  # Set this to the actual number of unique tokens + 1 for OOV token
embedding_dim = 128
num_classes = 2  # Set this to the number of unique classes in your classification problem

# Ensure all indices are within the valid range [0, vocabulary_size-1]
train_padded = np.array([[min(word, vocabulary_size - 1) for word in seq] for seq in train_padded])
validation_padded = np.array([[min(word, vocabulary_size - 1) for word in seq] for seq in validation_padded])

# Convert labels to NumPy arrays as well
training_labels_final = np.array(training_labels_final)
validation_labels_final = np.array(validation_labels_final)

# Build the model
model = Sequential()

# Embedding layer
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=train_padded.shape[1]))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set the number of training epochs
num_epochs = 5

# Train the model
history = model.fit(
    train_padded, training_labels_final,
    epochs=num_epochs,
    validation_data=(validation_padded, validation_labels_final)
)

print("Model training completed successfully.")

     
Epoch 1/5
1384/1384 ━━━━━━━━━━━━━━━━━━━━ 14s 9ms/step - accuracy: 0.7344 - loss: 0.5026 - val_accuracy: 0.8975 - val_loss: 0.2569
Epoch 2/5
1384/1384 ━━━━━━━━━━━━━━━━━━━━ 20s 9ms/step - accuracy: 0.9331 - loss: 0.1819 - val_accuracy: 0.9463 - val_loss: 0.1551
Epoch 3/5
1384/1384 ━━━━━━━━━━━━━━━━━━━━ 21s 9ms/step - accuracy: 0.9728 - loss: 0.0852 - val_accuracy: 0.9667 - val_loss: 0.1171
Epoch 4/5
1384/1384 ━━━━━━━━━━━━━━━━━━━━ 13s 9ms/step - accuracy: 0.9862 - loss: 0.0459 - val_accuracy: 0.9731 - val_loss: 0.1007
Epoch 5/5
1384/1384 ━━━━━━━━━━━━━━━━━━━━ 13s 9ms/step - accuracy: 0.9922 - loss: 0.0266 - val_accuracy: 0.9790 - val_loss: 0.0976
Model training completed successfully.

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

# Plot validation loss
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss',color='black')
ax1.set_title('Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot validation accuracy
ax2.plot(history.history['accuracy'], label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='black')
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

     


import numpy as np

# Ensure test_padded and testing_labels_final are NumPy arrays
test_padded = np.array([[min(word, vocabulary_size - 1) for word in seq] for seq in test_padded])
testing_labels_final = np.array(testing_labels_final)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_padded, testing_labels_final)

# Print the accuracy on the test dataset
print(f'Accuracy on test dataset: {round(accuracy * 100, 2)}%')


     
173/173 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9787 - loss: 0.0953
Accuracy on test dataset: 97.8%

# Assuming pred_prob is 2D with shape (n_samples, 1)
pred_prob = model.predict(test_padded).flatten()

# Apply threshold to convert probabilities to binary labels
pred_label = [1 if prob >= 0.5 else 0 for prob in pred_prob]

# Print the first 5 predicted labels
print(pred_label[:5])

     
173/173 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step
[0, 1, 0, 1, 0]

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Make predictions
pred_prob = model.predict(test_padded).flatten()

# Convert probabilities to binary labels
pred_label = [1 if prob >= 0.5 else 0 for prob in pred_prob]

# Ensure both arrays have the same length
if len(pred_label) != len(testing_labels_final):
    # Truncate pred_label to match the length of testing_labels_final if necessary
    pred_label = pred_label[:len(testing_labels_final)]

# Compute the confusion matrix
conf_matrix = confusion_matrix(testing_labels_final, pred_label)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Not Sarcastic', 'Sarcastic'],
            yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

     
173/173 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step


	# Print Classification Report
print("\nClassification Report:")
print(classification_report(testing_labels_final, pred_label,
							target_names=['Not Sarcastic', 'Sarcastic']))

     
Classification Report:
               precision    recall  f1-score   support

Not Sarcastic       0.52      0.50      0.51      2916
    Sarcastic       0.47      0.49      0.48      2617

     accuracy                           0.50      5533
    macro avg       0.50      0.50      0.49      5533
 weighted avg       0.50      0.50      0.50      5533


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Assuming max_len is the length the model was trained with
max_len = 120  # Adjust this to match the length used during training

# Function to predict sarcasm
def predict_sarcasm(model, tokenizer, text, max_len):
    # Step 1: Tokenize the input text
    input_seq = tokenizer.texts_to_sequences([text])

    # Step 2: Pad the input text to the required max_len
    padded_input = pad_sequences(input_seq, maxlen=max_len, padding='post')

    # Debug: Print the shape of the padded input
    print(f"Padded input shape: {padded_input.shape}")

    # Step 3: Predict sarcasm using the model
    prediction = model.predict(padded_input)

    # Debug: Print the prediction value
    print(f"Prediction: {prediction}")

    # Step 4: Convert the prediction to binary label (1 = Sarcastic, 0 = Not Sarcastic)
    pred_label = 1 if prediction[0] >= 0.5 else 0
    return pred_label

# Example usage
while True:
    headline = input("Enter a headline for prediction (or type 'exit' to quit): ")
    if headline.lower() == 'exit':
        break
    label = predict_sarcasm(model, tokenizer, headline, max_len)
    print("Sarcastic" if label == 1 else "Not Sarcastic")

     
Enter a headline for prediction (or type 'exit' to quit): " suicide of a hacker"
Padded input shape: (1, 120)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
Prediction: [[0.48946118]]
Not Sarcastic
Enter a headline for prediction (or type 'exit' to quit): exit

from google.colab import drive
drive.mount('/content/drive')
     
Mounted at /content/drive

!mkdir -p "/content/drive/MyDrive/output_data/"
!cp "/content/drive/MyDrive/Final Project NLP/sarcasm nlp" "/content/drive/MyDrive/output_data/"

     

!ls "/content/drive/MyDrive/output_data/"

     
'sarcasm nlp'

with open("/content/drive/MyDrive/output_data/sarcasm nlp", 'r') as file:
    content = file.read()
    print(content)

     

!pip install streamlit

     
Collecting streamlit
  Downloading streamlit-1.37.1-py2.py3-none-any.whl.metadata (8.5 kB)
Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)
Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)
Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.4.0)
Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)
Requirement already satisfied: numpy<3,>=1.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)
Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.1)
Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.1.4)
Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)
Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)
Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (14.0.2)
Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.32.3)
Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.7.1)
Collecting tenacity<9,>=8.1.0 (from streamlit)
  Downloading tenacity-8.5.0-py3-none-any.whl.metadata (1.2 kB)
Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)
Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)
Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)
  Downloading GitPython-3.1.43-py3-none-any.whl.metadata (13 kB)
Collecting pydeck<1,>=0.8.0b4 (from streamlit)
  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)
Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)
Collecting watchdog<5,>=2.1.5 (from streamlit)
  Downloading watchdog-4.0.2-py3-none-manylinux2014_x86_64.whl.metadata (38 kB)
Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)
Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)
Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)
Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)
  Downloading gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)
Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.7.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)
Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)
  Downloading smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.5)
Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.2.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)
Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)
Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.20.0)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)
Downloading streamlit-1.37.1-py2.py3-none-any.whl (8.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 21.9 MB/s eta 0:00:00
Downloading GitPython-3.1.43-py3-none-any.whl (207 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.3/207.3 kB 11.3 MB/s eta 0:00:00
Downloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.9/6.9 MB 35.8 MB/s eta 0:00:00
Downloading tenacity-8.5.0-py3-none-any.whl (28 kB)
Downloading watchdog-4.0.2-py3-none-manylinux2014_x86_64.whl (82 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 82.9/82.9 kB 3.2 MB/s eta 0:00:00
Downloading gitdb-4.0.11-py3-none-any.whl (62 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.7/62.7 kB 3.5 MB/s eta 0:00:00
Downloading smmap-5.0.1-py3-none-any.whl (24 kB)
Installing collected packages: watchdog, tenacity, smmap, pydeck, gitdb, gitpython, streamlit
  Attempting uninstall: tenacity
    Found existing installation: tenacity 9.0.0
    Uninstalling tenacity-9.0.0:
      Successfully uninstalled tenacity-9.0.0
Successfully installed gitdb-4.0.11 gitpython-3.1.43 pydeck-0.9.1 smmap-5.0.1 streamlit-1.37.1 tenacity-8.5.0 watchdog-4.0.2

!wget -q -O - ipv4.icanhazip.com
     
34.136.250.55

from google.colab import drive
drive.mount('/content/drive')


     
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

!ls "/content/drive/MyDrive/Final Project NLP/.ipynb_checkpoints/"

     

!ls "/content/drive/MyDrive/Final Project NLP/"

     
'sarcasm nlp'
