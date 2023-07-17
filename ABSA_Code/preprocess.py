import pickle 
import re
from vncorenlp import VnCoreNLP
from nltk import flatten

tokenizer = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

VN_CHARS_LOWER = u'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð'
VN_CHARS_UPPER = u'ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸÐĐ'
VN_CHARS = VN_CHARS_LOWER + VN_CHARS_UPPER
def no_marks(s):
    __INTAB = [ch for ch in VN_CHARS]
    __OUTTAB = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"*2
    __OUTTAB += "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"*2
    __r = re.compile("|".join(__INTAB))
    __replaces_dict = dict(zip(__INTAB, __OUTTAB))
    result = __r.sub(lambda m: __replaces_dict[m.group(0)], s)
    return result

replace_list = pickle.load(open('./data/replace.pkl','rb'))

def text_preprocess(text):
    check = re.search(r'([a-z])\1+',text)
    if check:
          if len(check.group())>2:
            text = re.sub(r'([a-z])\1+', lambda m: m.group(1), text, flags=re.IGNORECASE) #remove các ký tự kéo dài như hayyy, ngonnnn...

          text = text.strip() #loại dấu cách đầu câu

          for k, v in replace_list.items():       #replace các từ có trong replace_list
            text = text.replace(k, v)
          text = re.sub('[!”"#$%&’()•/:;<=>-?@[\]^`{|}~+*_-]', ' ', text) #special character
          text = re.sub('\d+k', '', text) #300k, 200k
          text = re.sub('\d+', '', text) #number
          text = re.sub('(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text) #web address
          text = re.sub('www\.\S+\.com', ' ', text) #web address
          text = re.sub('@\S+', ' ', text) #user mention
          text = re.sub('[0-9]k', ' ', text)
          emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
          text = re.sub(emoji_pattern, '', text)

          text = ' '.join(i for i in flatten(tokenizer.tokenize(text)))
          text = text.lower()
    return text

def preprocess(data):
    token = []
    token_nomarks = []
    for text in data:
        check = re.search(r'([a-z])\1+',text)
        if check:
          if len(check.group())>2:
            text = re.sub(r'([a-z])\1+', lambda m: m.group(1), text, flags=re.IGNORECASE) #remove các ký tự kéo dài như hayyy, ngonnnn...

        text = text.strip() #loại dấu cách đầu câu
      
        for k, v in replace_list.items():       #replace các từ có trong replace_list
          text = text.replace(k, v)       
        text = re.sub('[!”"#$%&’()•/:;<=>-?@[\]^`{|}~+*_-]', ' ', text) #special character
        text = re.sub('\d+k', '', text) #300k, 200k 
        text = re.sub('\d+', '', text) #number
        text = re.sub('(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text) #web address
        text = re.sub('www\.\S+\.com', ' ', text) #web address
        text = re.sub('@\S+', ' ', text) #user mention
        text = re.sub('[0-9]k', ' ', text)
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        text = re.sub(emoji_pattern, '', text)
        
        text = ' '.join(i for i in flatten(tokenizer.tokenize(text)))
        text = text.lower()             
        text_nomarks = token_nomarks.append(no_marks(text))
        token.append(text)
    return token, token_nomarks
