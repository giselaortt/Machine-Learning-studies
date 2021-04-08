from pathlib import Path
import re
import os
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

#best stemmer from the tested versions were poter stemmer
from nltk.stem import PorterStemmer
#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.lancaster import LancasterStemmer
#from nltk.stem.api import StemmerI


folder_input_path = '20_newsgroups/'


#This function will concatenate every file in directory and save in a file.
def concatenate( file_name, directory_name ):
    files = os.listdir( directory_name )
    fpointer = open( file_name, 'w' )
    for name in files:
        temp = open( os.path.join(directory_name, name), "r" )
        for line in temp.readlines():
            fpointer.write( line.strip('[\"^><#%*()!$,/\]') )
        temp.close()
    fpointer.close()


#This function will split the files in train and test sets, acording to the percentage that was passed
def split_files( source_dir, train_dir_name, test_dir_name, train_size = 0.7 ):
    file_names = os.listdir( source_dir )
    train_files, test_files = train_test_split(file_names,train_size = train_size)    
    #copy training set
    for name in train_files:
        os.system('cp '+ os.path.join(source_dir, name) + ' ' + os.path.join(train_dir_name, name) )
        #print('cp '+ os.path.join(source_dir, name) + ' ' + os.path.join(train_dir_name, name)
    #copy testing set
    for name in test_files:
        os.system('cp '+ os.path.join(source_dir, name) + ' ' + os.path.join(test_dir_name, name) )
    return



#this function should remove things that are not relevant for naive bays algorithm
def clean_text( file_name ):

    filetempname = file_name.rstrip('.txt') + 'temp'
    filetemp = open( filetempname,'w')
    arquivo = open(file_name,'r', encoding = "ISO-8859-1" )
    text = arquivo.read()
    lines = text.split('\n')

    #remover todos os emails, datas e url's
    #deletando as descrições da base de dados. essa etapa não é essencial, mas quero fazer mesmo asism.
    lines_to_delete = [
	"Xref:",
        "Approved:",
	"Path:",
	"From:",
        "Reply-To:",
	"Newsgroups:",
	"Subject:",
	"Message-ID:",
	"Date:",
	"Article-I.D.:",
	"Sender:",
	"Distribution:",
	"Organization:",
	"Lines:",
	"Nntp-Posting-Host:",
    ]

    to_delete = []
    for i, line in zip( range(len(lines)),lines ):
        words = line.strip('\n\r\f\t\b').split(' ')
        if words[0] in lines_to_delete:
            to_delete.append(line)

    for i in to_delete:
        lines.remove(i)

    text = ' '.join(lines)

    #converter tudo para low-case letter
    text.lower()

    #remove stop-words: a, the, of, for etc.
    #pega apenas o radical de cada palavra
    #apaga todos os emails
#    st = SnowballStemmer('english')
    st = PorterStemmer()
    text = ' '.join([st.stem(word)+' ' for word in text.split(' ') if(word not in stopwords.words('english') and '@' not in word)])    

    #remove ponctuation
    text = re.sub("([^a-z \n])", '', text)

    #remove numbers
    #tentar também: re.sub('\d', '', text) ou "\d+"
    #text = re.sub('[0,9]', '', text)

    filetemp.write( text )
    filetemp.close()
    arquivo.close()
    #filetemp e voltar para o file original, que é deletado
    os.system('rm '+ file_name)
    os.system('mv ' + filetempname + ' ' + file_name )


def train_prepare( folder_input_name, training_size = 0.7 ):

    classes = os.listdir(folder_input_name)

    #cria o diretorio de output caso ele nao exista.
    try:
        os.mkdir("dados_processados/")
        os.mkdir("dados_concatenados/")
        os.mkdir("dados_concatenados/train/")
        os.mkdir("dados_concatenados/test/")
    except:
        pass

    for entry in classes:
        #cria os diretorios
        train_dir = "dados_processados/"+entry+"/train/"
        test_dir = "dados_processados/"+entry+"/test/"
        try:
            os.mkdir("dados_processados/"+entry+"/")
            os.mkdir(train_dir)
            os.mkdir(test_dir)
        except:
            pass

        #separa os testes e os treinamentos
        print(os.path.join(folder_input_name, entry))
        split_files(os.path.join(folder_input_name, entry), train_dir, test_dir, train_size = training_size)

        #limpar os textos
        for name in os.listdir( train_dir ):
            clean_text( os.path.join(train_dir, name))

        for name in os.listdir( test_dir ):
            clean_text( os.path.join(test_dir, name))

        #concatenar os dados
        concatenate( "dados_concatenados/train/" + entry + ".txt", train_dir )
        concatenate( "dados_concatenados/test/" + entry + ".txt", test_dir )

train_prepare( folder_input_path )

