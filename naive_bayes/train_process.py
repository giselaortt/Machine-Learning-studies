from pathlib import Path
import re
import os
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


folder_input_path = '20_newsgroups/'
folder_output_name = 'dados_processados/'


#This function will concatenate every file in directory and save in a file.
def concatenate( file_name, directory_name ):
    files = os.listdir( directory_name )
    fpointer = open( file_name, 'w' )
    for name in files:
        temp = open( os.path.join(directory_name, name), "r" )
        for line in temp.readlines():
            fpointer.write( line.strip('[^><#%*()!$,/\]') )
        temp.close()
    fpointer.close()


#This function will split the files in train and test sets, acording to the percentage that was passed
def split_files( source_dir, train_dir_name, test_dir_name, train_size = 0.7 ):
    file_names = os.listdir( source_dir )
    train_files, test_files = train_test_split(file_names,train_size = train_size)    
    #copy training set
    for name in train_files:
        os.system(' cp '+ os.path.join(source_dir, name) + ' ' + os.path.join(train_dir_name, name) )
    #copy testing set
    for name in test_files:
        os.system(' cp '+ os.path.join(source_dir, name) + ' ' + os.path.join(train_dir_name, name) )
    return


#this function should remove things that are not relevant for naive bays algorithm
def clean_text( file_name ):

    filetemp = open( ,'w')
    arquivo = open(file_name,'r')
    text = file.read()

    #converter tudo para low-case letter
    text.lower()

    #remove stop-words: a, the, of, for etc.
    text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])

    #TODO: extração de radicais

    #remove ponctuation
    text.strip('[.^><#%&*()!$/\]')

    #remove numbers
    #tentar também: re.sub('\d', '', text) ou "\d+"
    text = re.sub('[0,9]', '', text)

    #filetemp e voltar para o file original, que é deletado
    os.system('rm '+file_name)
    os.system('mv ' + filetempname + ' ' + file_name )


def train_prepare( folder_input_name, folder_output_name, training_size = 0.7 ):
    
    classes = os.listdir(folder_input_name)

    #cria o diretorio de output caso ele nao exista.
    os.mkdir("dados_processados/")
    os.mkdir("dados_concatenados/")
    os.mkdir("dados_concatenados/train/")
    os.mkdir("dados_concatenados/test/")

    for entry in classes:

        #cria os diretorios
        train_dir = "dados_processados/"+entry+"/train/"
        test_dir = "dados_processados/"+entry+"/test/"
        os.mkdir("dados_processados/"+entry+"/")
        os.mkdir(train_dir)
        os.mkdir(test_dir)

        #separa os testes e os treinamentos
        split_files(os.path.join(folder_input_name, entry), train_dir, test_dir, train_size = training_size)

        #limpar os textos
        for name in os.listdir( train_dir ):
            clean_text( os.path.join(train_dir, name))

        for name in os.listdir( test_dir ):
            clean_text( os.path.join(test_dir, name))
 
        #concatenar os dados
        concatenate( test_dir, "dados_concatenados/train/" + entry + ".txt" )
        concatenate( test_dir, "dados_concatenados/test/" + entry + ".txt" )
        

trai_prepare( folder_input_path , folder_output_name)

