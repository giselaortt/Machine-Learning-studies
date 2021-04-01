
from pathlib import Path
import os

#TODO: how to list all the subdirectories in a directory
# List all subdirectories using os.listdir

#TODO: how to list all files in a folder



#TODO: how to concatenate every file in folder


folder_input_path = '20_newsgroups/'
folder_output_name = 'nb_dados_processados/'

def train_prepare( folder_input_name, folder_output_name, training_size = 0.7 ):
    subdiretorios = []
    #Le todos os subdiretorios e adiciona em uma lista
    for entry in os.listdir(folder_input_name):
        if os.path.isdir(os.path.join(folder_input_name, entry)):
            subdiretorios.append( entry )

    #cria o diretorio de output caso ele nao exista.
    try:
        os.mkdir(folder_output_name + '/')
    except FileExistsError as exc:
        pass

    for entry in subdiretorios:
        #TODO: create a file names entry
        f = open( folder_output_name+'/'+entry, 'w' )
        for lines in os.listdir( os.path.join(folder_input_name, entry) ):
        #TODO print lines in the file
            f.write(lines)
        f.close()


train_prepare( folder_input_path , folder_output_name)
