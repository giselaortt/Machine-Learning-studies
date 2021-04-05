from pathlib import Path
import os

#TODO: parsear caracteres especiais: %@#/n etc

folder_input_path = '20_newsgroups/'
folder_output_name = 'dados_processados/'

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
        f_processado = open( folder_output_name+'/'+entry, 'w' )
        for name in os.listdir( os.path.join(folder_input_name, entry) ):
            for line in open(os.path.join(folder_input_name, entry, name),'r', encoding="ISO-8859-1" ):
                f_processado.write( line )
        f_processado.close()


train_prepare( folder_input_path , folder_output_name)
