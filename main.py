# ATENÇÃO - ANTES DE EXECUTAR O ARQUIVO, POR FAVOR DISPONIBILIZAR A PASTA "chihuahua-muffin" CONTENDO AS IMAGENS DOS CHIHUAHUAS E MUFFINS PARA DENTRO DO ROOT DO PROJETO. ESSA PASTA ESTA NO REPOSITORIO DO GITHUB, QUE PODE SER ACESSADO COM O LINK QUE FORA DISPONIBILIZADO PARA A REALIZAÇÃO DESTE TESTE.

# Obs: Espero que não incomode meu código em ingles, mas os comentários em portugues. Acostumei a codar em ingles e nao tenho costume de comentários em codigo...

# 1. Descreva o pré-processamento utilizado.
# O pre-processamento utilizado neste projeto esta explicado ao decorrer do codigo. Não sigo muito a prática de comentários em códigos, mas deixei comentado tanto para explicar quanto para tambem entender e absorver o conhecimento do que realizei aqui. Todo o precho do pre-processamento utilizado esta na sessão ---- INICIO PRE-PROCESSAMENTO que vai até o ----- FIM PRE-PROCESSAMENTO -----.


# 2. Quais métricas para análise de performance foram utilizadas? Por quê? Como o modelo performou?
# Utilizei o Accuracy para medir a proporção de previsões. Matriz de confusão para visualizar o desempenho do modelo em termos de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos e Relatorio de classificação de inclui precisão, recall e F1-Score.
# Eu utilizei essas métricas porque elas em conjunto me ajudaram a ter uma visão mais detalhada do desempenho do modelo de classificação. Por exemplo, eu tive muito problema para entender por que estava tendo tanta falha de accuracy, após implementar a visualização do relatório. Analisando a acurácia de treino e validação e tambem a perda de treino e validação, eu pude perceber que a rede neural estava com um problema de overfitting e então isso me fez mudar de approach, chegando a um resultado positivo.

# 3. Como melhoraria a assertividade do modelo?
# Aumentando a base de dados seria uma opção. Tambem se eu não fosse tão limitado em questão de hardware, eu tentaria novos modelos pré-treinados, redes mais complexas ou desenvolveria minha propria rede para obter um bom resultado. Tentaria usar o VGG19 que é muito custoso em processamento computacional tambem.

# 4. Teve dificuldade em alguma parte específica? Se tivesse mais tempo, mudaria algo?
# Ao tentar fazer minha propria rede eu senti uma dificuldade muito grande. Obtive muitos overffitings, então eu perdi muito tempo tentando ajustar muitos hiperametros para que encaixassem de uma forma ideal, mas não tive sucesso. Se eu tivesse mais tempo eu tentaria como mencionei acima explorar mais tecnicas avançadas de pré-processamento e aumento de dados. Tambem me aprofundaria mais em mais conceitos e tentaria fazer um algoritimo para automatizar e rodar testes baseado nos resuldados de acurácia e perca, fazendo alterar alguns parametros no treino do modelo e pre-processamento até encontrar o melhor resultado...

# 5. Utilizou algum modelo pré-treinado ou desenvolveu a rede neural do zero? Justifique
# sua escolha. Se optou por um pré-treinado, por que esse modelo, especificamente?
# Eu usei o MobileNetV2 que se mostrou muito rapido e eficiente nos termos de computação e ele é muito útil para tarefas que exigem processamento rápido e eficiente de recursos, ainda mais com uma base pequena de dados como essa que foi disponibilizada. Tentei outros modelos e tentei construir um do 0, mas nao obtive muito sucesso.


from keras.src.applications.mobilenet_v2 import MobileNetV2
import os
import glob
import random
import shutil
import keras
from keras.src.callbacks import ModelCheckpoint
from keras.src.layers.preprocessing.rescaling import Rescaling
from keras.src.layers import Flatten, Dense, Dropout
from keras.src.utils import image_dataset_from_directory
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import itertools
from keras.api.preprocessing.image import img_to_array, load_img

# -------------------- INICIO PRE-PROCESSAMENTO -------------------- #

# Capturando diretório do projeto independente de OS (uso o linux-ubuntu)
current_dir = os.path.abspath(os.getcwd())

# Definição da estrutura pastas que o tensorflow precisa
folder = os.path.join(current_dir, 'dataset')
train_folder = os.path.join(folder, 'train')
val_folder = os.path.join(folder, 'validation')
test_folder = os.path.join(folder, 'test')

# Verificação para sempre deixar atualizado o diretorio de pastas da estrutura do tensorflow
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)

# Conjunto de dados (dataset) que se encontra dentro do .zip fornecido para este case. ATENÇÃO -> Necessita ser movido a pasta "chihuahua-muffin" para o root do projeto antes do codigo ser executado.
img_base_folder = os.path.join(current_dir, 'chihuahua-muffin')

# Funções auxiliares para calculo de proporção e redirecionamento de arquivos para as pastas corretas


def move_files(files, train_folder, val_folder, test_folder):
    # Neste trecho abaixo eu utilizei o trai_test_split da lib sklearn para poder criar divisões consistentes de dados pelo fato de que estamos trabalhando apenas com 16 imagens, o que é pouco e pode acarretar no overfitting.
    train_files, temp_files = train_test_split(
        files, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(
        temp_files, test_size=0.5, random_state=42)

    # Renomeia os arquivos e copia eles até a pasta de destino.
    def move_to_folder(file_list, dest_folder):
        for _, file in enumerate(file_list):
            filename = file.split("/")[-1]
            shutil.copy(file, os.path.join(dest_folder, filename))

    # Chama a função para renomear e enviar os arquivos para a pasta de destino.
    move_to_folder(train_files, train_folder)
    move_to_folder(val_files, val_folder)
    move_to_folder(test_files, test_folder)

# Função para capturar as imagens e passá-las para as funções de redirecionamento


def moveImagesToCorrectFolder():

    # Arquivos de muffins
    muffin_img_files = glob.glob(os.path.join(img_base_folder, 'muffin*.jpeg'))

    # Arquivos de chihuahuas
    chihuahua_img_files = glob.glob(
        os.path.join(img_base_folder, 'chihuahua*.jpg'))

    random.shuffle(muffin_img_files)
    random.shuffle(chihuahua_img_files)

    # Abaixo é definido o path (caminho) para onde os arquivos serão enviados
    train_chihuahuas_folder = os.path.join(folder, 'train', 'chihuahuas')
    val_chihuahuas_folder = os.path.join(folder, 'validation', 'chihuahuas')
    test_chihuahuas_folder = os.path.join(folder, 'test', 'chihuahuas')

    train_muffins_folder = os.path.join(folder, 'train', 'muffins')
    val_muffins_folder = os.path.join(folder, 'validation', 'muffins')
    test_muffins_folder = os.path.join(folder, 'test', 'muffins')

    # Cria os diretórios se eles não existirem.
    os.makedirs(train_chihuahuas_folder, exist_ok=True)
    os.makedirs(val_chihuahuas_folder, exist_ok=True)
    os.makedirs(test_chihuahuas_folder, exist_ok=True)

    os.makedirs(train_muffins_folder, exist_ok=True)
    os.makedirs(val_muffins_folder, exist_ok=True)
    os.makedirs(test_muffins_folder, exist_ok=True)

    # Chama a função para mover os arquivos. Aqui passo os arquivos no primeiro parametro, o caminho do aquivo de treino no segundo, o caminho do aquivo de validação no terceiro e o de teste por ultimo. Lembrando que é necessário essa estrutura para o tensorflow.
    move_files(chihuahua_img_files, train_chihuahuas_folder,
               val_chihuahuas_folder, test_chihuahuas_folder)

    move_files(muffin_img_files, train_muffins_folder,
               val_muffins_folder, test_muffins_folder)


# Chama a função principa para arquivos e diretorios.
moveImagesToCorrectFolder()


#  Aqui eu carrego eu o metodo do image_dataset_from_directory do tensorflow. Passo o caminho da pasta dos arquivos de treino, validação ou teste, o tamanho da imagem redimencionada (em redes neurais convulocionais, todas as imagens precisam ter o mesmo tamanho para que possa ser feito o reconhecimento). Eu usei 128 pq as imagens eram simples e pequenas e me deram bons resultados. Tambem é passado o batch size ou tamanho do lote, que eu defini para ser 2. Dai entra a divisão dos lotes. Se eu tenho para treino 8 imagens, vou ter 4 lotes cada um com 2 imagens. Na validação vou ter 4 imagens, vou ter 2 lotes e cada um com 2 imagens e o mesmo para teste. Dai em cada iteração do treinamento o modelo vai processar 2 imagens de uma só vez. Ter usado apenas 2 batches me ajudou a resolver um problema de overfitting que estava tendo (tem mais algumas outras coisas mais abaixo que usei para prevenir o overfitting) e tambem nos recursos de hardwares. Como meu notebook é AMD, tanto em processador quanto em GPU, eu nao tinha CUDAS para usar. Resumindo, foi uma escolha sensata. Ao aumentar os batches, consequentemente eu tinha um resultado de redução drástica na qualidade de treino do modelo.

# Existem configurações especificas para AMD, e seria uma oportunidade de expliração e implementação se eu tivesse mais tempo.
train_dataset = image_dataset_from_directory(
    train_folder, image_size=(128, 128), batch_size=2)

validation_dataset = image_dataset_from_directory(
    val_folder, image_size=(128, 128), batch_size=2)

test_dataset = image_dataset_from_directory(
    test_folder, image_size=(128, 128), batch_size=2, shuffle=False)

# -------------------- FIM PRE-PROCESSAMENTO -------------------- #


# -------------------- INICIO MODELO -------------------- #
def train_model():
    # Antes de optar por um modelo pré treinado eu fiz algunas pesquisas e cheguei a testar alguns modelos. O VGG19 mesmo, ele consumia muitos recurso de hardware e senti algumas dificuldades em fazer os ajustes necessários. Tambem estava obtendo muito overfitting, mesmo fazendo o data_augmentation. Então vi o MobileNetV2 que se mostrou muito rapido e eficiente nos termos de computação e ele é muito útil para tarefas que exigem processamento rápido e eficiente de recursos, ainda mais com uma base pequena de dados como essa que foi disponibilizada. Também tentei fazer a rede do 0 utilizando alguns metodos do proprio keras, mas obtive resultados ruins mostrados no grafico de perca de treino e validação. Acho que o uso de alguns modelos pre treinados evita o processo de "Recriar a roda". Usei a API Sequencial do Keras por ser simples e facil de configurar e depois transferi o aprendizado do modelo do MobileNet ja que ele tem um grande conjunto de dados que vem do imagenet.

    # configuração para o modelo pre-treinado
    base_model = MobileNetV2(input_shape=(128, 128, 3),
                             include_top=False,
                             weights='imagenet')

    base_model.trainable = False

    # Configuração de Data Augmentation para evitar o overfitting. Como eu nunca havia treinado uma rede neural, foi dificil para entender e achar o motivo de não estar obtendo bons resultados e essa configuração abaixo foi o que ajudou. Resumindo, como eu tenho uma pequena base de dados, ele cria novas imagens por debaixo dos panos baseadas nas imagens ja existentes, aplicando filtros para aumentar as comparações e ser mais assertivo.
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
    ])

    # A camada Rescaling pega as cores das imagens e representa elas entre 0 e 1. Ele faz isso pegando todas as cores rgb da imagem e divide por 255. Isso ajuda nos calculos e ajuda a convergir melhor. Isso é uma boa prática e garante um resultado melhor nao só em redes neurais convolucionais, mas sim em todo algoritimo de deep learning. Depois passei o data_aumentation para evitar o overfitting e aumentando a diversidade do treinamento. Passei o modelo base que será do mobilenet, trazendo os pesos pre-treinados do imageNet. Depois eu adicionei o flatten para vetorizar a imagem onde cada posição vai ser os pixels, transformando em um mapa de caracteristica bidimensional da saida do mobilenet para um vetor unidimensional peparado para a camada densa que espera um vetor unidimensional como entrada. Adicionei uma camada densa com 128 neuronios com ativação reLu que vai permitir que o operador de convolução percorra a imagem e extraia as informações necessárias. Adicionei o Dropout que vai aleatoriamente derrubar metade dos neuronios durante o treinamento (inclusive há um artigo publicado por Geoffrey Hinton que fala sobre o Dropout e a ideia que ele teve para criar isso. Resumindo... ele ia a um banco e sempre que via pessoas nos caixas eram diferentes, elas eram substituidas frequentemente, e dai depois de um tempo refletindo e sem a resposta para isso, ele chegou numa conclusão que aquilo provavelmente tinha haver com fraudes. Pq se um funcionario descobre algo fragil no sistema, ele literalmente poderia passar para uma pessoa de fora e etc. Então o Dropout é basicamente isso, para manter a rotação dos neuronios para que todos eles possam tomar decisões e não fiquem tão dependentes uns dos outros). Depois disso finalizo com uma camada densa com apenas 1 neuronio e função de ativação sigmoid. Ele vai produzir uma saida binaria indicando a probabilidade da imagem, informando se ela vai pertencer a uma das classes, neste caso sendo 0 para chihuahua e 1 para muffin.

    model = keras.Sequential([
        Rescaling(scale=1.0/255),
        data_augmentation,
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    # Para compilar o modelo eu escolhi configurações mais comumente utilizadas nesses tipos de treinamentos de redes neurais. A função de perda eu usei o binary_crossentropy que vai calcular a diferença entre as saidas previstas pelo modelo e as saidas reais, punindo os erros de classificação. O otimizador Adam para atualizar os pessos do modelo durante, se adaptar dinamicamente as taxas de aprendizado e ajuda a convergir mais rapido durante o treinamento. Eu usei a métrica de accuracy que vai definir as metricas usadas pra avaliar o desempenho do modelo durante o treino e a validação e dai posteriormente vou poder ver isso nos graficos gerados para ver o se o modelo performou bem ou nao. Tambem medir a proporção de previsões corretas sobre o total de previsões.
    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


#  DEFINIÇÃO/TREINO DO MODELO
model = train_model()

# O metodo de ModelCheckpoint é para salvar o melhor modelo do treino (save_best_only) e sempre quue a perda de validação melhora que seria o val_loss. Isso vai garantir o melhor modelo, previnindo o overfitting.
callbacks = [ModelCheckpoint(
    filepath="model5.keras", save_best_only=True, monitor="val_loss")]

# Aqui é a hora de brilhar. Onde acontece o treinamento inicial! O modelo começa usando os dados de treino e validação por 30 épocas. Aqui as camadas da rede neural são ajustadas para aprender os padroes a partir dos dados passados e os callbacks garantem que o melhor modelo do treino seja salvo, como mencionado acima.
history = model.fit(train_dataset, epochs=30,
                    validation_data=validation_dataset, callbacks=callbacks)

# Depois do treinamento inicial com as camadas congeladas, eu tornei o modelo completamente treinavel para realizar mais um ajuste fino de 10 épocas. Isso vai permitir que todas as camadas sejam atualizadas.
model.trainable = True

# Compila mais uma vez o modelo, porem com uma taxa de aprendizado menor de (1e-5 ou 0.00001) durante o ajuste fino, permitindo que o modelo só faça pequenos ajustes nos pesos, refinando o aprendizado sem perder os padrões que ele já pegou antes.
model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(1e-5), metrics=["accuracy"])

# Ajuste Fino de 10 épocas
history_fine = model.fit(train_dataset, epochs=10,
                         validation_data=validation_dataset, callbacks=callbacks)
# -------------------- FIM MODELO --------------------#

# Resultado do desempenho do modelo nos dados de teste. Isso vai fornecer uma métrica imparcial do desempenho do modelo depois do treinamento, indicando o loss (perda) e a accuracy (precisão)
loss, accuracy = model.evaluate(test_dataset)
print(f'Loss: {loss}, Accuracy: {accuracy:.3f}')

# Previsões no conjunto de dados de teste e avaliações do resultados
predictions = model.predict(test_dataset)

# Convertendo a previsão da probabilidade em classes binarias (0 ou 1). Que serão utilizadas para fazer a classificação baseado nas classes do teste (0 chihuahua - 1 muffin)
predicted_classes = np.where(predictions > 0.5, 1, 0)

# Obtendo Classes do teste (0 chihuahua - 1 muffin)
true_classes = np.concatenate([y for x, y in test_dataset], axis=0)

# Gera o relatorio que vai incluir a precisão, recall e F1-score para cada classe
report = classification_report(
    true_classes, predicted_classes, target_names=['chihuahuas', 'muffins'])


#  Gera a matriz de confusão para visualizar os acertos e erros do modelo.
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Os trechos mencionados acima de geração da matriz de confusão e o relatorio que inclui as precisões, recall e f1-score me ajudaram a definir e a perder menos tempo nos ajustes necessários para um maior acerto e evitar o overfitting.


# Funções para gerar o grafico da matriz de confusão e relatorio de classificação.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predict class')
    plt.show(block=False)


def plot_training_history(history, history_fine=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r', label='Train accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    if history_fine:
        acc_fine = history_fine.history['accuracy']
        val_acc_fine = history_fine.history['val_accuracy']
        plt.plot(range(1, len(acc_fine) + 1), acc_fine, 'ro',
                 label='Train Accuracy (fine-tuning)')
        plt.plot(range(1, len(val_acc_fine) + 1), val_acc_fine,
                 'r', label='Validation accuracy (fine-tuning)')
    plt.title('Train and Validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Train loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    if history_fine:
        loss_fine = history_fine.history['loss']
        val_loss_fine = history_fine.history['val_loss']
        plt.plot(range(1, len(loss_fine) + 1), loss_fine,
                 'ro', label='Train loss (fine-tuning)')
        plt.plot(range(1, len(val_loss_fine) + 1), val_loss_fine,
                 'r', label='Validation loss (fine-tuning)')
    plt.title('Train and validation loss')
    plt.legend()

    plt.show(block=False)


# Chamada de funções
plot_confusion_matrix(conf_matrix, classes=['chihuahuas', 'muffins'])
plot_training_history(history)

# Função pra checar se o valor é inteiro


def is_valid_int(input_str):
    try:
        int(input_str)
        return True
    except ValueError:
        return False


# Loop para interação com o usuario
while True:
    user_opt = input(
        'Choose an option between (1) chihuahua or (2) muffin. To exit the program (0): ')

    if not is_valid_int(user_opt):
        print('Please inform a valid option.')
        continue

    user_opt = int(user_opt)

    if user_opt == 1:
        print('You chose the option number 1. Chihuahua!!')

        img_opt_qty = input(
            f'There are  8 images available, pick one choosing a number between (1-8): ')

        if not is_valid_int(img_opt_qty):
            print('Please select a valid option.')
            continue

        img_opt_qty = int(img_opt_qty)

        if img_opt_qty == 0:
            break

        if img_opt_qty < 0 or img_opt_qty > 9:
            print('Please select a valid option.')
            continue

        img_path = os.path.join(
            img_base_folder, f"chihuahua-{img_opt_qty}.jpg")

    elif user_opt == 2:
        print('You chose the option number 2. Muffins!!')

        img_opt_qty = input(
            'There are 8 images available, pick one choosing a number between (1-8). To exit the program (0): ')

        if not is_valid_int(img_opt_qty):
            print('Please select a valid option.')
            continue

        img_opt_qty = int(img_opt_qty)

        if img_opt_qty == 0:
            break

        if img_opt_qty < 0 or img_opt_qty > 9:
            print('Please select a valid option.')
            continue

        img_choosed = load_img(os.path.join(
            img_base_folder, f"muffin-{img_opt_qty}.jpeg"), target_size=(128, 128))
        img_path = os.path.join(
            img_base_folder, f"muffin-{img_opt_qty}.jpeg")

    elif user_opt == 0:
        break

    else:
        print('Please inform a valid option.')
        continue

    # Mostrar imagem para o usuario
    plt.figure()
    img_to_process = plt.imread(img_path)
    plt.imshow(img_to_process)
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_to_pred = np.expand_dims(img_array, axis=0)
    preprocessed_image = np.vstack([img_to_pred])
    predictions = model.predict(preprocessed_image)
    pred = (model.predict(img_to_pred) > 0.5).astype('int32')[0][0]

    # RESULTADO DA PREDIÇÂO
    if pred == 1:
        title = 'Muffin'
    else:
        title = 'Chihuahua'

    print(f"Predicted:  {title}")

    print(f"Probabilities: {predictions}")
    plt.show(block=False)

    input("Press enter to continue...")
    plt.close()
