##encoding:utf-8

class TextConfig():

    embedding_size=100     #dimension of word embedding
    vocab_size=6000        #number of vocabulary=20000
    pre_trianing = None   #use vector_char trained by word2vec

    seq_length=600         #max length of sentence
    num_classes=4          #number of labels

    num_filters=256        #number of convolution kernel
    kernel_size=5          #size of convolution kernel
    hidden_dim=128         #number of fully_connected layer units

    keep_prob=0.5          #droppout每个元素被保留的概率，一般在大量数据训练时，为了防止过拟合，添加Dropout层，设置一个0~1之间的小数，
    lr= 1e-3               #learning rate
    lr_decay= 0.9          #learning rate decay
    clip= 5.0              #gradient clipping threshold

    num_epochs=10          #epochs
    batch_size= 64         #batch_size
    print_per_batch =100   #print result

    filename='../data/train.csv'  #train data
    test_filename='../data/val.csv'    #test data
    val_filename='../data/test.csv'      #validation data

    vocab_dir='../data/vocab_list.txt'        #vocabulary
    
    vector_word_filename='../data/vector_word.txt'  #训练好的模型vector_word trained by word2vec
    vector_word_npz='../data/vector_word.npz'   # save vector_word to numpy file
