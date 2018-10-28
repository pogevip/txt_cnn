#encoding:utf-8
import  tensorflow as tf
from config import TextConfig as config

class TextCNN(object):

    def __init__(self,config):
        self.config=config

        self.input_x=tf.placeholder(tf.int32,shape=[None,self.config.seq_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,shape=[None,self.config.num_classes],name='input_x')

        self.keep_prob=tf.placeholder(tf.float32,name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        
        self.cnn()

    def cnn(self):
        #前向传播
        with tf.device('/cpu:0'):
            #设置embedding
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_trianing))
            #根据索引转换成矩阵
            embedding_inputs=tf.nn.embedding_lookup(self.embedding,self.input_x)

        with tf.name_scope('cnn'):  #从输入到池化层
            #卷积神经网络，图层是1层，参数是输入，卷积核的数量256（产生256（对应数量）个神经元），卷积核的大小5
            conv= tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            #使用最大池化的方法
            outputs= tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            #创建一个卷积层，把输入放进去，给定卷积核的各种参数，使用最大池化的方法，输出

        with tf.name_scope('fc'):   #从池化层到第一个全连接层
            #全连接，输入，该层输出的大小
            fc=tf.layers.dense(outputs,self.config.hidden_dim,name='fc1')
            #使用dropout
            fc = tf.nn.dropout(fc, self.keep_prob)
            #指定激活函数
            fc=tf.nn.relu(fc)
            #创建一个全连接层，以上一层的输出作为输入，指定维度（神经元个数），使用dropout，激活函数是relu
        
        with tf.name_scope('logits'):
            #再接一个全连接（最后一个全连接没有激活函数）
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='logits')
            #用softmax做“归一化”，输出概率
            self.prob = tf.nn.softmax(self.logits)
            #选择概率最大作为类别的预测结果
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        #反向传播
        with tf.name_scope('loss'):#构建损失函数
            #计算交叉熵，最后一层全连接的输出与实际的y的交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            #计算loss，交叉熵取平均
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):#使用梯度下降拟合，降低loss
            #使用AdamOptimizer梯度下降方法，指定学习率
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            #没看懂，看意思是在“计算梯度”？
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            #好像是在做梯度修剪？clip是阈值
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        #统计准确率
        with tf.name_scope('accuracy'):
            #判断预测结果与实际值是否相等
            correct_pred=tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
            #tf.cast是将bool类型的转换成了float32类型，输出准确率
            self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
