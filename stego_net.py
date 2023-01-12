# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:33:35 2018

@author: Suraj
"""

import tensorflow as tf #tensorflow 1.10
import os
import hiding_net
import revealing_net
import discriminators
import preparation
import data
import cv2 #opencv 3.4
import numpy as np #numpy 1.14.5
import pickle
import data
from tensorflow.python.client import device_lib
from time import time
import matplotlib.pyplot as plt #matplotlib 2.1.0
import random


os.environ["CUDA_VISIBLE_DEVICES"] = '0' #本次运行文件使用的是第二块显卡
print(tf.test.is_gpu_available())
print(device_lib.list_local_devices())
test_container_loc = 'ucf/test_rel/'
test_loc = 'ucf/test/'#测试用的数据集目录
train_loc = 'ucf/train/'#训练用的数据集目录
#rel_test获取数据
def get_test_container_data():

    # dirs = os.listdir(test_loc)
    # dirs = sorted(dirs, key=lambda x: int(x))
    # cov = []
    # sec = []
    # for i in range(len(dirs)):
    #     for j in range(len(dirs)):
    #         if i != j:
    #             cov.append(test_loc + dirs[i])
    #             sec.append(test_loc + dirs[j])
    #构造数组dirs遍历保存test_loc下的第一层文件和文件夹名并加上路径前缀
    dirs = [test_container_loc + dir_name for dir_name in os.listdir(test_container_loc)]
    #从0-len(dirs)中随机选择x个数组成一维数组,相当于随机抽取测试或训练样本
    indices = np.random.choice(len(dirs), 1)
    #定义载体和秘密数组变量
    containers = []
    ori_secrets = []
    ori_covers = []
    #遍历indices数组， 遍历抽取的样本初始化covers和secrets
    for i in range(len(indices)):
        indirname = dirs[indices[i]]
        indirs = [indirname + '/' + file_name for file_name in os.listdir(indirname+'/')]
        #将样本加入containers数组中
        for file in indirs:
            if file == indirname+'/'+"container":
                containers.append(file)
            if file == indirname+'/'+"secret":
                ori_secrets.append(file)
            if file == indirname+'/'+"cover":
                ori_covers.append(file)

    # frames = []
    # for i in range(len(cov)):
    #     n1 = len(os.listdir(cov[i]))
    #     n2 = len(os.listdir(sec[i]))
    #     frames.append(np.min(n1, n2))

    # cover_tensor_data = []
    # secret_tensor_data = []
    # for i in range(len(cov)):
    #     frs = sorted(os.listdir(cov[i]), key=lambda x: int(x.split('.')[0]))
    #     for j in range(frames[i]):
    #         cover_tensor_data.append(cov[i] + '/' + frs[j])
    #         secret_tensor_data.append(sec[i] + '/' + frs[j])

    # return cover_tensor_data, secret_tensor_data
    #返回covers和secrets
    return containers,ori_secrets,ori_covers
#获取测试数据
def get_test_data():

    # dirs = os.listdir(test_loc)
    # dirs = sorted(dirs, key=lambda x: int(x))
    # cov = []
    # sec = []
    # for i in range(len(dirs)):
    #     for j in range(len(dirs)):
    #         if i != j:
    #             cov.append(test_loc + dirs[i])
    #             sec.append(test_loc + dirs[j])
    #构造数组dirs遍历保存test_loc下的第一层文件和文件夹名并加上路径前缀
    dirs = [test_loc + dir_name for dir_name in os.listdir(test_loc)]
    #加上遍历train_loc下的第一层文件和文件夹名
    dirs.extend([train_loc + dir_name for dir_name in os.listdir(train_loc)])
    #从0-len(dirs)中随机选择75个数组成一维数组,相当于随机抽取测试或训练样本
    indices = np.random.choice(len(dirs), 1)
    #定义载体和秘密数组变量
    covers = []
    secrets = []
    #遍历indices数组， 遍历抽取的样本初始化covers和secrets
    for i in range(len(indices)):
        #将样本加入covers数组中
        covers.append(dirs[indices[i]])
        #再从dirs中随机抽一个样本加入secrets且该样本不等于本次加入covers的
        index = np.random.choice(len(dirs))
        while dirs[index] == covers[i]:
            index = np.random.choice(len(dirs))
        secrets.append(dirs[index])

    # frames = []
    # for i in range(len(cov)):
    #     n1 = len(os.listdir(cov[i]))
    #     n2 = len(os.listdir(sec[i]))
    #     frames.append(np.min(n1, n2))

    # cover_tensor_data = []
    # secret_tensor_data = []
    # for i in range(len(cov)):
    #     frs = sorted(os.listdir(cov[i]), key=lambda x: int(x.split('.')[0]))
    #     for j in range(frames[i]):
    #         cover_tensor_data.append(cov[i] + '/' + frs[j])
    #         secret_tensor_data.append(sec[i] + '/' + frs[j])

    # return cover_tensor_data, secret_tensor_data
    #返回covers和secrets
    return covers, secrets
#获取训练数据
def get_train_data():
    #获取train_loc下的文件名
    dirs = os.listdir(train_loc)
    #文件名按数字排序
    dirs = sorted(dirs, key=lambda x: int(x))
    #构造dat数组将dirs中的文件名加上路径后保存
    dat = []
    for i in range(len(dirs)):
        dat.append(train_loc + dirs[i])
    #返回dat
    return dat
#SingleSizeModel类
class SingleSizeModel():
    # def get_noise_layer_op(self,tensor,std=.1):
    #     with tf.variable_scope("noise_layer"):
    #         return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32) 

    def __init__(self, beta, log_path, input_shape=(None,4,240,320,3), input_shape1=(None,4,240,320,3),input_shape2=(None,4,240,320,3),input_shape3=(None,4,240,320,3),input_shape4=(None,4,240,320,3)):
        #模型的初始化
        #shape1为secret,shape为cover
        #输入8个秘密帧和载体帧以及beta和log
        #检查点目录名
        self.checkpoint_dir = 'checkpoints_new'
        # self.model_name = 'stegnet_with_disc'
        #模型名
        self.model_name = 'stegnet'
        #数据集名
        self.dataset_name = 'ucf'
        #测试集的目录名
        self.test_dir_all = 'test_for_video'
        #log的目录名
        self.log_dir = 'logs'
        #图像高、宽
        self.img_height = 240
        self.img_width = 320
        #channel = 3
        self.channels = 3
        #每一批帧的个数 = 8
        self.frames_per_batch = 4

        #判断并创建目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.test_dir_all):
            os.makedirs(self.test_dir_all)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.cover_tensor_data_test, self.secret_tensor_data_test = get_test_data()
        self.container_tensor_data_test,self.ori_secret_tensor_data_test,self.ori_cover_tensor_data_test = get_test_container_data()
        pickle.dump(self.container_tensor_data_test, open('con_data_vid.pkl', 'wb'))
        pickle.dump(self.cover_tensor_data_test, open('cov_data_vid.pkl', 'wb'))
        pickle.dump(self.secret_tensor_data_test, open('sec_data_vid.pkl', 'wb'))
        pickle.dump(self.ori_secret_tensor_data_test, open('ori_sec_data_vid.pkl', 'wb'))
        pickle.dump(self.ori_cover_tensor_data_test, open('ori_cov_data_vid.pkl', 'wb'))
        #加载pkl数据集为测试数据
        self.container_tensor_data_test = pickle.load(open('con_data_vid.pkl', 'rb'))
        self.cover_tensor_data_test = pickle.load(open('cov_data_vid.pkl', 'rb'))
        self.secret_tensor_data_test = pickle.load(open('sec_data_vid.pkl', 'rb'))
        self.ori_secret_tensor_data_test = pickle.load(open('ori_sec_data_vid.pkl', 'rb'))
        self.ori_cover_tensor_data_test = pickle.load(open('ori_cov_data_vid.pkl', 'rb'))
        #调用get_train_data()获取train_loc下的训练数据
        self.tensor_data_train = get_train_data()

        #设置beta,学习率，
        self.beta = beta
        self.learning_rate = 0.0001
        #InteractiveSession()直接用eval()就可以直接获得结果，相较Session无需运行sess.run()
        self.sess = tf.InteractiveSession()
        #设置和初始化占位变量（待输入量）secret_tensor，cover_tensor，初值为输入的input_shape1和input_shape，作为input_secret和input_cover
        self.secret_tensor = tf.placeholder(shape=input_shape1,dtype=tf.float32,name="input_secret")
        self.cover_tensor = tf.placeholder(shape=input_shape,dtype=tf.float32,name="input_cover")
        self.container_tensor = tf.placeholder(shape=input_shape2, dtype=tf.float32, name="input_container")
        self.ori_secret_tensor = tf.placeholder(shape=input_shape3, dtype=tf.float32, name="input_ori_secret")
        self.ori_cover_tensor = tf.placeholder(shape=input_shape4, dtype=tf.float32, name="input_ori_cover")
        #设置全局深度变量记录迭代次数，用于保存模型，
        #在tensorflow中，变量是存在于 Session 环境中，也就是说，只有在 Session 环境下才会存有变量值
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        #调用self的prepare_training_graph方法来初始化构造训练模型，返回得到优化后的结果
        self.train_op , self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op = self.prepare_training_graph(self.secret_tensor,self.cover_tensor,self.global_step_tensor)
        # self.train_op , self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op, self.disc_loss_cc, self.disc_loss_sr = self.prepare_training_graph(self.secret_tensor,self.cover_tensor,self.global_step_tensor)
        #定义一个写入summary的目标文件，log_dir为写入文件地址,sess.graph为要保存的图
        self.writer = tf.summary.FileWriter(self.log_dir,self.sess.graph)

        # self.hiding_output, self.reveal_output, self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op, self.cover_acc, self.secret_acc = self.prepare_test_graph(self.cover_tensor, self.secret_tensor)
        #构造test的计算图
        self.hiding_output, self.reveal_output, self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op = self.prepare_test_graph(self.cover_tensor, self.secret_tensor)
        self.reveal_container_output,self.ori_secret_output,self.ori_cover_output,self.loss_a,self.loss_s,self.loss_c = self.prepare_test_reveal_graph(self.container_tensor,self.ori_secret_tensor,self.ori_cover_tensor)
        #构造test_rel的计算图

        #初始化所有变量并打印值
        self.sess.run(tf.global_variables_initializer())
        
        print("OK")

    #准备训练计算图
    def prepare_training_graph(self,secret_tensor,cover_tensor,global_step_tensor):
    
        # prep_secret = preparation.prep_network(secret_tensor)
        #构造变量保存输入数据的占位变量
        prep_cover = cover_tensor  #network.cover_prep_network(cover_tensor)
        prep_secret = secret_tensor
        #调用hiding_net中创建的hiding_net训练数据
        hiding_output = hiding_net.hiding_network(prep_cover, prep_secret)
        #调用revealing_net中创建的revealing_net训练数据
        reveal_output = revealing_net.revealing_network(hiding_output)
        #noise_add_op = self.get_noise_layer_op(hiding_output_op)

        # loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output,beta=self.beta)
        #调用self的get_loss_op方法得到损失函数
        loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output,beta=self.beta)
        #按设置好的深度不断训练优化
        minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op, global_step=global_step_tensor)
        #输出结果
        print(hiding_output.shape)
        print(reveal_output.shape)
        print(secret_tensor.shape)
        print(cover_tensor.shape)
        #添加变量到直方图中
        tf.summary.scalar('loss', loss_op,family='train')
        tf.summary.scalar('reveal_net_loss', secret_loss_op,family='train')
        tf.summary.scalar('cover_net_loss', cover_loss_op,family='train')
        # tf.summary.scalar('secret_acc', secret_acc,family='train')
        # tf.summary.scalar('cover_acc', cover_acc,family='train')



        # tf.summary.image('secret',secret_tensor[:,:,3]*255,max_outputs=1,family='train')
        # tf.summary.image('cover',cover_tensor[:,:,3]*255,max_outputs=1,family='train')
        # tf.summary.image('hidden',hiding_output[:,:,3]*255,max_outputs=1,family='train')
        # #tf.summary.image('hidden_noisy',self.get_tensor_to_img_op(noise_add_op),max_outputs=1,family='train')
        # tf.summary.image('revealed',reveal_output[:,:,3]*255,max_outputs=1,family='train')
        #将所有summary全部保存到磁盘，以便tensorboard显示
        merged_summary_op = tf.summary.merge_all()

        # return minimize_op,merged_summary_op, loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc
        #返回优化后的结果
        return minimize_op,merged_summary_op, loss_op,secret_loss_op,cover_loss_op



    # def prepare_training_graph(self,secret_tensor,cover_tensor,global_step_tensor):
    
    #     # prep_secret = preparation.prep_network(secret_tensor)
    #     prep_cover = cover_tensor  #network.cover_prep_network(cover_tensor)
    #     prep_secret = secret_tensor
    #     hiding_output = hiding_net.hiding_network(prep_cover, prep_secret)
    #     reveal_output = revealing_net.revealing_network(hiding_output)
    #     discriminator_cover_logits = discriminators.disc_container_cover_network(prep_cover)
    #     discriminator_container_logits = discriminators.disc_container_cover_network(hiding_output, reuse=True)
    #     discriminator_secret_logits = discriminators.disc_secret_revealed_network(prep_secret)
    #     discriminator_revealed_logits = discriminators.disc_secret_revealed_network(reveal_output, reuse=True)
    #     #noise_add_op = self.get_noise_layer_op(hiding_output_op)

    #     # loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output,beta=self.beta)
    #     loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output,beta=self.beta)
    #     disc_loss_cc = discriminators.discriminator_loss(discriminator_cover_logits, discriminator_container_logits)
    #     disc_loss_sr = discriminators.discriminator_loss(discriminator_secret_logits, discriminator_revealed_logits)

    #     # t_vars = tf.trainable_variables()
    #     # d_vars = [var for var in t_vars if 'disc' in var.name]

    #     # minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op + disc_loss_cc + disc_loss_sr, var_list=d_vars, global_step=global_step_tensor)
    #     minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op + disc_loss_cc + disc_loss_sr, global_step=global_step_tensor)

    #     tf.summary.scalar('loss', loss_op,family='train')
    #     tf.summary.scalar('reveal_net_loss', secret_loss_op,family='train')
    #     tf.summary.scalar('cover_net_loss', cover_loss_op,family='train')
    #     tf.summary.scalar('disc_con_cov_loss', disc_loss_cc,family='train')
    #     tf.summary.scalar('disc_sec_rev_loss', disc_loss_sr,family='train')
    #     # tf.summary.scalar('secret_acc', secret_acc,family='train')
    #     # tf.summary.scalar('cover_acc', cover_acc,family='train')



    #     # tf.summary.image('secret',secret_tensor[:,:,3]*255,max_outputs=1,family='train')
    #     # tf.summary.image('cover',cover_tensor[:,:,3]*255,max_outputs=1,family='train')
    #     # tf.summary.image('hidden',hiding_output[:,:,3]*255,max_outputs=1,family='train')
    #     # #tf.summary.image('hidden_noisy',self.get_tensor_to_img_op(noise_add_op),max_outputs=1,family='train')
    #     # tf.summary.image('revealed',reveal_output[:,:,3]*255,max_outputs=1,family='train')

    #     merged_summary_op = tf.summary.merge_all()

    #     # return minimize_op,merged_summary_op, loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc
    #     return minimize_op,merged_summary_op, loss_op,secret_loss_op,cover_loss_op, disc_loss_cc, disc_loss_sr

    #与train类似
    #准备测试计算图
    def prepare_test_graph(self, cover_tensor, secret_tensor):
        with tf.variable_scope("",reuse=True):

            # Image_Data_Class = ImageData(self.img_width, self.img_height, self.channels)
            # cover_tensor = tf.data.Dataset.from_tensor_slices(self.cover_tensor_data_test)
            # secret_tensor = tf.data.Dataset.from_tensor_slices(self.secret_tensor_data_test)

            # gpu_device = '/gpu:0'
            # cover_tensor = cover_tensor.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
            # secret_tensor = secret_tensor.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))

            # inputs_iterator = inputs.make_one_shot_iterator()

            # self.inputs = inputs_iterator.get_next()

            # prep_secret = preparation.prep_network(secret_tensor)
            prep_secret = secret_tensor
            prep_cover = cover_tensor  #network.cover_prep_network(cover_tensor)
            hiding_output = hiding_net.hiding_network(prep_cover, prep_secret)
            print("\n\nhiding_output=\n",hiding_output)
            reveal_output = revealing_net.revealing_network(hiding_output)


            # loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output)
            loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output)
            
        
            tf.summary.scalar('loss', loss_op,family='test')
            tf.summary.scalar('reveal_net_loss', secret_loss_op,family='test')
            tf.summary.scalar('cover_net_loss', cover_loss_op,family='test')
            # tf.summary.scalar('secret_acc', secret_acc,family='test')
            # tf.summary.scalar('cover_acc', cover_acc,family='test')
            # tf.summary.image('secret',secret_tensor[:,:,3]*255,max_outputs=1,family='test')
            # tf.summary.image('cover',cover_tensor[:,:,3]*255,max_outputs=1,family='test')
            # tf.summary.image('hidden',hiding_output[:,:,3]*255,max_outputs=1,family='test')
            # tf.summary.image('revealed',reveal_output[:,:,3]*255,max_outputs=1,family='test')

            merged_summary_op = tf.summary.merge_all()


            # return hiding_output, reveal_output, merged_summary_op, loss_op, secret_loss_op, cover_loss_op, cover_acc, secret_acc
            return hiding_output, reveal_output, merged_summary_op, loss_op, secret_loss_op, cover_loss_op
    def prepare_test_reveal_graph(self, container_tensor,ori_secret_tensor,ori_cover_tensor):
        with tf.variable_scope("", reuse=True):
            prep_container = container_tensor
            prep_secret = ori_secret_tensor
            prep_cover = ori_cover_tensor
            print("\n\ncontainer_input=\n", prep_container)
            # prep_container shape = (?, 8, 240, 320, 3), dtype=float32
            reveal_output = revealing_net.revealing_network(prep_container)
            loss_op, secret_loss_op, cover_loss_op = self.get_loss_op(prep_secret, reveal_output, prep_cover,
                                                                      prep_container, beta=self.beta)
            #print("loss:",loss_op," secret_loss:",secret_loss_op,"cover_loss:",cover_loss_op,"\n")
            return reveal_output,prep_secret,prep_cover,loss_op,secret_loss_op,cover_loss_op
    #得到损失函数
    def get_loss_op(self,secret_true,secret_pred,cover_true,cover_pred,beta=1.0):
        #创建全局共享变量losses
        with tf.variable_scope("losses"):

            beta = tf.constant(beta, name="beta")
            secret_mse = tf.losses.mean_squared_error(secret_true,secret_pred)
            cover_mse = tf.losses.mean_squared_error(cover_true,cover_pred)

            final_loss = cover_mse + beta * secret_mse

            # secret_acc= tf.equal(tf.argmax(secret_true,1),tf.argmax(secret_pred,1))
            # secret_acc = tf.reduce_mean(tf.cast(secret_acc, tf.float32))

            # cover_acc= tf.equal(tf.argmax(cover_true,1),tf.argmax(cover_pred,1))
            # cover_acc = tf.reduce_mean(tf.cast(cover_acc, tf.float32))

            # return final_loss , secret_mse , cover_mse, cover_acc, secret_acc
            return final_loss , secret_mse , cover_mse


    # def make_chkp(self,saver, path):
    #     global_step = self.sess.run(self.global_step_tensor)
    #     saver.save(self.sess,path,global_step)

    # def load_chkp(self,saver, path):
    #     print("LOADING")
    #     global_step = self.sess.run(self.global_step_tensor)
    #     tf.reset_default_graph()
    #     imported_meta = tf.train.import_meta_graph("./model8_beta0.75/my-model.ckpt-45174.meta")
    #     imported_meta.restore(self.sess, tf.train.latest_checkpoint('./model8_beta0.75/'))
    #     #saver.restore(self.sess, path)
    #     print("LOADED")
        
       
    def train(self):
        #训练集的个数/训练的总次数
        epochs = 5500
        #设置保存类，保存最近2个检查点
        self.saver = tf.train.Saver(max_to_keep=2)
        #加载检查点目录
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            count = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            count = 0
            print(" [!] Load failed...")


        # # For saving generator weights.
        # t_vars = tf.trainable_variables()
        # weights = self.sess.run(t_vars)
        # t_weights_file = open('initial_weights_4100.npy','wb')
        # np.save(t_weights_file, weights)
        # print('saved')
        # input()

        # initial_weights_4100 = np.load('initial_weights_4100.npy')
        # t_vars = tf.trainable_variables()
        # prev_vars = [var for var in t_vars if 'disc' not in var.name]
        # for i in range(len(prev_vars)):
        #     t_vars[i].assign(initial_weights_4100[i])
        # print ('assigned weights successfully.')

        def load_t(base_dir, frame_names, ind):
            #c_path/s_patn,frs,j
            fpb = []
            #取8帧
            for i in range(self.frames_per_batch):
                #用cv2读取帧，大小为240*320*3，放到fpb中
                frame = base_dir + '/' + frame_names[ind * self.frames_per_batch + i]
                frame = cv2.imread(frame)
                frame = frame / 255.0
                frame = frame.reshape((240,320,3))
                fpb.append(frame)
            #返回fpb,8*240*320*3
            return np.reshape(np.array(fpb), (1, self.frames_per_batch, 240, 320, 3))

        def generator():
            #从记录的检查点开始，到设置训练集的个数为止
            for i in range(count, epochs):
                #随机选取载体数据和秘密数据（怀疑这里对应的是文件夹名不是文件名）
                c_path = random.choice(self.tensor_data_train)
                s_path = random.choice(self.tensor_data_train)
                while s_path == c_path:
                    s_path = random.choice(self.tensor_data_train)
                #记录文件夹内文件个数
                n1 = len(os.listdir(c_path))
                n2 = len(os.listdir(s_path))
                #取t,t*self.frames_per_batch刚好接近并<=n1,<=n2，由此判断文件夹里面是帧
                t = int(min(n1, n2) / self.frames_per_batch)
                #这里是关键，如何排序帧（推测是文件名是x.jpg这样，按照x数字从小到大排序，并且frs只取前t * self.frames_per_batch个）
                frs = sorted(os.listdir(c_path), key=lambda x: int(x.split('.')[0]))[:t * self.frames_per_batch]
                #取t批，每批8帧
                for j in range(t):
                    #load_t是加载帧的关键
                    cov_tens = load_t(c_path, frs, j)
                    sec_tens = load_t(s_path, frs, j)
                    #使用yield保存并返回，待下次调用generator()时才会再进入下一轮循环
                    yield cov_tens, sec_tens, c_path.split('/')[-1], s_path.split('/')[-1], i, j
                    #以‘/ ’为分割符，保留最后一段

        #无论加载检查点有无成功都会来到这
        for covers, secrets, c_name, s_name, epoch, batch in generator():
            #每次从generator中拿到8*240*320*3的载体和秘密帧数组，以及他们对应的文件夹名和对应的第epoch次训练的第batch批数据
            #输入训练模型
            self.sess.run([self.train_op],feed_dict={"input_secret:0":secrets,"input_cover:0":covers})
            summaree, gs, tl, sl, cl = self.sess.run([self.summary_op, self.global_step_tensor, self.loss_op,
                                                    self.secret_loss_op,self.cover_loss_op],
                                                    feed_dict={"input_secret:0":secrets,"input_cover:0":covers})
            self.writer.add_summary(summaree,gs)
            #训练完成，打印相关信息
            print('\nEpoch: '+str(epoch)+' Batch: '+str(batch)+' Loss: '+str(tl)+' Cover_Loss: '+str(cl)+' Secret_Loss: '+str(sl))
            #每100次保存检查点
            if np.mod(epoch + 1, 2) == 0:
                #这里会保存检查点
                self.save(self.checkpoint_dir, epoch)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.model_name, self.dataset_name, self.img_height, self.img_width, self.beta)

    @property
    def test_dir(self):
        return "{}_{}_{}_{}_{}_test".format(
            self.model_name, self.dataset_name, self.img_height, self.img_width, self.beta)    


    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        #model_dir = <model_name>_<dataset_name>_<img_height>_<img_width_beta>
        #不存在则创建目录
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #调用saver类保存检查点
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        #连接路径<checkpoint_dir>\<self.model_dir>
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        #从checkpoint_dir路径下查找“checkpoint”的文件（检查点）
        #如果路径有效则返回一个CheckpointState proto对象（该对象有两个可用的属性值），否则返回None
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            #根据保存点恢复状态
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            #失败
            print(" [*] Failed to find a checkpoint")
            return False, 0

    # def test(self, test_cover, test_secret, c_name, s_name):

    #     # self.load_chkp(saver, path)

    #     self.saver = tf.train.Saver(max_to_keep=2)

    #     could_load, checkpoint_counter = self.load(self.checkpoint_dir)

    #     if could_load:
    #         # start_epoch = (int)(checkpoint_counter / self.iteration)
    #         # start_batch_id = checkpoint_counter - start_epoch * self.iteration
    #         count = checkpoint_counter
    #         print(" [*] Load SUCCESS")
    #     else:
    #         # start_epoch = 0
    #         # start_batch_id = 0
    #         count = 1
    #         print(" [!] Load failed...")
    #         exit()

    #     print ('c_name: ' + c_name + ' s_name: ' + s_name)

    #     batch_size = 1
    #     Num = test_cover.shape[0]*test_cover.shape[1]
    #     num_of_batches = test_cover.shape[0] // batch_size
    #     cover_loss = 0
    #     secret_loss = 0
    #     cover_accuracy = 0 
    #     secret_accuracy = 0 

    #     test_dir = os.path.join(self.test_dir_all, self.test_dir)
    #     video_dir = os.path.join(test_dir, 'c_'+c_name+'_s_'+s_name)
    #     cover_dir = os.path.join(video_dir, 'cover')
    #     container_dir = os.path.join(video_dir, 'container')
    #     secret_dir = os.path.join(video_dir, 'secret')
    #     revealed_secret_dir = os.path.join(video_dir, 'revealed_secret')

    #     if not os.path.exists(test_dir):
    #         os.makedirs(test_dir)
    #     if not os.path.exists(video_dir):
    #         os.makedirs(video_dir)
    #         os.makedirs(cover_dir)
    #         os.makedirs(container_dir)
    #         os.makedirs(secret_dir)
    #         os.makedirs(revealed_secret_dir)
    
    #     for i in range(num_of_batches):

    #         print("Frame: "+str(i))

    #         test_cover_input = test_cover[i]
    #         test_secret_input = test_secret[i]

    #         test_cover_input = np.reshape(test_cover_input,(1,8,240,320,3))
    #         test_secret_input = np.reshape(test_secret_input,(1,8,240,320,3))

    #         #covers, secrets = covers, secrets
    #         # hiding_b, reveal_b, summaree, tl, cl, sl, ca, sa= self.sess.run([self.hiding_output, self.reveal_output,  self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op, self.cover_acc, self.secret_acc],feed_dict={"input_secret:0":test_secret_input,"input_cover:0":test_cover_input})
    #         hiding_b, reveal_b, summaree, tl, cl, sl= self.sess.run([self.hiding_output, self.reveal_output,  self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op],feed_dict={"input_secret:0":test_secret_input,"input_cover:0":test_cover_input})
    #         print(hiding_b.shape)
    #         self.writer.add_summary(summaree)

    #         for j in range(8):
    #             im = np.reshape(hiding_b[0][j]*255,(240,320,3))
    #             im1 = np.reshape(reveal_b[0][j]*255,(240,320,3))
    #             cv2.imwrite(container_dir+'/'+str(i*8 + j)+'.png', im)
    #             cv2.imwrite(revealed_secret_dir+'/'+str(i*8 + j)+'.png', im1)
    #             im = np.reshape(test_cover_input[0][j]*255,(240,320,3))
    #             im1 = np.reshape(test_secret_input[0][j]*255,(240,320,3))
    #             cv2.imwrite(cover_dir+'/'+str(i*8 + j)+'.png', im)
    #             cv2.imwrite(secret_dir+'/'+str(i*8 + j)+'.png', im1)
    #             # cover_accuracy = cover_accuracy + ca
    #             # secret_accuracy = secret_accuracy + sa
    #             cover_loss = cover_loss + cl
    #             secret_loss = secret_loss + sl
            
    #     # print('Cover_Loss: '+str(cover_loss/Num)+'\tSecret_Loss: '+str(secret_loss/Num)+'\tCover_Accuracy: '+str(cover_accuracy/Num)+'\tSecret_Accuracy: '+str(secret_accuracy/Num))
    #     print('Cover_Loss: '+str(cover_loss/Num)+'\tSecret_Loss: '+str(secret_loss/Num))

    #测试
    def test(self):

        # self.load_chkp(saver, path)
        #设置保存类
        self.saver = tf.train.Saver(max_to_keep=2)
        #加载
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        #判断，注意这里没有checkpoint是直接退出
        if could_load:
            # start_epoch = (int)(checkpoint_counter / self.iteration)
            # start_batch_id = checkpoint_counter - start_epoch * self.iteration
            count = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            # start_epoch = 0
            # start_batch_id = 0
            count = 1
            print(" [!] Load failed...")
            exit()
        #取8帧
        def load_t(base_dir, frame_names, ind):

            fpb = []
            for i in range(self.frames_per_batch):
                frame = base_dir + '/' + frame_names[ind * self.frames_per_batch + i]
                frame = cv2.imread(frame)
                frame = frame / 255.0
                frame = frame.reshape((240,320,3))
                fpb.append(frame)

            return np.reshape(np.array(fpb), (1, self.frames_per_batch, 240, 320, 3))
        #取帧
        def generator():
            for i in range(len(self.cover_tensor_data_test)):
                c_name = '_'.join(self.cover_tensor_data_test[i].split('/')[-2:])
                s_name = '_'.join(self.secret_tensor_data_test[i].split('/')[-2:])
                n1 = len(os.listdir(self.cover_tensor_data_test[i]))
                n2 = len(os.listdir(self.secret_tensor_data_test[i]))
                t = int(min(n1, n2) / self.frames_per_batch)
                frs = sorted(os.listdir(self.cover_tensor_data_test[i]), key=lambda x: int(x.split('.')[0]))[:t * self.frames_per_batch]
                for j in range(t):
                    cov_tens = load_t(self.cover_tensor_data_test[i], frs, j)
                    sec_tens = load_t(self.secret_tensor_data_test[i], frs, j)
                    yield cov_tens, sec_tens, c_name, s_name



        # test_cover, test_secret, c_name, s_name= data.test_data()

        prev_name = ''
        i = 0
        vid = 0

        start_time = time()
        total_frames = 0
        for test_cover, test_secret, c_name, s_name in generator():

            print ('c_name: ' + c_name + ' s_name: ' + s_name + ' vid: '+ str(vid) + ' i: ' + str(i))
            
            if c_name + s_name != prev_name:
                i = 0
                vid += 1
            # batch_size = 1
            # Num = test_cover.shape[0]*test_cover.shape[1]
            # num_of_batches = test_cover.shape[0] // batch_size
            cover_loss = 0
            secret_loss = 0
            cover_accuracy = 0 
            secret_accuracy = 0 
            #创建输出文件夹
            test_dir = os.path.join(self.test_dir_all, self.test_dir)
            video_dir = os.path.join(test_dir, 'c_'+c_name+'_s_'+s_name)
            cover_dir = os.path.join(video_dir, 'cover')
            container_dir = os.path.join(video_dir, 'container')
            secret_dir = os.path.join(video_dir, 'secret')
            revealed_secret_dir = os.path.join(video_dir, 'revealed_secret')
            diff_cover_container_dir = os.path.join(video_dir, 'diff_cc')
            diff_secret_revealed_dir = os.path.join(video_dir, 'diff_sr')

            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
                os.makedirs(cover_dir)
                os.makedirs(container_dir)
                os.makedirs(secret_dir)
                os.makedirs(revealed_secret_dir)
                os.makedirs(diff_cover_container_dir)
                os.makedirs(diff_secret_revealed_dir)
        
            # for i in range(num_of_batches):

            # print("Frame: "+str(i))

            # test_cover_input = test_cover[i]
            # test_secret_input = test_secret[i]

            # test_cover_input = np.reshape(test_cover_input,(1,8,240,320,3))
            # test_secret_input = np.reshape(test_secret_input,(1,8,240,320,3))

            #covers, secrets = covers, secrets
            # hiding_b, reveal_b, summaree, tl, cl, sl, ca, sa= self.sess.run([self.hiding_output, self.reveal_output,  self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op, self.cover_acc, self.secret_acc],feed_dict={"input_secret:0":test_secret_input,"input_cover:0":test_cover_input})
            #输入模型
            hiding_b, reveal_b = self.sess.run([self.hiding_output, self.reveal_output],feed_dict={"input_secret:0":test_secret, "input_cover:0":test_cover})
            # hiding_b = self.sess.run(self.hiding_output,feed_dict={"input_secret:0":test_secret, "input_cover:0":test_cover})
            # self.writer.add_summary(summaree)

            #print(hiding_b)
            #print(reveal_b)
            #每8帧输出
            for j in range(4):
                im = np.reshape(hiding_b[0][j] * 255, (240,320,3))
                im1 = np.reshape(reveal_b[0][j] * 255,(240,320,3))
                cv2.imwrite(container_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)
                cv2.imwrite(revealed_secret_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im1)
                im = np.reshape(test_cover[0][j] * 255,(240,320,3))
                im1 = np.reshape(test_secret[0][j] * 255,(240,320,3))
                cv2.imwrite(cover_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)
                cv2.imwrite(secret_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im1)
                im = np.reshape(np.absolute(hiding_b[0][j] - test_cover[0][j]) * 255, (240, 320, 3))
                im1 = np.reshape(np.absolute(reveal_b[0][j] - test_secret[0][j]) * 255, (240, 320, 3))
                cv2.imwrite(diff_cover_container_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)
                cv2.imwrite(diff_secret_revealed_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im1)
                # cover_accuracy = cover_accuracy + ca
                # secret_accuracy = secret_accuracy + sa
                # cover_loss = cover_loss + cl
                # secret_loss = secret_loss + sl
            
            i += 1    
            total_frames += self.frames_per_batch
            prev_name = c_name + s_name

            # print('Cover_Loss: '+str(cover_loss/Num)+'\tSecret_Loss: '+str(secret_loss/Num)+'\tCover_Accuracy: '+str(cover_accuracy/Num)+'\tSecret_Accuracy: '+str(secret_accuracy/Num))
            # print('Cover_Loss: '+str(cover_loss)+'\tSecret_Loss: '+str(secret_loss/Num))

        total_time = time() - start_time
        pickle.dump(total_time, open('total_time.pkl', 'wb'))
        time_per_frame = float(total_time) / total_frames 
        pickle.dump(time_per_frame, open('time_per_frame.pkl', 'wb'))
        print ('Total time: '+str(total_time)+' Time Per Frame: '+str(time_per_frame))



    # 测试
    def test_reveal(self):
        # 设置保存类
        self.saver = tf.train.Saver(max_to_keep=2)
        # 加载
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        # 判断，注意这里没有checkpoint是直接退出
        if could_load:
            count = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            count = 1
            print(" [!] Load failed...")
            exit()

        # 取8帧
        def load_t(base_dir, frame_names, ind):

            fpb = []
            for i in range(self.frames_per_batch):
                frame = base_dir + '/' + frame_names[ind * self.frames_per_batch + i]
                frame = cv2.imread(frame)
                frame = frame / 255.0
                frame = frame.reshape((240, 320, 3))
                fpb.append(frame)

            return np.reshape(np.array(fpb), (1, self.frames_per_batch, 240, 320, 3))

        # 取帧
        def generator():
            for i in range(len(self.container_tensor_data_test)):
                c_name = '_'.join(self.container_tensor_data_test[i].split('/')[-2:])
                o_s_name = '_'.join(self.ori_secret_tensor_data_test[i].split('/')[-2:])
                o_c_name = '_'.join(self.ori_cover_tensor_data_test[i].split('/')[-2:])
                n1 = len(os.listdir(self.container_tensor_data_test[i]))
                t = int(n1 / self.frames_per_batch)
                frs = sorted(os.listdir(self.container_tensor_data_test[i]), key=lambda x: int(x.split('.')[0]))[
                      :t * self.frames_per_batch]
                for j in range(t):
                    con_tens = load_t(self.container_tensor_data_test[i], frs, j)
                    ose_tens = load_t(self.ori_secret_tensor_data_test[i], frs, j)
                    oco_tens = load_t(self.ori_cover_tensor_data_test[i], frs, j)
                    yield con_tens, c_name,ose_tens,o_s_name,oco_tens,o_c_name

        prev_name = ''
        i = 0
        vid = 0

        start_time = time()
        total_frames = 0
        for test_container, c_name,test_ori_secret,o_s_name,test_ori_cover,o_c_name in generator():

            print('c_name: ' + c_name + ' vid: ' + str(vid) + ' i: ' + str(i))

            if c_name != prev_name:
                i = 0
                vid += 1

            cover_loss = 0
            secret_loss = 0
            cover_accuracy = 0
            secret_accuracy = 0
            # 创建输出文件夹
            test_dir = os.path.join(self.test_dir_all, self.test_dir)
            video_dir = os.path.join(test_dir, 'c_' + c_name)
            container_dir = os.path.join(video_dir, 'container')
            revealed_secret_dir = os.path.join(video_dir, 'revealed_secret')
            ori_secret_dir = os.path.join(video_dir, 'ori_secret')
            diff_secret_revealed_dir = os.path.join(video_dir, 'diff_sr')
            ori_cover_dir = os.path.join(video_dir, 'ori_cover')
            diff_cover_container_dir = os.path.join(video_dir, 'diff_cc')
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
                os.makedirs(container_dir)
                os.makedirs(revealed_secret_dir)
                os.makedirs(ori_secret_dir)
                os.makedirs(diff_secret_revealed_dir)
                os.makedirs(ori_cover_dir)
                os.makedirs(diff_cover_container_dir)

            # 输入模型
            print("begin run")
            reveal_b,os_b,oc_b,loss_a,loss_s,loss_c = self.sess.run([self.reveal_container_output,self.ori_secret_output,self.ori_cover_output,self.loss_a,self.loss_s,self.loss_c],
                                     feed_dict={"input_container:0": test_container,"input_ori_secret:0": test_ori_secret,"input_ori_cover:0": test_ori_cover})
            print("succes！")
            print("\nloss:",str(loss_a),"secret_loss:",str(loss_s),"cover_loss:",loss_c,"\n")
            # print(hiding_b)
            #print(reveal_b)
            # 每8帧输出
            for j in range(4):
                im1 = np.reshape(reveal_b[0][j] * 255, (240, 320, 3))
                cv2.imwrite(revealed_secret_dir + '/' + str(i * self.frames_per_batch + j) + '.jpg', im1)
                im1 = np.reshape(test_container[0][j] * 255, (240, 320, 3))
                cv2.imwrite(container_dir + '/' + str(i * self.frames_per_batch + j) + '.jpg', im1)
                im1 = np.reshape(test_ori_secret[0][j] * 255, (240, 320, 3))
                cv2.imwrite(ori_secret_dir + '/' + str(i * self.frames_per_batch + j) + '.jpg', im1)
                im1 = np.reshape(np.absolute(reveal_b[0][j] - test_ori_secret[0][j]) * 255, (240, 320, 3))
                cv2.imwrite(diff_secret_revealed_dir + '/' + str(i * self.frames_per_batch + j) + '.jpg', im1)
                im1 = np.reshape(test_ori_cover[0][j] * 255, (240, 320, 3))
                cv2.imwrite(ori_cover_dir + '/' + str(i * self.frames_per_batch + j) + '.jpg', im1)
                im1 = np.reshape(np.absolute(test_container[0][j] - test_ori_cover[0][j]) * 255, (240, 320, 3))
                cv2.imwrite(diff_cover_container_dir + '/' + str(i * self.frames_per_batch + j) + '.jpg', im1)

            i += 1
            total_frames += self.frames_per_batch
            prev_name = c_name

        total_time = time() - start_time
        pickle.dump(total_time, open('total_time.pkl', 'wb'))
        time_per_frame = float(total_time) / total_frames
        pickle.dump(time_per_frame, open('time_per_frame.pkl', 'wb'))
        print('Total time: ' + str(total_time) + ' Time Per Frame: ' + str(time_per_frame))

#显示所有变量
def show_all_variables():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)
  
#初始化模型
m = SingleSizeModel(beta=0.75, log_path='log/')
show_all_variables()
#开始训练
#m.train()
#m.train()
# test_cover, test_secret, c_name, s_name= data.test_data()
# m.test(test_cover, test_secret, c_name, s_name)
#m.test()
m.test_reveal()