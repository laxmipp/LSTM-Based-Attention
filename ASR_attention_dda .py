'''
Author: Laxmi Pandey
Date: 27/Sep/2017

Brief: Script to training and validating the model for automatic speech recognition based on attention netowrk for weighing the spectral and prosodic features and finally implementing the DDA based error modelling and thus finally predicting the improved phoneme sequence. 

'''




import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cs
from os import listdir
import csv
#import pandas as pd
datafolder = '/workspace/features/'
mfccfolder = datafolder + 'mfcc_pool/feats/'
prosfolder = datafolder + 'prosody/feats/'
labelfolder = datafolder + 'lab_idx/'
batch_size = 16 
mfcc = 13
pros = 6
rep_dim = 32
att_dim = 128
state_dim =  153
output_dim = 64
num_hidden = 50

def pad(features,labels):
'''
	params: 
		features-> the input features training data 
		labels-> training labels
	
	brief: 
		to make the inputs in an batch of same time length by appending zeros!
	
	return: 
		features: padded training data
		labels-> training labels
		mask-> original length of the inputs in the batch 
	
'''
	steps = []
	seq_len = []
	for i in range(len(features)):
		steps.append(features[i].shape[0])
	maxlength = max(steps)
	for i in range(len(features)):
		steps = features[i].shape[0]
		seq_len.extend([steps])
		toadd = maxlength - steps
		toadd = np.zeros((toadd,19))
		features[i] =  np.concatenate((features[i],toadd),axis= 0)
		toadd = maxlength - steps
		toadd = np.zeros((toadd,1))
		labels[i] = np.concatenate((labels[i],toadd),axis= 0)
	features = np.reshape(features, (batch_size,-1,19))
	seq_len =  np.array(seq_len)
	labels = np.reshape(labels,(batch_size,-1,1))
	mask = np.zeros((batch_size, maxlength),dtype = np.int)
	for i in range(batch_size):
		mask[i,:seq_len[i]] = np.ones((1,seq_len[i]),dtype = np.int)
	mask = mask == 1
	return features, labels, mask

def get_data(batch_num,filelist):
'''
	param:
		batch_num-> 
		filelist-> 
	brief:
		function to return the formated data for a batch by reading from a filelist
	return: 
		features-> formatted features for the batch
		labels-> labels for the batch
		mask-> original sequence length of the padded features in the batch
'''

	start = batch_num*batch_size
	end = (batch_num + 1)*batch_size
	features = []
	labels = []
	for i in range(start,end):
		filename = filelist[i]
		# print filename
		mfccfile = mfccfolder + filename
		prosfile = prosfolder + filename
		labelfile = labelfolder + filename
		r2 = csv.reader(open(labelfile,'r'),delimiter=',')
		r2 =  list(r2)
		r2 = np.array(r2)
		lab =  r2.astype(np.int32)
		labels.append(lab)
		lablength = lab.shape[0]
		r = csv.reader(open(mfccfile,'r'),delimiter=',')
		r =  list(r)
		r = np.array(r)
		mfccfeat =  r.astype(np.float)
		mfcclen = mfccfeat.shape[0]
		r1 = csv.reader(open(prosfile,'r'),delimiter=',')
		r1 =  list(r1)
		r1 = np.array(r1)
		prosfeat =  r1.astype(np.float)
		proslen = prosfeat.shape[0]
		if(proslen>lablength):
			prosody = prosfeat[:lablength]
		else:
			add = lablength - proslen
			add  = np.zeros((add,6))
			prosfeat = np.concatenate((prosfeat, add),axis = 0)
		if(mfcclen > lablength):
			mfcclen = mfccfeat[:lablength]
		else:
			add = lablength - mfcclen
			add = np.zeros((add,13))
			mfccfeat = np.concatenate((mfccfeat,add),axis = 0)
		feat = np.concatenate((mfccfeat,prosfeat),axis= 1)
		features.append(feat)

	return pad(features, labels)

def last_relevant(outputs, length):
'''
	param: 
		outputs-> output from the LSTM for all the nodes
		length-> original length of the sequence

	brief: function to obtain the last node output of the LSTM
	
	return:
		relevant-> the last node output from LSTM for all the samples in a batch
'''
	batch_size = tf.shape(outputs)[0]
	print ("batch_size", batch_size)
	max_length = tf.shape(outputs)[1]
	print ("max_length", max_length)
	out_size = 1
	index = tf.range(0,batch_size)*max_length+ (length-1)
	flat = tf.reshape(outputs,[-1, out_size])
	relevant = tf.gather(flat,index)
	return relevant

def recurrent_step(p_states,input_vec):
'''
	parma:
		p_states-> the output for the previous step of the LSTM cell containing output as well as previous states
		
	brief: function to be called by tf.scan recurrsively for all the time steps. Fuction computes the attention for the spectral and prosodic information based on the current input and the previous state of the LSTM. The attended features are then passed to LSTM cell to compute the output and the SLTM states
	
	return: 
		vector containing the states of the LSTM cell and the corresponding output of the step.
'''
	
	
	print ("input",input_vec.get_shape())
	# num_batches = tf.shape(input_vec)[0]## why not fix batch_size
	num_batches = batch_size
	print ("p_states", p_states.get_shape())
	# print ("num_batches", num_batches)
	# prev_states, output = tf.split(0,2,prev_states)
	prev_states = p_states[:,:306]
	output = p_states[:,306:]
	# prev_states = tf.squeeze(prev_states)
	print ("prev_states",prev_states.get_shape())
	print ("output",output.get_shape())
	# cell = tf.nn.rnn_cell.BasicRNNCell(50)
	with tf.variable_scope("recurrence") as scope:
		W_mr = tf.get_variable("weight_mr",[mfcc,rep_dim],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_mr = tf.get_variable("bias_mr",[rep_dim], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		W_pr = tf.get_variable("weight_pr",[pros,rep_dim],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_pr = tf.get_variable("bias_pr",[rep_dim], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		W_mr1 = tf.get_variable("weight_mr1",[32,64],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_mr1 = tf.get_variable("bias_mr1",[64], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		W_pr1 = tf.get_variable("weight_pr1",[32,64],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_pr1 = tf.get_variable("bias_pr1",[64], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		W_att = tf.get_variable("weight_att",[64,att_dim],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_att = tf.get_variable("bias_att",[att_dim], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		W_sr = tf.get_variable("weight_sr",[2*state_dim,rep_dim*2],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_sr = tf.get_variable("bias_sr",[rep_dim*2], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		W_soft = tf.get_variable("weight_soft",[rep_dim*12,2],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_soft = tf.get_variable("bias_soft",[2], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		W_pro = tf.get_variable("weight_pro",[256,64],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_pro = tf.get_variable("bias_pro",[64], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		

		mfcc1 = input_vec[:,:13]
		mfcc1 = tf.cast(mfcc1,tf.float64)
		prosody = input_vec[:,13:]
		prosody = tf.cast(prosody, tf.float64)
		cell = tf.nn.rnn_cell.LSTMCell(50,state_is_tuple=False,num_proj = 256)
		# cell = tf.nn.rnn_cell.LSTMCell(50,state_is_tuple=False)
		scope.reuse_variables()

		rm = tf.tanh(tf.add(tf.matmul(mfcc1,W_mr),b_mr))
		rp = tf.tanh(tf.add(tf.matmul(prosody,W_pr),b_pr))
		rs = tf.tanh(tf.add(tf.matmul(prev_states,W_sr),b_sr))
		rm1 = tf.tanh(tf.add(tf.matmul(rm,W_mr1),b_mr1))
		rp1 = tf.tanh(tf.add(tf.matmul(rp,W_pr1),b_pr1))
		#rs1 = tf.tanh(tf.add(tf.matmul(rs,W_sr1),b_sr1))
		matt = tf.tanh(tf.add(tf.matmul(rm1,W_att),b_att))
		patt = tf.tanh(tf.add(tf.matmul(rp1,W_att),b_att))
		satt = tf.tanh(tf.add(tf.matmul(rs,W_att),b_att))
		print ("rm",rm.get_shape())
		print ("matt",matt.get_shape())
		print ("patt",patt.get_shape())
		print ("satt",satt.get_shape())

		ccat = tf.concat(1,[matt,patt,satt])
		print("ccat",ccat.get_shape())
		attention =  tf.add(tf.matmul(ccat,W_soft),b_soft)
		attention =  tf.transpose(attention, perm= [1,0])
		attention =  tf.nn.softmax(attention, dim=0)
				

		diag = tf.diag(tf.reshape(attention, [-1]))
		f_cat = tf.concat(0,[rm,rp])
		feat = tf.matmul(diag,f_cat)
		lstmipt0 = feat[:batch_size,:]
		lstmipt1 = feat[batch_size:,:]
		lstmipt =  tf.concat(1,[lstmipt0,lstmipt1])
		print ("lstmipt",lstmipt.get_shape())
	update_o, update_s = cell(lstmipt,prev_states,scope="recurrence")
	print ("output", update_o.get_shape())
	print ("state", update_s.get_shape())
	pro =  tf.add(tf.matmul(update_o,W_pro),b_pro)
	print ("prowa", pro.get_shape())
	return tf.concat(1, [update_s, pro])

def dda(out_labs):
'''
	parma: 
		out_labs-> the one-hot representation of the predicted phonemens from the attention LSTM network   
		
	brief-> function to return the corrected version of the predicted sequence based on the DDA architecture.
	
	return-> outlabs3-> corrected phoneme sequence
'''

	with tf.variable_scope("recurrence") as scope:
		W_outlabs = tf.get_variable("weight_outlabs",[64,32],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_outlabs = tf.get_variable("bias_outlabs",[32], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		W_outlabs1 = tf.get_variable("weight_outlabs1",[32,64],dtype =  tf.float64, initializer = tf.random_normal_initializer(0,0.01))
		b_outlabs1 = tf.get_variable("bias_outlabs1",[64], dtype =  tf.float64,initializer = tf.constant_initializer(0.0))
		scope.reuse_variables()
		outlabs2 = tf.tanh(tf.add(tf.matmul(out_labs,W_outlabs),b_outlabs))
		print ("outlabs2", outlabs2.get_shape())
		outlabs3 = tf.tanh(tf.add(tf.matmul(outlabs2,W_outlabs1),b_outlabs1))
		print ("outlabs3", outlabs3.get_shape())	
	return outlabs3
		

features = tf.placeholder(tf.float64, shape=[batch_size, None,19])
labs = tf.placeholder(tf.int64,shape= [None])
seq_len = tf.placeholder(tf.bool,shape= [None])
mask = seq_len
labels =  labs
# mask = tf.squeeze(seq_len)
# labels = tf.squeeze(labs)
initial_states = tf.zeros([batch_size,306+output_dim],dtype = tf.float64)
features_1 = tf.transpose(features, perm=[1,0,2])
print ("features",features.get_shape())
scan_out = tf.scan(recurrent_step, features_1, initializer = initial_states,back_prop=True) ##TBD required
states = scan_out[:,:,:306]
opts = scan_out[:,:,306:]
print("opts",opts.get_shape())
outputs = tf.transpose(opts,perm = [1,0,2])
print("output",outputs.get_shape())
# out = last_relevant(outputs, seq_len)
# out = out.astype(np.int32)
# out = tf.to_int32(out, name='ToInt32')
# out_labs= tf.argmax(outputs,2)
# out_labs= tf.argmax(out, dim= 2)
# out_labs = tf.squeeze(out_labs)
out_labs = tf.reshape(outputs,[-1,output_dim])
print "labels: ",out_labs
# mask = tf.reshape(mask,[-1])
out_labs = tf.boolean_mask(out_labs, mask)
# out_labs = last_relevant(out_labs,mask)
# out_labs = tf.to_int32(out_labs, name='ToInt32')
# out_labs = tf.cast(out_labs,tf.int32)
# labs= tf.squeeze(la bs)
# labels = tf.reshape(labyels,[-1])
labels = tf.boolean_mask(labels,mask)
#print "labels: ",labels
# labs= tf.squeeze(labs)
# labs = tf.cast(labs,tf.int32)
# labs = tf.to_int64(labs,name = 'ToInt64')

#out_labs = dda(out_labs)
print "labs: ",labels
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = out_labs, labels = labels)
train_op =  tf.train.RMSPropOptimizer(0.008).minimize(cost)
out_labss1 = tf.argmax(out_labs,1)
##########crossentropyloss#########
equal = tf.reduce_sum(tf.to_float(tf.equal(out_labss1, labels)),name = 'ToFloat')
not_equal = tf.reduce_sum(tf.to_float(tf.not_equal(out_labss1, labels)),name = 'ToFloat')
accuracy = (equal)/(equal+not_equal)
#out_labs = out_labs.reshape(out_labs.size, 1)	
sess = tf.Session()

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

filelist = listdir(mfccfolder)
numbatch = len(filelist)/batch_size
# f,l = get_data(1,filelist)
print "batch: ",numbatch
epochs = 50
#save_path = saver.save(sess, "/workspace/model.ckpt")
#print("Model saved in file: %s" % save_path)
for j in range(epochs):
	print "epoch: ",j
	accur = 0
	for i in range(numbatch):
		f, l, seq_l  = get_data(i,filelist)
		# seq_len =  np.reshape(seq_len, (batch_size,-1,1))
		l = l.astype(np.int64)
		l = np.reshape(l, (batch_size,-1))
		l = np.reshape(l,[-1])
		seq_l = np.reshape(seq_l, [-1])
		l= l-1
		#print "l:",l
		_,outl,lax,acc = sess.run([train_op,out_labss1,labels,accuracy], feed_dict={features:f, labs:l, seq_len:seq_l})
		accur  = accur + acc
	print "accuracy:", accur/numbatch
	#outll = tf.convert_to_tensor(outl)
	#print("output",outll.get_shape())
	#labl = tf.convert_to_tensor(lax)
	#print("lab",labl.get_shape())
	#print "outlab:",outl
	#print "lab:",lax
	aa1 = np.array(outl)
	bb1 = np.zeros((695, 64))
	bb1[np.arange(695), aa1] = 1
	np.savetxt('one.csv', bb1, delimiter=',')
	aa2 = np.array(lax)
	bb2 = np.zeros((695, 64))
	bb2[np.arange(695), aa2] = 1
	np.savetxt('onelab.csv', bb2, delimiter=',')
	#with open(outfile+'.x.betas','a') as f_handle:
    	#np.savetxt(f_handle,dataPoint)
	np.savetxt('out.csv', outl)
	np.shape(outl)
	np.savetxt('lab.csv', lax)	
	ac = 0
	for i in range(numbatch):
		f, l, seq_l  = get_data(i,filelist)
		# seq_len =  np.reshape(seq_len, (batch_size,-1,1))
		l = l.astype(np.int64)
		l = np.reshape(l, (batch_size,-1))
		l = np.reshape(l,[-1])
		seq_l = np.reshape(seq_l, [-1])
		l= l-1
		ip = sess.run([accuracy],feed_dict={features:f, seq_len:seq_l})
		print "eq:", ip
		f = open( 'outt.csv', 'a' )
		f.write( 'dict = ' + repr(ip) + '\n' )
		f.close()

	


















