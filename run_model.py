import sys
import os
import numpy as np
import tensorflow as tf #tf.__version__ = '2.5.0'
import random
import time
import unicodedata
from collections import defaultdict

if not os.path.exists('model_fits'):
    os.makedirs('model_fits')

K = int(sys.argv[1])
fold = int(sys.argv[2])
seed = int(sys.argv[3])

n_folds = 6

assert(fold in range(0,n_folds))

text = [l.strip('\n').split('\t') for l in open('cdial_wordlist.csv','r')]

lang_counts = defaultdict(int)
for l in text:
    lang_counts[l[0]] += 1

#lang_key = [l.strip('\n').split('\t') for l in open('lang_key.tsv','r')]
#lang_key = {l[3]:l[2] for l in lang_key[1:]}

#all romani, nuristani, and middle indo-aryan langs
langs_to_exclude = ['loma1235',
                    'doma1258',
                    'bohe1241',
                    'balk1252',
                    'doma1260',
                    'wels1246',
                    'vlax1238',
                    'sepe1242',
                    'tran1280',
                    'abru1239',
                    'kara1460',
                    'nawa1257',
                    'tavr1235',
                    'qina1238',
                    'sint1235',
                    'kald1238',
                    'nort2655',
                    'span1238',
                    'kati1270',
                    'ashk1246',
                    'treg1243',
                    'pras1239',
                    'waig1243',
                    'pali1273',
                    'maha1305']

langs_to_keep = [l for l in lang_counts.keys() if lang_counts[l] > 200 and l not in langs_to_exclude]

print('{} languages retained'.format(len(langs_to_keep)))

text = [l for l in text if l[0] in langs_to_keep]

lang_raw = [l[0] for l in text]
pos_raw = [l[3] for l in text]
input_raw = [['<bos>']+list(unicodedata.normalize('NFD',l[2].lower()))+['<eos>'] for l in text]
output_raw = [['<bos>']+list(unicodedata.normalize('NFD',l[1].lower()))+['<eos>'] for l in text]

N = len(text)

lang_types = sorted(set(lang_raw))
pos_types = sorted(set(pos_raw))
input_types = sorted(set([s for w in input_raw for s in w]))
output_types = sorted(set([s for w in output_raw for s in w]))

n_lang = len(lang_types)
n_pos = len(pos_types)
n_input = len(input_types)
n_output = len(output_types)

print(N)
print(n_lang)

T_i = max([len(l) for l in input_raw])
T_o = max([len(l) for l in output_raw])

lang_id = np.array([lang_types.index(l) for l in lang_raw])
pos_id = np.array([pos_types.index(l) for l in pos_raw])
input_seq = np.zeros([N,T_i])
output_seq = np.zeros([N,T_o])
for i,w in enumerate(input_raw):
  for j,s in enumerate(w):
    input_seq[i,j] = input_types.index(s)+1

for i,w in enumerate(output_raw):
  for j,s in enumerate(w):
    output_seq[i,j] = output_types.index(s)+1

encoder_input = input_seq
decoder_input = output_seq[:,:-1]
decoder_output = output_seq[:,1:]

T_o = decoder_output.shape[1]

print(T_i)
print(T_o)

hidden_dim = 64
embed_dim = 64

np.random.seed(seed)
indices = np.arange(N)
np.random.shuffle(indices)

fold_inds = [[indices[i] for i in range(j,k)] for (j,k) in zip(list(range(0,N,int(N/n_folds)))+[N][:-1],(list(range(0,N,int(N/n_folds)))+[N])[1:])]
train_inds = [i for j in range(n_folds) for i in fold_inds[j] if j != fold]
test_inds = [i for j in range(n_folds) for i in fold_inds[j] if j == fold]

class mixtureED(tf.keras.models.Model):
    def __init__(self):
        super(mixtureED,self).__init__()
        self.log_p_z_lang_embed = tf.keras.layers.Embedding(n_lang+1,K)
        self.dec_lang_embed = tf.keras.layers.Embedding(n_lang+1,embed_dim)
        self.enc_embed = tf.keras.layers.Embedding(n_input+1,embed_dim)
        self.component_embed = tf.keras.layers.Embedding(K,embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(n_pos,embed_dim)
        self.dec_embed = tf.keras.layers.Embedding(n_output+1,embed_dim)
        self.lstm_enc = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_dim,return_sequences=True))
        self.lstm_dec = tf.keras.layers.LSTM(units=hidden_dim,return_sequences=True)
        self.V = tf.keras.layers.Dense(units=hidden_dim)
        self.W = tf.keras.layers.Dense(units=hidden_dim*3)
        self.U = tf.keras.layers.Dense(units=n_output)
        self.dropout_comp = tf.keras.layers.Dropout(.2, input_shape=(embed_dim,))
        self.dropout_lang = tf.keras.layers.Dropout(.2, input_shape=(embed_dim,))
    def call(self,inputs):
        lang_id,pos_id,encoder_input,decoder_input = inputs
        log_p_z_lang_embedded = self.log_p_z_lang_embed(lang_id)
        log_p_z = tf.nn.log_softmax(log_p_z_lang_embedded,-1)
        dec_lang_embedded = self.dropout_lang(self.dec_lang_embed(lang_id))
        enc_in_embedded = self.enc_embed(encoder_input)
        dec_in_embedded = self.dec_embed(decoder_input)
        pos_embedded = self.pos_embed(pos_id)
        dec_in_embedded_concat = tf.concat([tf.tile(tf.expand_dims(tf.concat([dec_lang_embedded,pos_embedded],-1),-2),(1,T_o,1)),dec_in_embedded],-1)
        h_enc = tf.stack([self.lstm_enc(tf.concat([tf.tile(tf.expand_dims(tf.expand_dims(self.dropout_comp(self.component_embed(k)),-2),-2),(enc_in_embedded.shape[0],T_i,1)),enc_in_embedded],-1)) for k in range(K)],-3)
        h_dec = tf.tile(tf.expand_dims(self.lstm_dec(dec_in_embedded_concat),-3),(1,K,1,1))
        alignment_probs = tf.nn.log_softmax(tf.einsum('nktj,nksj->nkst',h_dec,self.V(h_enc)),-2)
        h_enc_rep = tf.tile(tf.expand_dims(h_enc,-2),(1,1,1,T_o,1))
        h_dec_rep = tf.tile(tf.expand_dims(h_dec,-3),(1,1,T_i,1,1))
        h_rep = tf.concat([h_enc_rep,h_dec_rep],-1)
        #alignment_probs_ = []
        #for i in range(T_o):
        #    if i == 0:
        #        align_prev_curr = alignment_probs[:,:,:,i]
        #    if i > 0:
        #        align_prev_curr = tf.einsum('nx,ny->nxy',alignment_probs[:,:,:,i],alignment_probs_[i-1])
        #        align_prev_curr *= struc_zeros
        #        align_prev_curr = tf.reduce_sum(align_prev_curr,1)+1e-6
        #        align_prev_curr /= tf.reduce_sum(align_prev_curr,-1,keepdims=True)
        #    alignment_probs_.append(align_prev_curr)
        #alignment_probs_ = tf.stack(alignment_probs_,-1)
        emission_probs = tf.nn.log_softmax(self.U(tf.nn.tanh(self.W(h_rep))),-1)
        pred_out = tf.reduce_logsumexp(tf.expand_dims(alignment_probs,-1)+emission_probs,-3)
        return(log_p_z,pred_out)

model = mixtureED()

learning_rate = 1e-3
batch_size = 32
epochs = 200

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def get_p_z(i):
    log_p_z,pred_out = model([lang_id[i:i+1],pos_id[i:i+1],encoder_input[i:i+1],decoder_input[i:i+1]])
    return(np.exp(log_p_z))

def decode_sequence(i,k):
    out_str = ['<bos>']
    decoder_in = np.zeros([1,T_o])
    decoder_in[0,0] = output_types.index(out_str[0])+1
    for t in range(T_o):
        dec_out = model([lang_id[i:i+1],pos_id[i:i+1],encoder_input[i:i+1],decoder_in])[1][:,k,:,:]
        z = np.argmax(dec_out[0,t,:])
        decoder_in[0,t+1] = z+1
        out_str.append(output_types[z])
        if output_types[z] == '<eos>' or t == T_o-2:
            break
    return(out_str)

val_losses = []
idx = train_inds
N_ = len(idx)
tolerance = 0
for epoch in range(epochs):
    np.random.shuffle(idx)
    train_idx = idx[:int(.9*N_)]
    val_idx = idx[int(.9*N_):]
    N__ = len(train_idx)
    step = 0
    epoch_losses = []
    for (i,j) in list(zip(list(range(0,N__,batch_size)),list(range(batch_size,N__,batch_size))+[N__])):
        start = time.time()
        with tf.GradientTape() as tape:
            batch_idx = train_idx[i:j]
            X = [lang_id[batch_idx],pos_id[batch_idx],encoder_input[batch_idx],decoder_input[batch_idx]]
            y = tf.one_hot(decoder_output[batch_idx],n_output+1)[:,:,1:]
            log_p_z,pred_out = model(X)
            losses_z = log_p_z + tf.reduce_sum(pred_out*tf.expand_dims(y,-3),[-1,-2])
            loss_value = -tf.reduce_mean(tf.reduce_logsumexp(losses_z,-1))
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        step += 1
        end = time.time()
        print(epoch,step,np.mean(epoch_losses),end-start)
        epoch_losses.append(loss_value)
    N__ = len(val_idx)
    #val_loss = 0
    #for (i,j) in list(zip(list(range(0,N__,1)),list(range(1,N__,1))+[N__])):
    #    batch_idx = val_idx[i:j]
    #    X = [lang_id[batch_idx],pos_id[batch_idx],encoder_input[batch_idx],decoder_input[batch_idx]]
    #    y = tf.one_hot(decoder_output[batch_idx],n_output+1)[:,:,1:]
    #    log_p_z,pred_out = model(X)
    #    losses_z = log_p_z + tf.reduce_sum(pred_out*tf.expand_dims(y,-3),[-1,-2])
    #    loss_value = -tf.reduce_sum(tf.reduce_logsumexp(losses_z,-1))
    #    val_loss += loss_value
    #    print(val_loss)
    X = [lang_id[val_idx],pos_id[val_idx],encoder_input[val_idx],decoder_input[val_idx]]
    y = tf.one_hot(decoder_output[val_idx],n_output+1)[:,:,1:]
    log_p_z,pred_out = model(X)
    losses_z = log_p_z + tf.reduce_sum(pred_out*tf.expand_dims(y,-3),[-1,-2])
    val_loss = -tf.reduce_sum(tf.reduce_logsumexp(losses_z,-1))
    val_losses.append(val_loss/N__)
    if epoch > 0:
        if val_loss < val_losses[epoch-1]:
            tolerance += 1
        else:
            tolerance = 0
    model.save_weights('model_fits/{}_{}_{}.h5'.format(K,fold,seed))
    print("epoch mean loss: {}".format(np.mean(epoch_losses)))
    #inds = random.sample(test_inds,50)
    #for i in inds:
    #    print(lang_raw[i],get_p_z(i),''.join(input_raw[i]),''.join(output_raw[i][1:-1]),' '.join([''.join(decode_sequence(i,k)[1:-1]) for k in range(K)]))
    if tolerance == 5:
        break

print("stopped after {} epochs".format(epoch))
