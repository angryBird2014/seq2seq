from __future__ import print_function
import numpy as np
import tensorflow as tf

vocab_size = 256
target_vocab_size = vocab_size
learning_rate = 0.1
buckets = [(10,10)]
PAD=[0]
batch_size  = 10

input_data = [list(map(ord,"hello")) + PAD * 5] * batch_size
targe_data = [list(map(ord,"world")) + PAD * 5] * batch_size
target_weight = [[1.0]*6 + [0.0]*4] * batch_size

class BabySeq2Seq(object):
    def __init__(self,source_vocab_size,target_vocab_size,buckets,size,num_layers,batch_size):
        self.buckets = buckets
        self.batch_size = batch_size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        cell = single_cell = tf.nn.rnn_cell.GRUCell(size)

        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        def seq2seq_f(encoder_inputs,decoder_inputs,do_decoder):
            out = tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs,decoder_inputs,cell,
                num_encoder_symbols = source_vocab_size,num_decoder_symbols = target_vocab_size,
                embedding_size = size,feed_previous = do_decoder
            )
            return out
        self.encoder_input = []
        self.decoder_input = []
        self.target_weights = []
        for i in range(buckets[-1][0]):
            self.encoder_input.append(tf.placeholder(tf.int32,shape=[None],name="encoder{0}".format(i)))
        for i in range(buckets[-1][1]):
            self.decoder_input.append(tf.placeholder(tf.int32,shape=[None],name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32,shape=[None],name="weight{0}".format(i)))

        target = [self.decoder_input[i] for i in range(len(self.decoder_input))]

        self.output,self.loss = tf.nn.seq2seq.model_with_buckets(self.encoder_input,self.decoder_input,
                                                                 target,self.target_weights,buckets,
                                                                 lambda x,y:seq2seq_f(x,y,False))

        params = tf.trainable_variables()
        self.update = []
        for b in range(len(buckets)):
            self.update.append(tf.train.AdadeltaOptimizer(learning_rate).minimize(self.loss[b]))

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self,session,encoder_input,decoder_input,target_weights,test):
        bucket_id = 0
        encoder_size,decoder_size = buckets[bucket_id]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_input[l].name] = encoder_input[l]
        for l in range(decoder_size):
            input_feed[self.decoder_input[l].name] = decoder_input[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        last_target = self.decoder_input[decoder_size-1].name
        input_feed[last_target] = np.zeros([self.batch_size],dtype=np.int32)

        if not test:
            output_feed = [self.update[bucket_id],self.loss[bucket_id]]
        else:
            output_feed = [self.loss[bucket_id]]
            for l in range(decoder_size):
                output_feed.append(self.output[bucket_id][l])

        outputs = session.run(output_feed,input_feed)
        if not test:
            return outputs[0], outputs[1]  # Gradient norm, loss
        else:
            return outputs[0], outputs[1:]  # loss, outputs.

def decode(bytes):
    return "".join(map(chr,bytes)).replace('\x00','').replace('\n','')

def test():
    perplexity, outputs = model.step(session, input_data, targe_data, target_weight, test=True)
    words = np.argmax(outputs, axis=2)  # shape (10, 10, 256)
    word = decode(words[-5])
    print("step %d, perplexity %f, output: hello %s?" % (step, perplexity, word))
    if word == "world":
        print(">>>>> success! hello " + word + "! <<<<<<<")

if __name__ == '__main__':

    step = 0
    test_step = 1
    with tf.Session() as session:
        model = BabySeq2Seq(vocab_size,target_vocab_size,buckets,size=10,num_layers=1,batch_size=batch_size)
        session.run(tf.global_variables_initializer())
        while step <= 1000:
            model.step(session,input_data,targe_data,target_weight,test=False)
            if step % test_step == 0:
                test()
            step += 1
