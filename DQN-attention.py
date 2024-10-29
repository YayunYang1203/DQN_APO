import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from APO_func import APO_func

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
from collections import deque
import tensorflow as tf
from keras import layers, models
from Job_shop import Situation
from keras.optimizers import Adam
import threading
import multiprocessing


def train(machine, e_ave, job_insert):
    d = DQN()
    Processing_time, A, D, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL = d.Instance_Generator(
        machine, e_ave, job_insert)

    Fid = 1
    dim = 2
    pop_size = 100
    iter_max = 200
    Xmin = -100
    Xmax = 100

    bestProtozoa, bestFit, record_time, best_learning_rate, best_epsilon, additional_value = APO_func(Fid, dim,
                                                                                                      pop_size,
                                                                                                      iter_max, Xmin,
                                                                                                      Xmax)

    protozoa = np.random.uniform(low=Xmin, high=Xmax, size=(pop_size, dim))
    protozoa[:, -2] = np.random.uniform(0.001, 0.1, size=pop_size)
    protozoa[:, -1] = np.random.uniform(0.1, 0.9, size=pop_size)

    Total_tard, Total_uk_ave, Total_makespan = d.main(J_num, M_num, O_num, J, Processing_time, D, A,
                                                      Change_cutter_time,
                                                      Repair_time, EL)

    tard_ave = sum(Total_tard) / d.L
    uk_ave = sum(Total_uk_ave) / d.L
    makespan_ave = sum(Total_makespan) / d.L
    std1 = np.std(Total_tard)
    std2 = np.std(Total_uk_ave)
    std3 = np.std(Total_makespan)
    print(f"{tard_ave},{std1},{uk_ave},{std2},{makespan_ave},{std3}")

    return Total_tard, Total_uk_ave, Total_makespan

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

class TransformerLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)  # Accessing the method from shared location

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Dense(d_model)

        self.enc_layers = [TransformerLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

        # 将positional_encoding移到TransformerEncoder中
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, self.d_model)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]  # 使用位置编码

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * angle_rates



    def point_wise_feed_forward_network(d_model, dff):
        return tf.keras.Sequential([
               tf.keras.layers.Dense(dff, activation='relu'),
               tf.keras.layers.Dense(d_model)
               ])

class TransformerLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)  # Accessing the method from shared location

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2



class DQN:
    def __init__(self,):
        self.Hid_Size1 = 10
        self.Hid_Size2 = 50
        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(num_layers=2, d_model=6, num_heads=2, dff=32, input_vocab_size=6,
                                                      maximum_position_encoding=10000)

        # ------------Hidden layer=3  10 nodes each layer--------------
        model = models.Sequential()
        model.add(layers.Input(shape=(6,)))
        model.add(layers.Dense(self.Hid_Size1, name='l1'))
        model.add(layers.Dense(self.Hid_Size1, name='l2'))
        model.add(layers.Dense(self.Hid_Size1, name='l3'))
        model.add(layers.Dense(2, name='l4'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001))
        inputs = layers.Input(shape=(6,))
        # ------------Hidden layer=7 50 nodes each layer--------------
        model1 = models.Sequential()
        model1.add(layers.Input(shape=(7,)))
        model1.add(layers.Dense(self.Hid_Size2, name='l1'))
        model1.add(layers.Dense(self.Hid_Size2, name='l2'))
        model1.add(layers.Dense(self.Hid_Size2, name='l3'))
        model1.add(layers.Dense(self.Hid_Size2, name='l4'))
        model1.add(layers.Dense(self.Hid_Size2, name='l5'))
        model1.add(layers.Dense(self.Hid_Size2, name='l6'))
        model1.add(layers.Dense(self.Hid_Size2, name='l7'))
        model1.add(layers.Dense(9, name='l8'))
        model1.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001))
        # # model.summary()
        self.model = model
        self.model1 = model1

        #------------Q-network Parameters-------------
        self.act_dim=[1,2,3,4,5,6,7,8,9]                        #output
        self.obs_n=[0,0,0,0,0,0]                    #input
        self.gama = 0.95  # γ
        self.global_step = 0
        self.update_target_steps = 200  # update step: C
        self.target_model = self.model
        self.target_model1 = self.model1

        # -------------------Agent-------------------
        self.e_greedy = 0.6
        self.e_greedy_decrement = 0.0001
        self.L = 20  # Number of training episodes L = 20

        # ---------------Replay Buffer---------------
        self.buffer = deque(maxlen=2000)
        self.Batch_size = 16  # Batch Size of Samples to perform gradient descent




    def replace_target(self):
        self.target_model.get_layer(name='l1').set_weights(self.model.get_layer(name='l1').get_weights())
        self.target_model.get_layer(name='l2').set_weights(self.model.get_layer(name='l2').get_weights())
        self.target_model.get_layer(name='l3').set_weights(self.model.get_layer(name='l3').get_weights())
        self.target_model.get_layer(name='l4').set_weights(self.model.get_layer(name='l4').get_weights())

        self.target_model1.get_layer(name='l1').set_weights(self.model1.get_layer(name='l1').get_weights())
        self.target_model1.get_layer(name='l2').set_weights(self.model1.get_layer(name='l2').get_weights())
        self.target_model1.get_layer(name='l3').set_weights(self.model1.get_layer(name='l3').get_weights())
        self.target_model1.get_layer(name='l4').set_weights(self.model1.get_layer(name='l4').get_weights())
        self.target_model1.get_layer(name='l5').set_weights(self.model1.get_layer(name='l5').get_weights())
        self.target_model1.get_layer(name='l6').set_weights(self.model1.get_layer(name='l6').get_weights())
        self.target_model1.get_layer(name='l7').set_weights(self.model1.get_layer(name='l7').get_weights())
        self.target_model1.get_layer(name='l8').set_weights(self.model1.get_layer(name='l8').get_weights())

    def replay(self):
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()
        # replay the history and train the model
        minibatch = random.sample(self.buffer, self.Batch_size)
        for state, action, reward, next_state, reward_id, done in minibatch:
            target = reward
            target1 = reward
            if not done:
                output = self.target_model.predict(next_state)
                k = np.max(output)
                target = (reward + self.gama * np.argmax(output))
                next_state1 = np.expand_dims(np.append(next_state[0], k), 0)
                target1 = (reward + self.gama * np.argmax(self.target_model1.predict(next_state1)))
            target_f = self.model.predict(state)
            k = np.max(target_f)
            state1 = np.expand_dims(np.append(state[0], k), 0)
            target_f1 = self.model1.predict(state1)

            print("Shape of target_f:", target_f.shape)  # 调试语句
            print("Shape of target_f1:", target_f1.shape)  # 调试语句
            print("Reward id:", reward_id)  # 调试语句

            # 确保 reward_id 在合理范围内
            if reward_id < target_f.shape[1]:
                target_f[0][reward_id] = target
            else:
                print("Invalid reward_id:", reward_id)

            if action < target_f1.shape[1]:
                target_f1[0][action] = target1
            else:
                print("Invalid action:", action)

            self.model.fit(state, target_f, epochs=1, verbose=0)
            self.model1.fit(state1, target_f1, epochs=1, verbose=0)
        self.global_step += 1

    def Select_action(self,obs):
        # obs = np.expand_dims(obs,0)
        if random.random() < self.e_greedy:
            rt = random.randint(0, 1)
            act = random.randint(0, 8)
        else:
            output = self.transformer_encoder(obs)  # 使用Transformer模型进行预测
            rt = np.argmax(output)
            input = np.expand_dims(np.append(obs[0], np.argmax(output)),0)
            act = np.argmax(self.model1.predict(input))
        self.e_greedy = max(
            0.01, self.e_greedy - self.e_greedy_decrement)
        return act,rt

    def _append(self, exp):
        self.buffer.append(exp)

    def Instance_Generator(self, M_num, E_ave, New_insert):
        '''
        :param M_num: Machine Number
        :param Initial_job: initial job number
        :param E_ave
        :return: Processing time,A:New Job arrive time,
                                    D:Deliver time,
                                    M_num: Machine Number,
                                    Op_num: Operation Number,
                                    J_num:Job NUMBER
                                    EL:ergency level of each job
        '''
        E_ave = E_ave
        Initial_Job_num = 5
        Op_num = [random.randint(1, 20) for i in range(New_insert + Initial_Job_num)]
        Processing_time = []
        for i in range(Initial_Job_num + New_insert):
            Job_i = []
            for j in range(Op_num[i]):
                k = random.randint(1, M_num - 2)
                T = list(range(M_num))
                random.shuffle(T)
                T = T[0:k + 1]
                O_i = list(np.ones(M_num) * (-1))
                for M_i in range(len(O_i)):
                    if M_i in T:
                        O_i[M_i] = random.randint(1, 50)
                Job_i.append(O_i)
            Processing_time.append(Job_i)
        A1 = [0 for i in range(Initial_Job_num)]
        A = np.random.exponential(E_ave, size=New_insert)
        A = [int(A[i]) for i in range(len(A))]  # New Insert Job arrive time
        A1.extend(A)
        EL = [random.randint(1,3) for i in range(len(A1))]
        T_ijave = []
        for i in range(Initial_Job_num + New_insert):
            Tad = []
            for j in range(Op_num[i]):
                T_ijk = [k for k in Processing_time[i][j] if k != -1]
                Tad.append(sum(T_ijk) / len(T_ijk))
            T_ijave.append(sum(Tad))
        D1 = [int((0.2 + 0.5 * EL[i]) * T_ijave[i]) for i in range(Initial_Job_num)]
        D = [int(A1[i] + (0.2 + 0.5 * EL[i]) * T_ijave[i]) for i in range(Initial_Job_num, Initial_Job_num + New_insert)]
        D1.extend(D)
        O_num = sum(Op_num)
        J = dict(enumerate(Op_num))
        J_num = Initial_Job_num + New_insert

        Change_cutter_time = list(np.zeros(M_num))
        Repair_time = list(np.zeros(M_num))
        for i in range(M_num):
            Change_cutter_time[i] = random.randint(1, 50)
            Repair_time[i] = random.randint(1, 99)

        return Processing_time, A1, D1, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL


    def main(self, J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time, EL):
        k = 0
        x = []
        Total_tard = []
        Total_makespan = []
        Total_uk_ave = []
        TR = []

        for i in range(self.L):
            i = 0
            Total_reward = 0
            x.append(i + 1)
            print('-----------------------start', i + 1, 'training------------------------------')
            obs = [0 for i in range(6)]
            obs = np.expand_dims(obs, 0)
            done = False
            Sit = Situation(J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time, EL)

            for i in range(O_num):
                k += 1
                # print(obs)
                at, rt = self.Select_action(obs)
                # print(at)

                if at == 0:
                    at_trans = Sit.rule1()
                if at == 1:
                    at_trans = Sit.rule2()
                if at == 2:
                    at_trans = Sit.rule3()
                if at == 3:
                    at_trans = Sit.rule4()
                if at == 4:
                    at_trans = Sit.rule5()
                if at == 5:
                    at_trans = Sit.rule6()
                if at == 6:
                    at_trans = Sit.rule7()
                if at == 7:
                    at_trans = Sit.rule8()
                if at == 8:
                    at_trans = Sit.rule9()
                # at_trans=self.act[at]
                print('The', i, 'th operation>>', 'select action:', at, ' ', 'job ', at_trans[0],
                      'is assigned for machine ', at_trans[1])
                Sit.scheduling(at_trans)
                obs_t = Sit.Features()

                if i == O_num - 1:
                    done = True
                # obs = obs_t
                obs_t = np.expand_dims(obs_t, 0)
                # obs = np.expand_dims(obs, 0)
                # print(obs,obs_t)
                if 0 == rt:
                    r_t = Sit.reward1(obs[0][5], obs[0][4], obs_t[0][5], obs_t[0][4])
                else:
                    r_t = Sit.reward2(obs[0][0], obs_t[0][0])
                self._append((obs, at, r_t, obs_t, rt, done))
                if k > self.Batch_size:
                    # batch_obs, batch_action, batch_reward, batch_next_obs,done= self.sample()
                    self.replay()
                Total_reward += r_t
                obs = obs_t


            total_tadiness = 0
            makespan = 0
            uk_ave = sum(Sit.UK) / M_num
            Job = Sit.Jobs
            E = 0
            K = [i for i in range(len(Job))]
            End = []
            for Ji in range(len(Job)):
                endTime = max(Job[Ji].End)
                makespan = max(makespan, endTime)
                End.append(endTime)
                if max(Job[Ji].End) > D[Ji]:
                    total_tadiness += abs(max(Job[Ji].End) - D[Ji])
            print('<<<<<<<<<-----------------total_tardiness:', total_tadiness, '------------------->>>>>>>>>>')
            Total_tard.append(total_tadiness)
            print('<<<<<<<<<-----------------uk_ave:', uk_ave, '------------------->>>>>>>>>>')
            Total_uk_ave.append(uk_ave)
            print('<<<<<<<<<-----------------makespan:', makespan, '------------------->>>>>>>>>>')
            Total_makespan.append(makespan)
            print('<<<<<<<<<-----------------reward:', Total_reward, '------------------->>>>>>>>>>')
            TR.append(Total_reward)
            data = {'Total Tardiness': Total_tard,
                    'UK Average': Total_uk_ave,
                    'Makespan': Total_makespan,
                    'Total Reward': TR}
            df = pd.DataFrame(data)

            excel_file = 'E:\迅雷下载\code3DDQN\\results\8-50-20.xlsx'
            df.to_excel(excel_file, index=False)
            print("Results saved to", excel_file)

            # plt.plot(K,End,color='y')
            # plt.plot(K,D,color='r')
            # plt.show()
        # plt.plot(x,Total_tard)
        # plt.show()
        return Total_tard, Total_uk_ave, Total_makespan



def call_back(v):
    print('----> callback pid:', os.getpid(), ',tid:', threading.currentThread().ident, ',v:', v)

if __name__ == '__main__':
    Total_Machine = [8, 12, 16]
    Job_insert = [20, 30, 40]
    E_ave = [50, 100, 200]
    machine = Total_Machine[0]
    e_ave = E_ave[0]
    job_insert = Job_insert[2]
    train(machine, e_ave, job_insert)


    # pool = multiprocessing.Pool(27)
    # results = [pool.apply_async(train, args=(machine, e_ave, job_insert, ), callback=call_back) for e_ave in E_ave for machine in Total_Machine for job_insert in Job_insert]
    # pool.close()
    # pool.join()
