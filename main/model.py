import torch.nn as nn
import torch.nn.functional as F

from aux_rnn import DynamicLSTM


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        self.num_inputs = num_inputs
        self.num_actions = num_actions

    def Model(self, for_display):
        super(Model, self).__init__()
        self.lstm_cell = nn.LSTMCell(256)

        self.A3C()

        if self.use_pixel_change:
            self.pixel_change()
            if for_display:
                self.pixel_change_for_display()

        if self.use_reward_prediction():
            self.reward_prediction()

        if self.use_value_replay():
            self.value_replay()


        #need to see if the variables are parsed here



    def A3C(self, num_inputs, num_actions):
        super(A3C, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        #self._initialize_weights()



    def _initialize_convs(w, h, state_input):
        d = 1.0 / np.sqrt(state_input * w * h)
        return torch.FloatTensor(a, b).uniform_(-d, d)


    def conv_layers(self, state_input, reuse=False):
        W_1, b_1 = self.test_convs([8, 8, 3, 16])
        W_2, b_2 = self.test_convs([4, 4, 16, 32])
        h_1 = nn.ReLU(self.conv2d(state_input, W_1, 4) + b_1)
        h_2 = nn.ReLU(self.conv2d(h_1, W_2, 2) + b_2)

        return h_2


    def lstm_layers(self, conv_output, last_action_reward_input, state_input, reuse=False):
        W_fc1, b_fc1 = self.fully_connected_layer([2592, 256])
        flat_conv_output = torch.reshape(conv_output, [-1, 2592])
        fc_conv_output = nn.ReLU(torch.matmul(flat_conv_output, W_fc1) + b_fc1)
        step_size = torch.Tensor.size(fc_conv_output)[:1]
        lstm_input = torch.cat([fc_conv_output, last_action_reward_input], 1)
        reshape_input = torch.reshape(lstm_input, [1, -1, 256+self.action_size+1])
        lstm_outputs, lstm_state = DynamicLSTM(self.lstm_cell, reshape_input)
        lstm_outputs = torch.reshape(lstm_outputs, [-1, 256])

        return lstm_outputs, lstm_state


    def policy(self, lstm_output, reuse=False):
        W_fc_p, b_fc_p = self.fully_connected_layer([256, self.action_size])
        policy = nn.Softmax(torch.matmul(lstm_outputs, W_fc_p) + b_fc_v)
        return p


    def value(self, lstm_outputs, reuse=False):
        W_fc_v, b_fc_v = self.fully_connected_layer([256, 1])
        value = torch.matmul(lstm_outputs, W_fc_v) + b_fc_v
        v = torch.reshape(value, [-1])
        return v 


    def pixel_change(self):
        self.pixel_change_input = torch.Tensor([None, 84, 84, 3])
        self.pc_last_action_reward_input = torch.Tensor([None, self.action_size+1])
        pc_conv_output = self.conv_layers(self.pc_input, reuse=False)
        pc_initial_lstm_state = self.lstm_cell.zero_state()



    def test_convs(self, weight_shape, deconv=False):
        w = weight_shape[0]
        h = weight_shape[1]
        if deconv:
            input_channel = weight_shape[3]
            output_channel = weight_shape[2]
        else:
            input_channel = weight_shape[2]
            output_channel = weight_shape[3]
        bias_shape = [output_channel]
        weight = (weight_shape, _initialize_convs(w, h, input_channel))
        bias = (bias_shape, _initialize_convs(w, h, input_channel))

        return weight, bias


    def conv2d(self, x, W, stride):
        layers = nn.Conv2d(x, w, stride)
        return layers



    def forward(self, x, hx, cx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx



    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)







