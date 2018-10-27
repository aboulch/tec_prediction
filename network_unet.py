"""
Unet network

License is from https://github.com/aboulch/tec_prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from convLSTM import CLSTM_cell as Recurrent_cell

class UnetConvRecurrent(nn.Module):
    """Segnet network."""

    def __init__(self, input_nbr, num_features=8):
        """Init fields."""
        super(UnetConvRecurrent, self).__init__()


        input_size = 4
        kernel_size = 3


        self.conv11 = nn.Conv2d(input_nbr, num_features, kernel_size=3, padding=1, stride=2)
        self.convRecurrentCell1 = Recurrent_cell(num_features, num_features, kernel_size)

        self.conv21 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride=2)
        self.convRecurrentCell2 = Recurrent_cell(num_features, num_features, kernel_size)

        self.conv31 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride=2)
        self.convRecurrentCell3 = Recurrent_cell(num_features, num_features, kernel_size)

        self.convd21 = nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.convd22 = nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, padding=1)
        self.convRecurrentCelld2 = Recurrent_cell(num_features, num_features, kernel_size)

        self.convd11 = nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.convd12 = nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, padding=1)
        self.convRecurrentCelld1 = Recurrent_cell(num_features, num_features, kernel_size)

        self.convd1 = nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.convd2 = nn.ConvTranspose2d(num_features+input_nbr, input_nbr, kernel_size=3, padding=1)




    def forward(self, z, prediction_len, diff=False, predict_diff_data=None):
        """Forward method."""
        # Stage 1

        output_inner = []
        size = z.size()
        seq_len=z.size(0)

        hidden_state1=None
        hidden_state2=None
        hidden_state3=None
        hidden_stated2=None
        hidden_stated1=None
        for t in range(seq_len):#loop for every step
            x = z[t,...]

            # coder
            x1 = F.relu(self.conv11(x))
            hidden_state1=self.convRecurrentCell1(x1,hidden_state1)
            x1 = hidden_state1[0]

            x2 = F.relu(self.conv21(x1))
            hidden_state2=self.convRecurrentCell2(x2, hidden_state2)
            x2 = hidden_state2[0]

            x3 = F.relu(self.conv31(x2))
            hidden_state3=self.convRecurrentCell3(x3, hidden_state3)
            x3 = hidden_state3[0]

            y2 = F.relu(self.convd21(x3))
            y2 = torch.cat([x2,y2],1)
            y2 = F.relu(self.convd22(y2))
            hidden_stated2 = self.convRecurrentCelld2(y2, hidden_stated2)
            y2 = hidden_stated2[0]

            y1 = F.relu(self.convd11(y2))
            y1 = torch.cat([x1,y1], 1)
            y1 = F.relu(self.convd12(y1))
            hidden_stated1 = self.convRecurrentCelld1(y1, hidden_stated1)
            y1 = hidden_stated1[0]

            y = F.relu(self.convd1(y1))
            y = torch.cat([y,x], 1)
            y = self.convd2(y)

        output_inner.append(y)

        for t in range(prediction_len-1):

            if(diff):
                x = y + predict_diff_data[t,...]
            else:
                x = y

            # coder
            x1 = F.relu(self.conv11(x))
            hidden_state1=self.convRecurrentCell1(x1,hidden_state1)
            x1 = hidden_state1[0]

            x2 = F.relu(self.conv21(x1))
            hidden_state2=self.convRecurrentCell2(x2, hidden_state2)
            x2 = hidden_state2[0]

            x3 = F.relu(self.conv31(x2))
            hidden_state3=self.convRecurrentCell3(x3, hidden_state3)
            x3 = hidden_state3[0]

            y2 = F.relu(self.convd21(x3))
            y2 = torch.cat([x2,y2],1)
            y2 = F.relu(self.convd22(y2))
            hidden_stated2 = self.convRecurrentCelld2(y2, hidden_stated2)
            y2 = hidden_stated2[0]

            y1 = F.relu(self.convd11(y2))
            y1 = torch.cat([x1,y1], 1)
            y1 = F.relu(self.convd12(y1))
            hidden_stated1 = self.convRecurrentCelld1(y1, hidden_stated1)
            y1 = hidden_stated1[0]

            y = F.relu(self.convd1(y1))
            y = torch.cat([y,x], 1)
            y = self.convd2(y)

            output_inner.append(y)

        expected_size = (len(output_inner), z.size(1), z.size(2), z.size(3), z.size(4))
        current_input = torch.cat(output_inner, 0).view(expected_size)

        return current_input


    def load_from_filename(self, model_path):
        """Load weights method."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)
