import torch
import torch.nn as nn

class Conv_block(nn.Module):

    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):

        super(Conv_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):

        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out       

class SE_block(nn.Module):

    def __init__(self,in_layer, out_layer):

        super(SE_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)

        return x_out

class RE_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(RE_block, self).__init__()
        
        self.cbr1 = Conv_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = Conv_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = SE_block(out_layer, out_layer)
    
    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)

        return x_out   

class UNET_1D(nn.Module):

    def __init__(
        self, 
        n_classes, 
        input_dim=1, 
        layer_n=128, 
        kernel_size=7, 
        depth=3
    ):

        super(UNET_1D, self).__init__()

        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.n_classes = n_classes
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,5, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,5, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.cbr_up1 = Conv_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = Conv_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = Conv_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')
        
        self.outcov = nn.Conv1d(self.layer_n, self.n_classes, kernel_size=self.kernel_size, stride=1, padding=3)

        self.adp = torch.nn.AdaptiveMaxPool1d(output_size=1)

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):

        block = []
        block.append(Conv_block(input_layer, out_layer, kernel, stride, 1))

        for i in range(depth):
            block.append(RE_block(out_layer,out_layer,kernel,1))

        return nn.Sequential(*block)
            
    def forward(self, x):
        
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)

        #############Decoder####################
        
        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        
        out = nn.functional.softmax(out,dim=1)
        
        return out

    def encode(self, x):

        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        out_3 = self.layer4(x)

        out_0 = self.adp(out_0)
        out_1 = self.adp(out_1)
        out_2 = self.adp(out_2)
        out_3 = self.adp(out_3)

        x = torch.cat([out_0, out_1, out_2, out_3], 1)

        return x

    def encoder(self, x):

        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)

        return [x, out_0, out_1, out_2]

    def decoder(self, inputs):

        x = inputs[0]
        out_0 = inputs[1]
        out_1 = inputs[2]
        out_2 = inputs[3]

        #############Decoder####################
        
        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        
        out = nn.functional.softmax(out,dim=1)
        
        return out

class Classification_Head(nn.Module):

    def __init__(self, abridged=False, x=None, encoder=None, n_classes=7):

        super(Classification_Head, self).__init__()

        self.adp = torch.nn.AdaptiveMaxPool1d(output_size=1)

        if abridged:
            x = encoder.encode(x)
            x = self.adp(x)
            x = torch.flatten(x)
            shape = x.shape[0]

            self.fc1 = nn.Linear(shape, 512)

        else:

            self.fc1 = nn.Linear(1280, 512)

        self.fc2 = nn.Linear(512, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.adp(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return torch.nn.functional.sigmoid(x)

    def penultimate_layer(self, x):

        x = self.adp(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.relu(x)

        return x

class PTB_Model(nn.Module):

    def __init__(
        self, 
        encoder, 
        head,
        n_classes=5
        ):

        super(PTB_Model, self).__init__()

        self.encoder = encoder
        self.head = head
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.encoder.encode(x)
        x = self.head.penultimate_layer(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return torch.nn.functional.sigmoid(x)

    def encode(self, x):

        return self.encoder.encoder(x)


if __name__ == "__main__":

    model = UNET_1D(11)
    X = torch.ones((8, 1, 120000))
    y = model.encode(X)
    print(y.shape)