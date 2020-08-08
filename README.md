# Incepto: Adversarial ML Framework for Medical Models

![Test Image 1](logo.png)

What Can You Do?

📤  Train an OOD Detector

🔫  Attack Your Model

🏋 Run Adversarial Training

🕶 Defend Your Model Against Attacks

🤓  Visualize whole datasets, datapoints, and feature maps in a dashboard

# How to Work With Incepto:

### Set up a client object
    new_client = client(model,in_dist_dataset,ood_dataset,data_labels=["MotionSense","mHealth"],num_classes=2)


# How to Setup Your Model:

### Add a feature_list, intermediate_forward, and penultimate_forward function to help our framework extract information from your model. Check out the model class below for an example!

    class Network(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm1d(8),
                nn.Tanh(),
                nn.MaxPool1d(2, stride=2))

            self.conv2 = nn.Sequential(
                nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm1d(16),
                nn.Tanh(),
                nn.MaxPool1d(2, stride=2))

            self.conv3 = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(32),
                 nn.Tanh(),
                nn.MaxPool1d(2, stride=2))

            self.conv4 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(32),
                 nn.Tanh(),
                nn.MaxPool1d(2, stride=2))

            self.conv5 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.Tanh(),
                nn.MaxPool1d(2, stride=2))

            self.conv6 = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.Tanh(),
                nn.MaxPool1d(2, stride=2)
                )

            self.conv7 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(128),
                 nn.Tanh(),
                nn.MaxPool1d(2, stride=2))

            self.conv8 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm1d(256),
                 nn.Tanh(),
                nn.MaxPool1d(2, stride=2))

            self.fc = nn.Linear(3072,1)

            self.activation = nn.Sigmoid()



        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.conv5(out)
            out = self.conv6(out)
            out = self.conv7(out)
            out = self.conv8(out)
            out = out.view(x.shape[0], out.size(1) * out.size(2))
            logit = self.fc(out)

            return logit


        # function to extact the multiple features
        def feature_list(self, x):
            out_list = []
            out = self.conv1(x)
            out_list.append(out)

            out = self.conv2(out)
            out_list.append(out)

            out = self.conv3(out)
            out_list.append(out)

            out = self.conv4(out)
            out_list.append(out)

            out = self.conv5(out)
            out_list.append(out)

            out = self.conv6(out)
            out_list.append(out)

            out = self.conv7(out)
            out_list.append(out)

            out = self.conv8(out)
            out_list.append(out)

            out = out.view(x.shape[0], out.size(1) * out.size(2))
            y = self.fc(out)
            return y, out_list

        # function to extact a specific feature
        def intermediate_forward(self, x, layer_index):
            out = self.conv1(x)
            if layer_index >= 1:
                out = self.conv2(out)
            if layer_index >= 2:
                out = self.conv3(out)
            if layer_index >= 3:
                out = self.conv4(out)
            if layer_index >= 4:
                out = self.conv5(out)
            if layer_index >= 5:
                out = self.conv6(out)
            if layer_index >= 6:
                out = self.conv7(out)
            if layer_index >= 7:
                out = self.conv8(out)
            if layer_index >= 8:
                out = out.view(x.shape[0], out.size(1) * out.size(2))
                out = self.fc(out)         
            return out

        # function to extact the penultimate features
        def penultimate_forward(self, x):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.conv5(out)
            out = self.conv6(out)
            out = self.conv7(out)
            penultimate = self.conv8(out)
            out = penultimate.view(x.shape[0], penultimate.size(1) * penultimate.size(2))
            y = self.fc(out)

            return y, penultimate

