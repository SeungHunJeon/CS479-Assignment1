import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 64, 1))

        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, 1024, 1))



    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        x = pointcloud
        stn3_input = torch.transpose(x, 1, 2)
        stn3_output = self.stn3(stn3_input)
        x = x@stn3_output
        x = torch.transpose(x,1,2)
        x = self.mlp1(x)
        x = torch.transpose(x,1,2)
        stn64_input = torch.transpose(x,1,2)
        stn64_output = self.stn64(stn64_input)
        x = x@stn64_output
        x = torch.transpose(x,1,2)
        x = self.mlp2(x)
        max_pool = nn.MaxPool1d(kernel_size=x.shape[2])
        x = max_pool(x)
        x = torch.transpose(x, 1, 2)

        return x, stn3_output, stn64_output

        # # TODO : Implement forward function.
        # pass


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.mlp1 = nn.Sequential(nn.Linear(1024, 512),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU(),
                                   nn.Linear(512, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.3),
                                   nn.Linear(256, self.num_classes),
                                   nn.Softmax())

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """

        x = pointcloud
        feature, stn3_output, stn64_output = self.pointnet_feat(x)
        feature = feature.squeeze(1)
        output = self.mlp1(feature)
        # TODO : Implement forward function.
        return output, stn3_output, stn64_output


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()
        self.num_classes = m
        self.stn3 = STNKd(k=3)
        self.stn64 = STNKd(k=64)
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 64, 1))

        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, 1024, 1))

        self.mlp3 = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
                                  nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Conv1d(256, 128, 1))

        self.mlp4 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, m, 1), nn.BatchNorm1d(m), nn.Softmax())

    # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        pass

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """

        x = pointcloud
        stn3_input = torch.transpose(x, 1, 2)
        stn3_output = self.stn3(stn3_input)
        x = x@stn3_output
        x = torch.transpose(x,1,2)
        x = self.mlp1(x)
        x = torch.transpose(x,1,2)
        stn64_input = torch.transpose(x,1,2)
        stn64_output = self.stn64(stn64_input)
        n64_feature = x@stn64_output
        x = torch.transpose(n64_feature,1,2)
        x = self.mlp2(x)
        max_pool = nn.MaxPool1d(kernel_size=x.shape[2])
        x = max_pool(x)
        global_feature = torch.transpose(x, 1, 2)
        global_feature = global_feature.repeat(1, n64_feature.shape[1], 1)
        concat_feature = torch.cat((n64_feature, global_feature), dim=-1) # [B, n, 1088]
        x = torch.transpose(concat_feature, 1, 2)
        point_feature = self.mlp3(x)
        output = self.mlp4(point_feature)
        # output = torch.transpose(output, 1, 2) # [B, m, N]

        # x = pointcloud
        # stn3_input = torch.transpose(x, 1, 2)
        # stn3_output = self.stn3(stn3_input)
        # x = x@stn3_output
        # x = self.mlp1(x)
        # stn64_input = torch.transpose(x, 1, 2)
        # stn64_output = self.stn64(stn64_input)
        # n64_feature = x@stn64_output # [B, n, 64]
        # x = self.mlp2(n64_feature)
        # x = torch.transpose(x, 1, 2)
        # max_pool = nn.MaxPool1d(kernel_size=x.shape[2])
        # x = max_pool(x)
        # global_feature = torch.transpose(x, 1, 2) # [B, 1, 1024]
        # global_feature = global_feature.repeat(1, n64_feature.shape[1], 1)
        # concat_feature = torch.cat((n64_feature, global_feature), dim=-1) # [B, n, 1088]
        # point_feature = self.mlp3(concat_feature)
        # output = self.mlp4(point_feature)
        # output = torch.transpose(output, 1, 2) # [B, m, N]

        # TODO: Implement forward function.
        return output, stn3_output, stn64_output


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        self.pointnet_feat = PointNetFeat(True, True)

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.decode = nn.Sequential(nn.Linear(1024, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 3*self.num_points))
    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """

        global_feature, stn3_matrix, stn64_matrix = self.pointnet_feat(pointcloud)
        output = self.decode(global_feature)
        output = output.reshape(-1, self.num_points, 3)

        # TODO : Implement forward function.
        return output, stn3_matrix, stn64_matrix


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
