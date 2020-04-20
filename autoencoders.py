# coding: UTF-8

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from unit_layers import Conv, Pool, UpConv, DownConv, FC, ResBlock


# オートエンコーダのサンプルコード
class myAutoEncoder(nn.Module):

    L1_CHANNELS = 8
    L2_CHANNELS = 16

    # コンストラクタ
    #   - img_width: 入力画像の横幅
    #   - img_height: 入力画像の縦幅
    #   - img_channels: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    #   - u_units: 特徴ベクトルの次元数
    def __init__(self, img_width, img_height, img_channels, n_units):
        super(myAutoEncoder, self).__init__()
        # 層の定義
        self.conv1 = Conv(in_channels=img_channels, out_channels=self.L1_CHANNELS, kernel_size=5, activation=F.leaky_relu)
        self.pool1 = Pool(method='max')
        self.conv2 = Conv(in_channels=self.L1_CHANNELS, out_channels=self.L2_CHANNELS, kernel_size=3, activation=F.leaky_relu)
        self.pool2 = Pool(method='avg')
        self.w = img_width // 4 # プーリング層を 2 回経由するので，特徴マップの横幅は 1/2^2 = 1/4 となる
        self.h = img_height // 4 # 縦幅についても同様
        self.nz = self.w * self.h * self.L2_CHANNELS # 全結合層の直前におけるユニットの総数
        self.fc3 = FC(in_units=self.nz, out_units=n_units, activation=torch.tanh)
        self.fc4 = FC(in_units=n_units, out_units=self.nz, activation=F.leaky_relu)
        self.conv5 = UpConv(in_channels=self.L2_CHANNELS, out_channels=self.L2_CHANNELS, activation=F.leaky_relu)
        self.conv6 = UpConv(in_channels=self.L2_CHANNELS, out_channels=self.L1_CHANNELS, activation=F.leaky_relu)
        self.conv7 = Conv(in_channels=self.L1_CHANNELS, out_channels=img_channels, kernel_size=5, activation=torch.sigmoid)

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x):
        y = self.encode(x)
        x = self.decode(y)
        return x

    # 順伝播（エンコーダ部分のみ）
    def encode(self, x):
        # コンストラクタで定義した層を順に適用していく
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.pool2(h)
        h = h.view(h.size()[0], -1) # 平坦化：特徴マップをベクトルの形に並べ直す
        y = self.fc3(h)
        return y

    # 順伝播（デコーダ部分のみ）
    def decode(self, y):
        # コンストラクタで定義した層を順に適用していく
        h = self.fc4(y)
        h = h.reshape(h.size()[0], self.L2_CHANNELS, self.h, self.w) # ベクトルを特徴マップの形に並べ直す
        h = self.conv5(h)
        h = self.conv6(h)
        x = self.conv7(h)
        return x


# カラー化オートエンコーダのサンプルコード
# 入力画像のチャンネル数は 1，出力画像のチャンネル数は 3 で固定とする
class myColorizationAE(nn.Module):

    L1_CHANNELS = 4
    L2_CHANNELS = 8
    L3_CHANNELS = 16
    L4_CHANNELS = 16

    # コンストラクタ
    #   - img_width: 入力画像の横幅
    #   - img_height: 入力画像の縦幅
    def __init__(self, img_width, img_height):
        super(myColorizationAE, self).__init__()
        # 層の定義
        self.conv1 = Conv(in_channels=1, out_channels=self.L1_CHANNELS, kernel_size=3, activation=F.leaky_relu)
        self.conv2 = DownConv(in_channels=self.L1_CHANNELS, out_channels=self.L2_CHANNELS, activation=F.leaky_relu)
        self.conv3 = DownConv(in_channels=self.L2_CHANNELS, out_channels=self.L3_CHANNELS, activation=F.leaky_relu)
        self.conv4 = DownConv(in_channels=self.L3_CHANNELS, out_channels=self.L4_CHANNELS, activation=F.leaky_relu)
        self.conv5 = UpConv(in_channels=self.L4_CHANNELS, out_channels=self.L3_CHANNELS, activation=F.leaky_relu)
        self.conv6 = UpConv(in_channels=self.L3_CHANNELS, out_channels=self.L2_CHANNELS, activation=F.leaky_relu)
        self.conv7 = UpConv(in_channels=self.L2_CHANNELS, out_channels=self.L1_CHANNELS, activation=F.leaky_relu)
        self.conv8 = Conv(in_channels=self.L1_CHANNELS, out_channels=3, kernel_size=3, activation=torch.sigmoid)

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x):
        # コンストラクタで定義した層を順に適用していく
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv7(h)
        y = self.conv8(h)
        return y


# アップサンプリング用オートエンコーダのサンプルコード
# 拡大率は縦横ともに 2 倍で固定
class myUpSamplingAE(nn.Module):

    L1_CHANNELS = 4
    L2_CHANNELS = 4
    L3_CHANNELS = 4
    L4_CHANNELS = 4

    # コンストラクタ
    #   - img_width: 入力画像の横幅
    #   - img_height: 入力画像の縦幅
    #   - img_channels: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    def __init__(self, img_width, img_height, img_channels):
        super(myUpSamplingAE, self).__init__()
        # 層の定義
        self.conv1 = Conv(in_channels=img_channels, out_channels=self.L1_CHANNELS, kernel_size=7, activation=F.relu)
        self.res2 = ResBlock(in_channels=self.L1_CHANNELS, out_channels=self.L2_CHANNELS, activation=F.relu)
        self.conv3 = UpConv(in_channels=self.L2_CHANNELS, out_channels=self.L3_CHANNELS, activation=F.relu)
        self.res4 = ResBlock(in_channels=self.L3_CHANNELS, out_channels=self.L4_CHANNELS, activation=F.relu)
        self.conv5 = Conv(in_channels=self.L4_CHANNELS, out_channels=img_channels, kernel_size=3, activation=torch.sigmoid)

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x):
        # コンストラクタで定義した層を順に適用していく
        h = self.conv1(x)
        h = self.res2(h)
        h = self.conv3(h)
        h = self.res4(h)
        y = self.conv5(h)
        return y
