# coding: UTF-8

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from unit_layers import Conv, DownConv, UpConv, Pool, FC, GlobalPool, Flatten, Reshape
from unit_layers import ResBlockP as ResBlock


# 多層パーセプトロンのサンプルコード
# 認識対象のクラスの数は 2 で固定とする
class myMLP(nn.Module):

    # コンストラクタ
    #   - Z: 入力特徴量の次元数
    #   - N: 
    def __init__(self, Z):
        super(myMLP, self).__init__()

        # ネットワーク構造
        self.layers = nn.Sequential(

            ### この部分を書き換えることにより，ネットワーク構造を変更できる ###

            # 全結合層（活性化関数:なし）
            # 入力ユニット数は入力特徴量の次元数に一致させる必要があるので，in_units = Z
            # 出力ユニット数は認識対象のクラス数に一致させる必要があるので，out_units = 2
            # なお，例えば活性化関数として ReLU を使いたい場合は activation=F.relu などと記載すれば良い
            FC(in_units=Z, out_units=2, activation=None) # 通常は各行の末尾に「,」をつけるが，最後の層だけはつけない

            ### ここまで ###

        )

    # 順伝播
    #   - x: 入力特徴量（ミニバッチ）
    def forward(self, x):
        h = self.layers(x)
        return F.softmax(h, dim=1) - 0.5 # 可視化の都合上，少し特殊な処理を加える


# 画像認識用ニューラルネットワークのサンプルコード
class myCNN(nn.Module):

    # コンストラクタ
    #   - W: 入力画像の横幅
    #   - H: 入力画像の縦幅
    #   - S: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    #   - N: 認識対象のクラスの数
    def __init__(self, W, H, S, N):
        super(myCNN, self).__init__()

        # ネットワーク構造
        self.layers = nn.Sequential(

            ### この部分を書き換えることにより，ネットワーク構造を変更できる ###

            # ユニット配列の縦幅・横幅が変わらないように調整した畳込み層（出力チャンネル数:2，カーネルサイズ:3，活性化関数:ReLU）
            # この層を通すことにより，ユニットの並びは W×H×S → W×H×2 と変わる
            Conv(in_channels=S, out_channels=2, kernel_size=3, activation=F.relu),

            # max-pooling
            # ユニットの並びは，縦横のサイズが半分に，チャンネル数はそのまま．つまり，W×H×2 → (W/2)×(H/2)×2
            Pool(method='max'),

            # 同様の畳込み層をもう一度（出力チャンネル数:4，カーネルサイズ:5，活性化関数:シグモイド）
            # 入力チャンネル数を正しく指定しないといけない．今回の場合，この層を通す前のユニットの並びが (W/2)×(H/2)×2 なので in_channels=2
            # この層を通すことにより，ユニットの並びは (W/2)×(H/2)×2 → (W/2)×(H/2)×4 に．
            Conv(in_channels=2, out_channels=4, kernel_size=5, activation=torch.sigmoid),

            # average-pooling
            # ユニットの並びは (W/2)×(H/2)×4 → (W/4)×(H/4)×4 に．
            Pool(method='avg'),

            # 平坦化
            # ユニットは一列に並べ直される．その個数は WH/4 個
            Flatten(),

            # 全結合層（活性化関数:なし）
            # 入力ユニット数を正しく指定しないといけない．今回の場合，この層を通す前のユニット数が WH/4 なので in_units = W*H / 4
            # なお，python では整数型の割り算は「/」でなく「//」であることに注意
            FC(in_units=W*H//4, out_units=N, activation=None) # 最後だけ「,」をつけない

            ### ここまで ###

        )

    # 順伝播
    #   - x: 入力特徴量（ミニバッチ）
    def forward(self, x):
        return self.layers(x)


# AlexNet, VGG, ResNet などの有名モデルをバックボーンとして活用したニューラルネットワークのサンプルコード
# この方法は転移学習（transfer learning）と呼ばれる
# なお，入力画像のチャンネル数は 3 で固定（カラー画像のみ）とする
class myCNN2(nn.Module):

    # コンストラクタ
    #   - W: 入力画像の横幅
    #   - H: 入力画像の縦幅
    #   - N: 認識対象のクラスの数
    def __init__(self, W, H, N):
        super(myCNN2, self).__init__()


        ### バックボーンとして使用する有名モデルを準備 ###

        # モデルをロード（どれか一つのみ残す）
        # なお，PyTorch で標準使用可能な有名モデルは他にもあるが，下と同様のコードで動作するかは分からない
        # 参考： https://pytorch.org/vision/stable/models.html
        Backbone = models.alexnet(pretrained=True)
        #Backbone = models.vgg16(pretrained=True)
        #Backbone = models.resnet18(pretrained=True)
        #Backbone = models.googlenet(pretrained=True)

        # ロードしたモデルから分類層（最終層）を削除
        # AlexNet, VGG16 の場合は以下の 2 行を使う
        Z = Backbone.classifier[-1].in_features # バックボーンモデルの出力層のユニット数
        del Backbone.classifier[-1]
        '''
        # ResNet18, GoogleNet の場合は以下の 4 行を使う
        backbone_layers = list(Backbone.children())
        Z = backbone_layers[-1].in_features # バックボーンモデルの出力層のユニット数
        del backbone_layers[-1]
        Backbone = nn.Sequential(*backbone_layers)
        '''

        # ロードしたモデルのパラメータを固定（ここは書き換えなくて良い）
        for param in Backbone.parameters():
            param.requires_grad = False

        ### ここまで ###


        # ネットワーク構造
        self.layers = nn.Sequential(

            ### この部分を書き換えることにより，ネットワーク構造を変更できる ###

            # データ拡張（画像に左右反転や回転などの処理をランダムに加えデータ量を擬似的に増やす技法）を試す
            # 一般に，拡張すればするほど必要エポック数は多くなる
            # 参考： https://pystyle.info/pytorch-list-of-transforms/
            transforms.RandomHorizontalFlip(), # 左右反転
            transforms.RandomRotation(degrees=180), # 180度回転

            # バックボーンモデルに画像を入力するための準備
            # 画像の縦幅・横幅をそれぞれ 224 にし，画素値を予め定められた値で正規化する
            # この部分はどの有名モデルをバックボーンにする場合でも基本このままで良い
            transforms.Resize(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

            # バックボーンモデル
            Backbone,

            # 平坦化
            # これにより Z 個のユニットが直列に並んだ形になる
            Flatten(),

            # 全結合層（活性化関数:なし）
            # 入力ユニット数を正しく指定しないといけない．今回の場合，バックボーンモデルの出力が Z 個のユニットなので in_units = Z
            FC(in_units=Z, out_units=N, activation=None) # 最後だけ「,」をつけない

            ### ここまで ###

        )

    # 順伝播
    #   - x: 入力特徴量（ミニバッチ）
    def forward(self, x):
        return self.layers(x)
