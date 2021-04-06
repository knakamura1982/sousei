# coding: UTF-8

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from unit_layers import Conv, DownConv, UpConv, Pool, FC, GlobalPool, Flatten, Reshape
from unit_layers import ResBlockP as ResBlock


# オートエンコーダのサンプルコード
class myAutoEncoder(nn.Module):

    # コンストラクタ
    #   - W: 入力画像の横幅
    #   - H: 入力画像の縦幅
    #   - S: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    #   - N: 特徴ベクトルの次元数
    def __init__(self, W, H, S, N):
        super(myAutoEncoder, self).__init__()

        # エンコーダ部分のネットワーク構造
        self.encoder_layers = nn.Sequential(

            ### この部分を書き換えることにより，エンコーダ部分の構造を変更 ###

            # ユニット配列の縦幅・横幅が半分になるように調整した畳込み層（出力チャンネル数:3，カーネルサイズ:6，活性化関数:Leaky-ReLU）
            # この層を通すことにより，ユニットの並びは W×H×S → (W/2)×(H/2)×3 と変わる
            DownConv(in_channels=S, out_channels=3, kernel_size=6, activation=F.leaky_relu),

            # 同様の畳込み層をもう一度（出力チャンネル数:2，カーネルサイズ:4，活性化関数:tanh）
            # 入力チャンネル数を正しく指定しないといけない．今回の場合，この層を通す前のユニットの並びが (W/2)×(H/2)×3 なので in_channels=3
            # この層を通すことにより，ユニットの並びは (W/2)×(H/2)×3 → (W/4)×(H/4)×2 と変わる
            DownConv(in_channels=3, out_channels=2, kernel_size=4, activation=torch.tanh),

            # 平坦化
            # ユニットは一列に並べ直される．その個数は WH/8 個
            Flatten(),

            # 全結合層（活性化関数:なし）
            # 入力ユニット数を正しく指定しないといけない．今回の場合，この層を通す前のユニット数が WH/8 なので in_units = W*H//8
            FC(in_units=W*H//8, out_units=N, activation=None) # 最後だけ「,」をつけない

            ### ここまで ###

        )

        # デコーダ部分のネットワーク構造
        self.decoder_layers = nn.Sequential(

            ### この部分を書き換えることにより，デコーダ部分の構造を変更 ###

            # 全結合層（活性化関数:ELU）
            # 出力ユニット数はとりあえず WH/8 個としておく
            FC(in_units=N, out_units=W*H//8, activation=F.elu),

            # 一列に並んでいる WH/8 個のユニットを，三次元的な並びになるよう再配置する
            # (W/4)×(H/4)×2 の並びにしたい → チャンネル数，縦幅，横幅の順で size=(2, H//4, W//4) と指定
            Reshape(size=(2, H//4, W//4)),

            # ユニット配列の縦幅・横幅が 2 倍になるように調整した畳込み層（出力チャンネル数:3，カーネルサイズ:6，活性化関数:ReLU）
            # 入力チャンネル数を正しく指定しないといけない．今回の場合，この層を通す前のユニットの並びが (W/4)×(H/4)×2 なので in_channels=2
            # この層を通すことにより，ユニットの並びは (W/4)×(H/4)×2 → (W/2)×(H/2)×3 と変わる
            UpConv(in_channels=2, out_channels=3, kernel_size=6, activation=F.relu),

            # 同様の畳込み層をもう一度（出力チャンネル数:4，カーネルサイズ:4，活性化関数:ReLU）
            # この層を通す前のユニットの並びが (W/2)×(H/2)×3 なので in_channels=3
            # この層を通すことにより，ユニットの並びは (W/2)×(H/2)×3 → W×H×4 と変わる
            UpConv(in_channels=3, out_channels=4, kernel_size=4, activation=F.relu),

            # 最後に，ユニット配列の縦幅・横幅が変わらないように調整した畳込み層（出力チャンネル数:S，カーネルサイズ:5，活性化関数:シグモイド）を入れる
            # この層を通すことにより，ユニットの並びは W×H×4 → W×H×S と変わり，入力画像と同じサイズに戻る
            Conv(in_channels=4, out_channels=S, kernel_size=5, activation=torch.sigmoid) # 最後だけ「,」をつけない

            ### ここまで ###

        )

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x):
        y = self.encode(x)
        x = self.decode(y)
        return x

    # 順伝播（エンコーダ部分のみ）
    def encode(self, x):
        return self.encoder_layers(x)

    # 順伝播（デコーダ部分のみ）
    def decode(self, y):
        return self.decoder_layers(y)


# カラー化オートエンコーダのサンプルコード
# 入力画像のチャンネル数は 1，出力画像のチャンネル数は 3 で固定とする
class myColorizationAE(nn.Module):

    # コンストラクタ
    #   - W: 入力画像の横幅
    #   - H: 入力画像の縦幅
    def __init__(self, W, H):
        super(myColorizationAE, self).__init__()

        # ネットワーク構造
        self.layers = nn.Sequential(

            ### この部分を書き換えることにより，ネットワーク構造を変更 ###

            # ユニット配列の縦幅・横幅が変わらないように調整した畳込み層（出力チャンネル数:2，カーネルサイズ:3，活性化関数:ReLU）
            # 入力はグレースケール画像，すなわちチャンネル数 1 の画像を想定しているため，in_channels=1
            # ドロップアウト率 0.1 でドロップアウトを試してみる
            # この層を通すことにより，ユニットの並びは W×H×1 → W×H×2 と変わる
            Conv(in_channels=1, out_channels=2, kernel_size=3, dropout_ratio=0.1, activation=F.relu),

            # 同様の畳込み層をもう一度（出力チャンネル数:3，カーネルサイズ:3，活性化関数:sigmoid）
            # この層を通す前のユニットの並びが W×H×2 なので in_channels=2
            # この層を通すことにより，ユニットの並びは W×H×2 → W×H×3 と変わり，出力のカラー画像と同じサイズになる
            Conv(in_channels=2, out_channels=3, kernel_size=3, activation=torch.sigmoid) # 最後だけ「,」をつけない

            ### ここまで ###

        )

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x):
        return self.layers(x)


# ノイズ除去オートエンコーダのサンプルコード
class myDenoisingAE(nn.Module):

    # コンストラクタ
    #   - W: 入力画像の横幅
    #   - H: 入力画像の縦幅
    #   - S: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    def __init__(self, W, H, S):
        super(myDenoisingAE, self).__init__()

        # ネットワーク構造
        self.layers = nn.Sequential(

            ### この部分を書き換えることにより，ネットワーク構造を変更 ###

            # ResBlock層（出力チャンネル数:3，カーネルサイズ:3，活性化関数:tanh）
            # この層では画像の縦幅・横幅は変化しない
            # この層を通すことにより，ユニットの並びは W×H×S → W×H×3 と変わる
            ResBlock(in_channels=S, out_channels=3, kernel_size=3, activation=torch.tanh),

            # ユニット配列の縦幅・横幅が変わらないように調整した畳込み層（出力チャンネル数:S，カーネルサイズ:3，活性化関数:sigmoid）
            # この層を通す前のユニットの並びが W×H×3 なので in_channels=3
            # この層を通すことにより，ユニットの並びは W×H×3 → W×H×S と変わり，入力画像と同じサイズに戻る
            Conv(in_channels=3, out_channels=S, kernel_size=3, activation=torch.sigmoid) # 最後だけ「,」をつけない

            ### ここまで ###

        )

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x):
        return self.layers(x)


# アップサンプリング用オートエンコーダのサンプルコード
# 拡大率は縦横ともに 2 倍で固定とする
class myUpSamplingAE(nn.Module):

    # コンストラクタ
    #   - W: 入力画像の横幅
    #   - H: 入力画像の縦幅
    #   - S: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    def __init__(self, W, H, S):
        super(myUpSamplingAE, self).__init__()

        # ネットワーク構造
        self.layers = nn.Sequential(

            ### この部分を書き換えることにより，ネットワーク構造を変更 ###

            # ユニット配列の縦幅・横幅が 2 倍になるように調整した畳込み層（出力チャンネル数:2，カーネルサイズ:4，活性化関数:softplus）
            # この層を通すことにより，ユニットの並びは W×H×S → (2W)×(2H)×2 と変わる
            UpConv(in_channels=S, out_channels=2, kernel_size=4, activation=F.softplus),

            # ユニット配列の縦幅・横幅が変わらないように調整した畳込み層（出力チャンネル数:S，カーネルサイズ:3，活性化関数:sigmoid）
            # この層を通す前のユニットの並びが (2W)×(2H)×2 なので in_channels=2
            # この層を通すことにより，ユニットの並びは (2W)×(2H)×2 → (2W)×(2H)×S と変わり，入力画像と同じチャンネル数かつ縦幅・横幅は２倍になる
            Conv(in_channels=2, out_channels=S, kernel_size=3, activation=torch.sigmoid) # 最後だけ「,」をつけない

            ### ここまで ###

        )

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x):
        return self.layers(x)
