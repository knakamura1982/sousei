import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# 線形二クラス識別器の識別関数（一次式）の表示に使用
def print_with_sign(x, end='', file=sys.stdout):
    if x < 0:
        print(' - {0:.3f}'.format(-x), end=end, file=file)
    else:
        print(' + {0:.3f}'.format(x), end=end, file=file)


# 一層パーセプトロンによる二クラス識別器
# BCSLP: Binary Classifier by Single Layer Perceptron
class BCSLP(nn.Module):

    # コンストラクタ
    #   - n_dims: 特徴量の次元数（0以下の時は自動決定）
    def __init__(self, n_dims):
        super(BCSLP, self).__init__()
        self.nd = n_dims if n_dims > 0 else None
        self.fc = nn.Linear(self.nd, 1)

    # 順伝播（主としてloss計算時に使用）
    #   - x: 入力特徴量（ミニバッチ）
    def forward(self, x):
        y1 = self.fc(x)
        y0 = -y1
        return torch.cat([y0, y1], dim=1)

    # 順伝播
    #   - x: 入力特徴量（ミニバッチ）
    def classify(self, x):
        return F.softmax(self.forward(x), dim=1)

    # 識別関数（一次式）の表示
    def print_discriminant_func(self):
        w = self.fc.weight[0].to('cpu')
        b = self.fc.bias[0].to('cpu')
        print('  boundary: y = {0:.3f} * x[0]'.format(w[0]), end='', file=sys.stderr)
        for i in range(1, self.nd):
            print_with_sign(w[i], file=sys.stderr)
            print(' * x[{0}]'.format(i), end='', file=sys.stderr)
        print_with_sign(b, file=sys.stderr)
        print('', file=sys.stderr)


# 多層パーセプトロンによる二クラス識別器（中間層の数は2つで固定）
# BCMLP: Binary Classifier by Multi Layer Perceptron
class BCMLP(nn.Module):

    # コンストラクタ
    #   - n_dims: 特徴量の次元数（0以下の時は自動決定）
    #   - n_units: 中間量のユニット数（2要素の配列，リスト，タプル等で指定）
    def __init__(self, n_dims, n_units):
        super(BCMLP, self).__init__()
        self.nd = n_dims if n_dims > 0 else None
        self.fc1 = nn.Linear(self.nd, n_units[0])
        self.fc2 = nn.Linear(n_units[0], n_units[1])
        self.fc3 = nn.Linear(n_units[1], 1)

    # 順伝播（主としてloss計算時に使用）
    #   - x: 入力特徴量（ミニバッチ）
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y1 = self.fc3(h)
        y0 = -y1
        return torch.cat([y0, y1], dim=1)

    # 順伝播
    #   - x: 入力特徴量（ミニバッチ）
    def classify(self, x):
        return F.softmax(self.forward(x), dim=1)
