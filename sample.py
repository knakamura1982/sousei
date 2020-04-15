import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from binary_classifiers import BCSLP, BCMLP
from visualizers import BCVisualizer
from data_io import read_features


# データフォルダ
DATA_DIR = './dataset/Example/'

# ダミーデータを対象とするか否か
USING_DMY = True

# 多層パーセプトロンを使用するか否か（使用しない場合は一層パーセプトロンを使用）
USING_MLP = True


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='Example of Perceptron')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epochs', '-e', default=50, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
    parser.add_argument('--v_interval', '-v', default=5, type=int, help='visualization interval')
    args = parser.parse_args()

    # デバイスの設定
    dev_str = 'cuda:{0}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    dev = torch.device(dev_str)

    # オプション情報の設定・表示
    epochs = max(1, args.epochs) # 総エポック数（繰り返し回数）
    batchsize = max(1, args.batchsize) # バッチサイズ
    visualization_interval = max(1, args.v_interval) # 何エポックごとに可視化結果を表示するか
    print('device: {0}'.format(dev_str), file=sys.stderr)
    print('epochs: {0}'.format(epochs), file=sys.stderr)
    print('batchsize: {0}'.format(batchsize), file=sys.stderr)
    print('visualization interval: {0}'.format(visualization_interval), file=sys.stderr)
    print('', file=sys.stderr)

    # データの読み込み
    if USING_DMY == False:
        # 男女別身長／体重データを対象とする場合
        labels, features, labeldict, labelnames = read_features(DATA_DIR + 'whg_train.csv') # 学習データ読み込み
        labels_ev, features_ev, labeldict, dmy = read_features(DATA_DIR + 'whg_test.csv', dic=labeldict) # 評価用データ読み込み
    else:
        # ダミーデータを対象とする場合
        labels, features, labeldict, labelnames = read_features(DATA_DIR + 'dmy_train.csv') # 学習データ読み込み
        labels_ev, features_ev, labeldict, dmy = read_features(DATA_DIR + 'dmy_test.csv', dic=labeldict) # 評価用データ読み込み
    n_samples = features.shape[0] # 学習データの総数
    n_samples_ev = features_ev.shape[0] # 評価用データの総数
    n_dims = features.shape[1] # データの次元数
    n_classes = len(labeldict) # クラス数

    # 二クラス識別器の作成
    if USING_MLP == False:
        model = BCSLP(n_dims) # 一層パーセプトロンによる識別器
    else:
        model = BCMLP(n_dims, n_units=(10, 10)) # 多層パーセプトロンによる識別器
    model = model.to(dev)

    # 可視化の準備
    NUM = 100 # 可視化結果に重畳する実データの数
    a = np.min(features_ev, axis=0)
    b = np.max(features_ev, axis=0)
    perm = np.random.permutation(n_samples_ev)
    hrange = ((11 * a[0] - b[0]) / 10, (11 * b[0] - a[0]) / 10) # 可視化結果における横軸の範囲
    vrange = ((11 * a[1] - b[1]) / 10, (11 * b[1] - a[1]) / 10) # 可視化結果における縦軸の範囲
    data = [labels_ev[perm[ : NUM]], features_ev[perm[ : NUM]]] # 可視化結果に重畳する実データ（評価用データからランダムに NUM 個を選択）
    if USING_DMY == False:
        visualizer_hlabel = 'height (cm)' # 可視化結果における横軸のラベル
        visualizer_vlabel = 'weight (kg)' # 可視化結果における縦軸のラベル
    else:
        visualizer_hlabel = 'data 1'
        visualizer_vlabel = 'data 2'
    visualizer = BCVisualizer(hrange=hrange, vrange=vrange, hlabel=visualizer_hlabel, vlabel=visualizer_vlabel, clabels=labelnames)

    # 初期状態の可視化
    if USING_MLP == False:
        model.print_discriminant_func() # 一層パーセプトロンの場合，識別関数は一次式になるので，それを表示
        print('', file=sys.stderr)
    visualizer.show(model, device=dev, samples=data, title='Initial State') # グラフを表示

    # 損失関数の定義
    loss_func = nn.CrossEntropyLoss() # softmax cross entropy

    # オプティマイザーの用意
    optimizer = optim.Adam(model.parameters())

    # 学習処理ループ
    for e in range(epochs):

        # 現在のエポック番号を表示
        print('Epoch {0}'.format(e + 1), file=sys.stderr)

        # 損失関数の値が小さくなるように識別器のパラメータを更新
        model.train()
        sum_loss = 0
        perm = np.random.permutation(n_samples)
        for i in range(0, n_samples, batchsize):
            model.zero_grad()
            x = torch.tensor(features[perm[i : i + batchsize]], device=dev)
            t = torch.tensor(labels[perm[i : i + batchsize]], device=dev, dtype=torch.long)
            loss = loss_func(model(x), t)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss) * len(x)
            del loss
            del t
            del x

        # 損失関数の現在値を表示
        print('  train loss = {0:.4f}'.format(sum_loss / n_samples), file=sys.stderr)

        # 評価用データに対する識別精度を計算・表示
        model.eval()
        n_failed = 0
        for i in range(0, n_samples_ev, batchsize):
            x = torch.tensor(features_ev[i : i + batchsize], device=dev)
            t = torch.tensor(labels_ev[i : i + batchsize], device=dev, dtype=torch.long)
            y = model.classify(x)
            y = y.to('cpu').detach().numpy().copy()
            t = t.to('cpu').detach().numpy().copy()
            n_failed += np.count_nonzero(np.argmax(y, axis=1) - t)
            del y
            del x
            del t
        acc = (n_samples_ev - n_failed) / n_samples_ev
        print('  accuracy = {0:.2f}%'.format(100 * acc), file=sys.stderr)

        # 現状態の可視化
        if (e + 1) % visualization_interval == 0:
            if USING_MLP == False:
                model.print_discriminant_func() # 一層パーセプトロンの場合，識別関数は一次式になるので，それを表示
            visualizer.show(model, device=dev, samples=data, title='Epoch {0}'.format(e + 1)) # グラフを表示

        print('', file=sys.stderr)

    print('', file=sys.stderr)
