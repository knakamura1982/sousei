import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cnn import myCNN
from data_io import read_image_list, load_single_image, load_images


# データセットの指定
DATA_DIR = './dataset/MNIST/' # データフォルダのパス
IMAGE_LIST = DATA_DIR + 'train_list.csv' # 学習データ
IMAGE_LIST_EV = DATA_DIR + 'test_list.csv' # 評価用データ

# モデルを保存するフォルダの指定
MODEL_DIR = './mnist_models/'


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for MNIST Image Recognition (Training)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
    parser.add_argument('--model', '-m', default='mnist_model.pth', type=str, help='file path of trained model')
    args = parser.parse_args()

    # デバイスの設定
    dev_str = 'cuda:{0}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    dev = torch.device(dev_str)

    # オプション情報の設定・表示
    epochs = max(1, args.epochs) # 総エポック数（繰り返し回数）
    batchsize = max(1, args.batchsize) # バッチサイズ
    model_filepath = args.model # 学習結果のモデルの保存先ファイル
    print('device: {0}'.format(dev_str), file=sys.stderr)
    print('epochs: {0}'.format(epochs), file=sys.stderr)
    print('batchsize: {0}'.format(batchsize), file=sys.stderr)
    print('model file: {0}'.format(model_filepath), file=sys.stderr)
    print('', file=sys.stderr)

    # 画像の縦幅・横幅・チャンネル数の設定
    width = 28 # MNIST文字画像の場合，横幅は 28 pixels
    height = 28 # MNIST文字画像の場合，縦幅も 28 pixels
    channels = 1 # MNIST文字画像はグレースケール画像なので，チャンネル数は 1
    color_mode = 0 if channels == 1 else 1

    # 学習データの読み込み
    labels, imgfiles, labeldict, labelnames = read_image_list(IMAGE_LIST, DATA_DIR)
    n_samples = len(imgfiles) # 学習データの総数
    n_classes = len(labeldict) # クラス数
    with open(MODEL_DIR + 'labeldict.pickle', 'wb') as f:
        pickle.dump(labeldict, f) # ラベル名からラベル番号を与える辞書をファイルに保存
    with open(MODEL_DIR + 'labelnames.pickle', 'wb') as f:
        pickle.dump(labelnames, f) # ラベル番号からラベル名を与える配列をファイルに保存

    # 評価用データの読み込み
    labels_ev, imgfiles_ev, labeldict, dmy = read_image_list(IMAGE_LIST_EV, DATA_DIR, dic=labeldict)
    n_samples_ev = len(imgfiles_ev) # 評価用データの総数

    # 画像認識器の作成
    model = myCNN(width, height, channels, n_classes)
    model = model.to(dev)

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
        n_input = 0
        sum_loss = 0
        perm = np.random.permutation(n_samples)
        for i in range(0, n_samples, batchsize * 10): # 高速化のため，ミニバッチ10個につき1個しか学習に用いないことにする
            model.zero_grad()
            x = torch.tensor(load_images(imgfiles, ids=perm[i : i + batchsize], mode=color_mode), device=dev) # 学習データを読み込む
            t = torch.tensor(labels[perm[i : i + batchsize]], device=dev, dtype=torch.long)
            loss = loss_func(model(x), t)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss) * len(x)
            n_input += len(x)
            del loss
            del t
            del x

        # 損失関数の現在値を表示
        print('  train loss = {0:.4f}'.format(sum_loss / n_input), file=sys.stderr)

        # 評価用データに対する識別精度を計算・表示
        model.eval()
        n_failed = 0
        for i in range(0, n_samples_ev, batchsize):
            x = torch.tensor(load_images(imgfiles_ev, ids=np.arange(i, i + batchsize), mode=color_mode), device=dev) # 評価用データを読み込む
            t = torch.tensor(labels_ev[i : i + batchsize], device=dev, dtype=torch.long)
            y = model.classify(x)
            y_cpu = y.to('cpu').detach().numpy().copy()
            t_cpu = t.to('cpu').detach().numpy().copy()
            n_failed += np.count_nonzero(np.argmax(y_cpu, axis=1) - t_cpu)
            del y_cpu
            del t_cpu
            del y
            del x
            del t
        acc = (n_samples_ev - n_failed) / n_samples_ev
        print('  accuracy = {0:.2f}%'.format(100 * acc), file=sys.stderr)

        # 現在のモデルをファイルに自動保存
        torch.save(model.to('cpu').state_dict(), MODEL_DIR + 'model_ep{0}.pth'.format(e + 1))
        model = model.to(dev)

        print('', file=sys.stderr)

    # 最終結果のモデルをファイルに保存
    torch.save(model.to('cpu').state_dict(), model_filepath)

    print('', file=sys.stderr)
