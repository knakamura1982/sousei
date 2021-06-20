import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from utils import save_progress
from data_io import load_single_image, load_images


# データセットを訓練データと検証用データに分割
#   - x: 入力データリスト（list）
#   - y: 出力データリスト（list）
#   - ratio: 全体の何割を検証用データにするか（float）
def split(x, y, ratio):

    n_total = len(x)
    n_valid = math.floor(n_total * ratio)
    n_train = n_total - n_valid

    perm = np.random.permutation(n_total)

    x_train = []
    x_valid = []
    y_train = []
    y_valid = []
    for i in range(0, n_valid):
        x_valid.append(x[perm[i]])
        y_valid.append(y[perm[i]])
    for i in range(n_valid, n_total):
        x_train.append(x[perm[i]])
        y_train.append(y[perm[i]])

    return x_train, x_valid, y_train, y_valid


# 特定条件でリストを numpy.ndarray に変換
def to_nparray(x):

    if type(x[0]) == int:
        x = np.asarray(x, dtype=np.int32)
    elif type(x[0]) == float:
        x = np.asarray(x, dtype=np.float32)
    elif type(x[0]) == np.ndarray:
        x = np.asarray(x, dtype=x[0].dtype)

    return x


# 学習
#   - device: 使用するデバイス
#   - model_dir: 一時ファイルを保存するフォルダ
#   - in_data: 入力データリスト（list）
#   - out_data: 出力データリスト（list）
#   - model: ニューラルネットワークモデル（GPUに移動させる前）
#   - loss_func: 損失関数（クラスインスタンス）
#   - batchsize: バッチサイズ
#   - epochs: エポック数
#   - ratio: 全体の何割を検証用データにするか（float）
#   - stepsize: >1 のとき，ミニバッチを stepsize 個につき一個しか使用しないようにする（int）
#   - additional_conditions: 追加条件（dict）
def train(device, model_dir, in_data, out_data, model, loss_func, batchsize, epochs, ratio=0.1, stepsize=1, additional_conditions={}):

    # データセットを分割
    x_train, x_valid, y_train, y_valid = split(in_data, out_data, ratio)

    # データ数を確認
    n_samples = len(x_train)
    n_samples_ev = len(x_valid)

    # 入力データタイプの確認
    if type(x_train[0]) == str:
        # 画像の場合は画像サイズ・チャンネル数の情報を取得しておく
        x_type_str = True
        x_channels, x_height, x_width = load_single_image(x_train[0]).shape
        x_size = None
        if 'input_scale' in additional_conditions:
            x_width = round(x_width * additional_conditions['input_scale'])
            x_height = round(x_height * additional_conditions['input_scale'])
            x_size = (x_width, x_height)
        if 'input_size' in additional_conditions:
            x_size = additional_conditions['input_size']
        if 'in_channels' in additional_conditions:
            x_channels = additional_conditions['in_channels']
        x_color_mode = 0 if x_channels == 1 else 1
    else:
        # 数値の場合は numpy.ndarray に変換しておく
        x_type_str = False
        x_train = to_nparray(x_train)
        x_valid = to_nparray(x_valid)

    # 出力データタイプも同様にして確認
    if type(y_train[0]) == str:
        y_type_str = True
        y_channels, y_height, y_width = load_single_image(y_train[0]).shape
        y_size = None
        if 'output_scale' in additional_conditions:
            y_width = round(y_width * additional_conditions['output_scale'])
            y_height = round(y_height * additional_conditions['output_scale'])
            y_size = (y_width, y_height)
        if 'output_size' in additional_conditions:
            y_size = additional_conditions['output_size']
        if 'out_channels' in additional_conditions:
            y_channels = additional_conditions['out_channels']
        y_color_mode = 0 if y_channels == 1 else 1
    else:
        y_type_str = False
        y_train = to_nparray(y_train)
        y_valid = to_nparray(y_valid)

    # 追加情報の確認
    model_dir = './'
    autosave = False
    x_with_noise = False
    y_with_noise = False
    x_noise_points = 0
    y_noise_points = 0
    if 'input_with_noise' in additional_conditions:
        x_with_noise = True
        x_noise_points = additional_conditions['input_with_noise']
    if 'output_with_noise' in additional_conditions:
        y_with_noise = True
        y_noise_points = additional_conditions['output_with_noise']
    if 'autosave_model' in additional_conditions:
        autosave = True
        if 'model_dir' in additional_conditions:
            model_dir = additional_conditions['model_dir']

    # モデルをGPU上に移動
    model = model.to(device)

    # オプティマイザーの用意
    optimizer = optim.Adam(model.parameters())

    # loss確認用グラフの用意
    plt.title('Loss history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    train_loss = []
    valid_loss = []

    # 学習処理ループ
    for e in range(epochs):

        # 現在のエポック番号を表示
        print('Epoch {0}'.format(e + 1), file=sys.stderr)

        # 損失関数の値が小さくなるようにモデルパラメータを更新
        model.train()
        n_input = 0
        sum_loss = 0
        perm = np.random.permutation(n_samples)
        for i in range(0, n_samples, batchsize * stepsize):
            model.zero_grad()
            # ミニバッチ（入力側）を作成
            if x_type_str:
                x = torch.tensor(load_images(x_train, ids=perm[i : i + batchsize], size=x_size, with_noise=x_with_noise, n_noise_points=x_noise_points, mode=x_color_mode), device=device)
            elif x_train.dtype == np.int32:
                x = torch.tensor(x_train[perm[i : i + batchsize]], device=device, dtype=torch.long)
            else:
                x = torch.tensor(x_train[perm[i : i + batchsize]], device=device)
            # ミニバッチ（出力側）を作成
            if y_type_str:
                y = torch.tensor(load_images(y_train, ids=perm[i : i + batchsize], size=y_size, with_noise=y_with_noise, n_noise_points=y_noise_points, mode=y_color_mode), device=device)
            elif y_train.dtype == np.int32:
                y = torch.tensor(y_train[perm[i : i + batchsize]], device=device, dtype=torch.long)
            else:
                y = torch.tensor(y_train[perm[i : i + batchsize]], device=device)
            loss = loss_func(model(x), y)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss) * len(x)
            n_input += len(x)
            del loss
            del y
            del x

        # 損失関数の現在値を表示
        train_loss.append(sum_loss / n_input)
        print('  train loss = {0:.6f}'.format(sum_loss / n_input), file=sys.stderr)

        # 検証用データを用いた場合の損失を計算・表示
        model.eval()
        n_input = 0
        n_failed = 0
        sum_loss = 0
        perm = np.arange(0, n_samples_ev)
        for i in range(0, n_samples_ev, batchsize):
            # ミニバッチ（入力側）を作成
            if x_type_str:
                x = torch.tensor(load_images(x_valid, ids=perm[i : i + batchsize], size=x_size, with_noise=x_with_noise, n_noise_points=x_noise_points, mode=x_color_mode), device=device)
            elif x_valid.dtype == np.int32:
                x = torch.tensor(x_valid[i : i + batchsize], device=device, dtype=torch.long)
            else:
                x = torch.tensor(x_valid[i : i + batchsize], device=device)
            # ミニバッチ（出力側）を作成
            calc_accuracy = False
            if y_type_str:
                y = torch.tensor(load_images(y_valid, ids=perm[i : i + batchsize], size=y_size, with_noise=y_with_noise, n_noise_points=y_noise_points, mode=y_color_mode), device=device)
            elif y_valid.dtype == np.int32:
                y = torch.tensor(y_valid[i : i + batchsize], device=device, dtype=torch.long)
                calc_accuracy = True
            else:
                y = torch.tensor(y_valid[i : i + batchsize], device=device)
            z = model(x)
            loss = loss_func(z, y)
            sum_loss += float(loss) * len(x)
            n_input += len(x)
            if calc_accuracy:
                y_cpu = y.to('cpu').detach().numpy().copy()
                z_cpu = z.to('cpu').detach().numpy().copy()
                n_failed += np.count_nonzero(np.argmax(z_cpu, axis=1) - y_cpu)
                del y_cpu
                del z_cpu
            if y_type_str and i == 0:
                if e == 0:
                    x_cpu = x.to('cpu').detach().numpy().copy()
                    y_cpu = y.to('cpu').detach().numpy().copy()
                    save_progress(model_dir + 'input.png', x_cpu, n_data_max=25, n_data_per_row=5, mode=x_color_mode)
                    save_progress(model_dir + 'ground_truth.png', y_cpu, n_data_max=25, n_data_per_row=5, mode=y_color_mode)
                    del x_cpu
                    del y_cpu
                z_cpu = z.to('cpu').detach().numpy().copy()
                save_progress(model_dir + 'output_ep{0}.png'.format(e + 1), z_cpu, n_data_max=25, n_data_per_row=5, mode=y_color_mode)
                del z_cpu
            del loss
            del z
            del y
            del x
        valid_loss.append(sum_loss / n_input)
        print('  valid loss = {0:.6f}'.format(sum_loss / n_input), file=sys.stderr)
        if calc_accuracy:
            acc = (n_samples_ev - n_failed) / n_samples_ev
            print('  accuracy = {0:.2f}%'.format(100 * acc), file=sys.stderr)

        # 現在のモデルを保存する
        if autosave:
            torch.save(model.to('cpu').state_dict(), os.path.join(model_dir, 'model_ep{0}.pth'.format(e + 1)))
            model = model.to(device)

        print('', file=sys.stderr)

    # loss確認用グラフの表示
    plt.plot(np.arange(1, epochs + 1), np.asarray(train_loss), label='train loss')
    plt.plot(np.arange(1, epochs + 1), np.asarray(valid_loss), label='valid loss')
    plt.legend(loc='upper right')
    plt.pause(2)
    plt.close()


    return model.to('cpu')


# 推論
#   - device: 使用するデバイス
#   - in_data: 入力データリスト（list）
#   - out_data: 出力（正解）データリスト（list）
#   - model: 学習済みニューラルネットワークモデル（GPUに移動させる前）
#   - loss_func: 損失関数（クラスインスタンス）
#   - batchsize: バッチサイズ
#   - additional_conditions: 追加条件（dict）
def predict(device, in_data, out_data, model, loss_func, batchsize, additional_conditions={}):

    x_data = in_data
    y_data = out_data

    # データ数を確認
    n_samples_ev = len(x_data)

    # 入力データタイプの確認
    if type(x_data[0]) == str:
        # 画像の場合は画像サイズ・チャンネル数の情報を取得しておく
        x_type_str = True
        x_channels, x_height, x_width = load_single_image(x_data[0]).shape
        x_size = None
        if 'input_scale' in additional_conditions:
            x_width = round(x_width * additional_conditions['input_scale'])
            x_height = round(x_height * additional_conditions['input_scale'])
            x_size = (x_width, x_height)
        if 'input_size' in additional_conditions:
            x_size = additional_conditions['input_size']
        if 'in_channels' in additional_conditions:
            x_channels = additional_conditions['in_channels']
        x_color_mode = 0 if x_channels == 1 else 1
    else:
        # 数値の場合は numpy.ndarray に変換しておく
        x_type_str = False
        x_data = to_nparray(x_data)

    # 出力データタイプも同様にして確認
    if type(y_data[0]) == str:
        y_type_str = True
        y_channels, y_height, y_width = load_single_image(y_data[0]).shape
        y_size = None
        if 'output_scale' in additional_conditions:
            y_width = round(y_width * additional_conditions['output_scale'])
            y_height = round(y_height * additional_conditions['output_scale'])
            y_size = (y_width, y_height)
        if 'output_size' in additional_conditions:
            y_size = additional_conditions['output_size']
        if 'out_channels' in additional_conditions:
            y_channels = additional_conditions['out_channels']
        y_color_mode = 0 if y_channels == 1 else 1
    else:
        y_type_str = False
        y_data = to_nparray(y_data)

    # 追加情報の確認
    x_with_noise = False
    y_with_noise = False
    x_noise_points = 0
    y_noise_points = 0
    if 'input_with_noise' in additional_conditions:
        x_with_noise = True
        x_noise_points = additional_conditions['input_with_noise']
    if 'output_with_noise' in additional_conditions:
        y_with_noise = True
        y_noise_points = additional_conditions['output_with_noise']

    # モデルをGPU上に移動
    model = model.to(device)
    model.eval()

    # 評価用データを用いて損失を計算・表示
    n_input = 0
    n_failed = 0
    sum_loss = 0
    perm = np.arange(0, n_samples_ev)
    for i in range(0, n_samples_ev, batchsize):
        # ミニバッチ（入力側）を作成
        if x_type_str:
            x = torch.tensor(load_images(x_data, ids=perm[i : i + batchsize], size=x_size, with_noise=x_with_noise, mode=x_color_mode), device=device)
        elif x_data.dtype == np.int32:
            x = torch.tensor(x_data[i : i + batchsize], device=device, dtype=torch.long)
        else:
            x = torch.tensor(x_data[i : i + batchsize], device=device)
        # ミニバッチ（出力側）を作成
        calc_accuracy = False
        if y_type_str:
            y = torch.tensor(load_images(y_data, ids=perm[i : i + batchsize], size=y_size, with_noise=y_with_noise, mode=y_color_mode), device=device)
        elif y_data.dtype == np.int32:
            y = torch.tensor(y_data[i : i + batchsize], device=device, dtype=torch.long)
            calc_accuracy = True
        else:
            y = torch.tensor(y_data[i : i + batchsize], device=device)
        z = model(x)
        loss = loss_func(z, y)
        sum_loss += float(loss) * len(x)
        n_input += len(x)
        if calc_accuracy:
            y_cpu = y.to('cpu').detach().numpy().copy()
            z_cpu = z.to('cpu').detach().numpy().copy()
            n_failed += np.count_nonzero(np.argmax(z_cpu, axis=1) - y_cpu)
            del y_cpu
            del z_cpu
        del loss
        del z
        del y
        del x
    print('test loss = {0:.6f}'.format(sum_loss / n_input), file=sys.stderr)
    if calc_accuracy:
        acc = (n_samples_ev - n_failed) / n_samples_ev
        print('accuracy = {0:.2f}%'.format(100 * acc), file=sys.stderr)

    print('', file=sys.stderr)
    
    model.to('cpu')


# 単一データに対する推論
#   - device: 使用するデバイス
#   - model: 学習済みニューラルネットワークモデル（GPUに移動させる前）
#   - in_data: 入力データ（numpy.ndarray）
#   - module: 使用する関数（autoencoderによる圧縮・復元時のみ使用）
def predict_once(device, model, in_data, module=None):

    # モデルをGPU上に移動
    model = model.to(device)
    model.eval()

    # モデルにデータを入力
    x = torch.tensor([in_data], device=device)
    if module == 'encode':
        y = model.encode(x)
    elif module == 'decode':
        y = model.decode(x)
    else:
        y = model(x)
    if len(y.size()) == 2:
        y = F.softmax(y, dim=1)
    y_cpu = y.to('cpu').detach().numpy().copy()
    del y
    del x

    model.to('cpu')

    return y_cpu[0]
