import os
import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM


def generate_fOU(hurst, nsteps=1000, alpha=-0.05, beta=0.1):
    t_start = 0.0
    t_end = 1.0
    dt = (t_end - t_start) / nsteps
    ts = np.linspace(t_start, t_end, nsteps + 1)

    B = FBM(n=nsteps, hurst=hurst, length=t_end - t_start).fbm()
    dB = np.diff(B)

    y = [1.0]
    for k in range(1, nsteps + 1):
        y_next = y[k - 1] + alpha * y[k - 1] * dt + beta * dB[k - 1]
        y.append(y_next)

    return ts, np.array(y)


def save_multi_column_csv():
    hurst_list = [0.1, 0.2, 0.3, 0.4]
    all_series = []
    
    # 時間軸（共通）
    ts, _ = generate_fOU(hurst_list[0])
    all_series.append(ts)

    plt.figure(figsize=(10, 6))

    # 各Hに対する系列を収集
    for H in hurst_list:
        _, X = generate_fOU(H)
        all_series.append(X)
        plt.plot(ts, X, label=f"H = {H}")

    # スタックして (nsteps+1, len(H) + 1) の行列に変換
    data = np.stack(all_series, axis=1)

    # ヘッダー作成
    header = "t," + ",".join([f"H={H}" for H in hurst_list])
    
    # 保存
    #os.makedirs("fOU_csv", exist_ok=True)
    filename = "fOU_multicolumn.csv"
    np.savetxt(filename, data, delimiter=",", header=header, comments='')
    print(f"Saved: {filename}")


    plt.title("Simulated fOU Process: $dX_t = \\alpha X_t dt + \\beta dB_t^H$")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fOU_plot.png")
    plt.show()



if __name__=='__main__':
    save_multi_column_csv()