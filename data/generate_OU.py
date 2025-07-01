import os
import matplotlib.pyplot as plt
import numpy as np


def generate_OU(seed=None, alpha=0):
    np.random.seed(seed)
    ts = np.linspace(0, 1, 1001)
    t_start = float(ts[0])
    t_end = float(ts[-1])
    nsteps = 1000
    dt = (t_end - t_start) / nsteps

    #alpha = -50
    beta = 1

    dW = np.sqrt(dt) * np.random.randn(nsteps)  # 標準ブラウン運動の差分
    y = [1.0]
    for k in range(1, nsteps + 1):
        y_next = y[k - 1] + alpha * y[k - 1] * dt + beta * dW[k - 1]
        y.append(y_next)

    return ts, y

def save_multi_column_csv():
    alpha_list = [-2, -10, -20, -50, -100]
    all_series = []
    
    # 時間軸（共通）
    ts, _ = generate_OU(alpha=alpha_list[0])
    all_series.append(ts)

    plt.figure(figsize=(10, 6))

    # 各Hに対する系列を収集
    for a in alpha_list:
        _, X = generate_OU(alpha=a)
        all_series.append(X)
        plt.plot(ts, X, label=f"alpha = {a}")

    # スタックして (nsteps+1, len(H) + 1) の行列に変換
    data = np.stack(all_series, axis=1)

    # ヘッダー作成
    header = "t," + ",".join([f"alpha={a}" for a in alpha_list])
    
    # 保存
    #os.makedirs("fOU_csv", exist_ok=True)
    filename = "OU_multicolumn.csv"
    np.savetxt(filename, data, delimiter=",", header=header, comments='')
    print(f"Saved: {filename}")


    plt.title("Simulated fOU Process: $dX_t = \\alpha X_t dt + \\beta dB_t^H$")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("OU_plot.png")
    plt.show()



if __name__=='__main__':
    save_multi_column_csv()