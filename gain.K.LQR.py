import numpy as np
from scipy.linalg import (
    solve_continuous_are,
    solve_continuous_lyapunov,
    block_diag,
)

np.set_printoptions(precision=6, suppress=True)


def skew(v):
    """將三維向量轉成叉積矩陣。"""
    a, b, c = v
    return np.array([
        [0.0, -c,   b],
        [c,    0.0, -a],
        [-b,   a,   0.0],
    ])


# 1. 固定示範參數
I3 = np.eye(3)
Z3 = np.zeros((3, 3))

omega = np.array([0.1, 0.1, 0.2])
T_bar = 10.0
e_T = np.array([0.0, 0.0, 1.0])

W = skew(omega)
C = skew(-T_bar * e_T)

# 三個局部設計模型相同
A_local = -W
B_local = I3

Q_list = [4.0 * I3, 9.0 * I3, 16.0 * I3]
R_list = [I3.copy() for _ in range(3)]
names = ["位置 p", "速度 v", "姿態 r"]

P_list = []
K_list = []


# 2. 分別解三組 LQR
for name, Q_i, R_i in zip(names, Q_list, R_list):
    P_i = solve_continuous_are(
        A_local, B_local, Q_i, R_i
    )

    # K = R^{-1} B^T P
    # 用 solve 避免顯式計算矩陣反矩陣
    K_i = np.linalg.solve(R_i, B_local.T @ P_i)

    A_closed_i = A_local - B_local @ K_i

    care_residual = (
        A_local.T @ P_i
        + P_i @ A_local
        - P_i @ B_local @ K_i
        + Q_i
    )

    P_list.append(P_i)
    K_list.append(K_i)

    print(f"\n===== {name} 子系統 =====")
    print("P =\n", P_i)
    print("K =\n", K_i)
    print("閉迴路特徵值 =", np.linalg.eigvals(A_closed_i))
    print("Riccati 殘差 =", np.linalg.norm(care_residual))


# 3. 組回完整九維系統，保留耦合
A0 = np.block([
    [-W, I3, Z3],
    [Z3, -W, C],
    [Z3, Z3, -W],
])

B = np.eye(9)
K = block_diag(*K_list)
A_cl = A0 - B @ K

eigenvalues = np.linalg.eigvals(A_cl)

print("\n===== 完整九維系統 =====")
print("K =\n", K)
print("閉迴路特徵值 =\n", eigenvalues)
print("最大特徵值實部 =", np.max(eigenvalues.real))
print("是否 Hurwitz =", np.all(eigenvalues.real < 0.0))


# 4. 檢查局部 P 直接拼起來是否足以證明穩定
P_local = block_diag(*P_list)

D_local = A_cl.T @ P_local + P_local @ A_cl

print("\n===== 直接拼接局部 P =====")
print(
    "Lyapunov 導數矩陣最大特徵值 =",
    np.max(np.linalg.eigvalsh(D_local)),
)


# 5. 固定 K，重新解完整系統的 Lyapunov 方程
Q = block_diag(*Q_list)
R = block_diag(*R_list)

S = Q + K.T @ R @ K

# SciPy 解 A X + X A^T = Q_rhs
# 因此傳入 A_cl.T 與 -S
P_full = solve_continuous_lyapunov(A_cl.T, -S)

lyap_residual = A_cl.T @ P_full + P_full @ A_cl + S

print("\n===== 完整系統的 P =====")
print("P_full =\n", P_full)
print("P_full 最小特徵值 =", np.min(np.linalg.eigvalsh(P_full)))
print("Lyapunov 殘差 =", np.linalg.norm(lyap_residual))