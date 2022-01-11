#  you can see the original source code from following:
#　https://qiita.com/Yuya-Shimizu/items/c7b69b4dfd63fb8facfa
"""
2021/02/14
@Yuya Shimizu

伝達関数モデル
"""
from control import tf, tfdata

##伝達関数を作成
#①式
Np = [0, 1]     #伝達関数の分子多項式の係数（0*s + 1）
Dp = [1, 2, 3]     #伝達関数の分母多項式の係数（1*s^2 + 2*s + 3）

P = tf(Np, Dp)

print(f"①式<伝達関数>\n{P}\n")


#②式
Np = [1, 2]     #伝達関数の分子多項式の係数（1*s + 2）
Dp = [1, 5, 3, 4]     #伝達関数の分母多項式の係数（1*s^3 + 5*s^2 + 3*s + 4）

P = tf(Np, Dp)

print(f"②式<伝達関数>\n{P}\n")


#③式
Np = [1, 3]     #伝達関数の分子多項式の係数（1*s + 3）
Dp = [1, 5, 8, 4]     #伝達関数の分母多項式の係数（(1*s + 1)(1*s + 2)^2  =  1*s^3 + 5*s^2 + 8*s + 4）


P = tf(Np, Dp)

print(f"③式<伝達関数>\n{P}\n")


#③式の別方法
Np1 = [1, 3]     #伝達関数の分子多項式の係数（1*s + 3）
Dp1 = [0, 1]     #伝達関数の分母多項式の係数（0*s + 1）
Np2 = [0, 1]     #伝達関数の分子多項式の係数（0*s + 1）
Dp2 = [1, 1]     #伝達関数の分母多項式の係数（1*s + 1）
Np3 = [0, 1]     #伝達関数の分子多項式の係数（0*s + 1）
Dp3 = [1, 2]     #伝達関数の分母多項式の係数（1*s + 2）

P1 = tf(Np1, Dp1)
P2 = tf(Np2, Dp2)
P3 = tf(Np3, Dp3)

P = P1 * P2 * P3**2

print(f"③式<伝達関数>（分割法）\n{P}\n")


##伝達関数の分子と分母の係数を抽出する
#シンプルに抽出する
numP = P.num
denP = P.den

print(f"<係数抽出>\n\n分子係数：{numP}, 分母係数：{denP}\n\n")


#入れ子を避ける抽出
[[numP]], [[denP]] = tfdata(P)
print(f"<係数抽出>（入れ子を避ける抽出）\n\n分子係数：{numP}, 分母係数：{denP}\n")