from platypus import NSGAII, Problem, Real, nondominated, Integer
import matplotlib.pyplot as plt
from platypus.problems import DTLZ2


def main():
    # 2変数2目的の問題
    problem = Problem(2, 2)
    # 最小化or最大化を設定
    problem.directions[:] = Problem.MINIMIZE
    # 決定変数の範囲を設定
    int1 = Integer(0, 100)
    int2 = Integer(0, 50)
    problem.types[:] = [int1, int2]
    problem.function = objective
    # アルゴリズムを設定し, 探索実行
    algorithm = NSGAII(problem, population_size=200)
    algorithm.run(1000)

    nondominated_solutions = nondominated(algorithm.result)

    # グラフを描画
    plt.scatter([s.objectives[0] for s in nondominated_solutions if s.feasible],
               [s.objectives[1] for s in nondominated_solutions if s.feasible])
    plt.show()

def objective(vars):
    x1 = int(vars[0])
    x2 = int(vars[1])
    return [2*(x1**2) + x2**2, -x1**2 -2*(x2**2)]


if __name__ == '__main__':
    main()