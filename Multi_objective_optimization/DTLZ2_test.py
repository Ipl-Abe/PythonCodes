from platypus import NSGAII, Problem, Real, nondominated, Integer
import matplotlib.pyplot as plt
from platypus.problems import DTLZ2


def main():
    # 問題, アルゴリズムを設定し, 探索実行
    problem = DTLZ2(2)
    algorithm = NSGAII(problem, population_size=100)
    algorithm.run(10000)

    # 非劣解をとりだす
    nondominated_solutions = nondominated(algorithm.result)

    # グラフを描画
    plt.scatter([s.objectives[0] for s in nondominated_solutions if s.feasible],
               [s.objectives[1] for s in nondominated_solutions if s.feasible])
    plt.show()

if __name__ == '__main__':
    main()