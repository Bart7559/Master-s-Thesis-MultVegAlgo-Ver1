import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

random.seed(42)

# 各野菜の価格の率表
cucumber_rate = {
    "cucumber_No2": 62,
    "cucumber_No3": 60.5,
    "cucumber_No4": 54.5,
    "cucumber_No5": 75,
    "cucumber_No6": 59.5,
    "cucumber_No7": 60.5,
    "cucumber_No8": 52.5,
}

carrot_rate = {
    "carrot_No2": 65.5,
    "carrot_No3": 68,
    "carrot_No4": 55.5,
    "carrot_No5": 60.5,
    "carrot_No6": 58.5,
    "carrot_No7": 65.5,
    "carrot_No8": 55.5,
}

tomato_rate = {
    "tomato_No2": 75.5,
    "tomato_No3": 77.5,
    "tomato_No4": 73,
    "tomato_No5": 78.5,
    "tomato_No6": 73,
    "tomato_No7": 74.5,
    "tomato_No8": 68,
}

# 現在の市場価格
cucumber_price = 70
carrot_price = 80
tomato_price = 50

# 価格と率を掛けて価格表を作成
cucumber_prices = {key: cucumber_price * rate / 100 for key, rate in cucumber_rate.items()}
carrot_prices = {key: carrot_price * rate / 100 for key, rate in carrot_rate.items()}
tomato_prices = {key: tomato_price * rate / 100 for key, rate in tomato_rate.items()}

# 野菜のデータフレーム作成
cucumber_df = pd.DataFrame(list(cucumber_rate.items()), columns=["No.", "Rate"])
cucumber_df["Price"] = cucumber_df["No."].map(cucumber_prices)
cucumber_df["Stock"] = [random.randint(0, 30) for _ in range(len(cucumber_df))]

carrot_df = pd.DataFrame(list(carrot_rate.items()), columns=["No.", "Rate"])
carrot_df["Price"] = carrot_df["No."].map(carrot_prices)
carrot_df["Stock"] = [random.randint(0, 30) for _ in range(len(carrot_df))]

tomato_df = pd.DataFrame(list(tomato_rate.items()), columns=["No.", "Rate"])
tomato_df["Price"] = tomato_df["No."].map(tomato_prices)
tomato_df["Stock"] = [random.randint(0, 5) for _ in range(len(tomato_df))]

# 野菜のデータフレーム統合
cucumber_df["Vegetable_Type"] = "Cucumber"
carrot_df["Vegetable_Type"] = "Carrot"
tomato_df["Vegetable_Type"] = "Tomato"

all_vegetables_df = pd.concat([cucumber_df, carrot_df, tomato_df]).sort_values(by="Price", ascending=False).reset_index(drop=True)

# 目標金額を設定
target_price = 500

# 箱数の計算
def calculate_box_count_by_target(total_stock_price, target_price):
    return max(int(total_stock_price // target_price), 1)

total_stock_price = all_vegetables_df["Price"].dot(all_vegetables_df["Stock"])
box_num = calculate_box_count_by_target(total_stock_price, target_price)
print(f"Total Stock Price: {total_stock_price}, Box Count: {box_num}\n")

# ボックスの初期化
def initialize_boxes(box_num):
    return [
        {"Box_Number": i, "Box_Value": 0, "Vegetables": [], "Total_Count": 0}
        for i in range(box_num)
    ]

# 初期解を生成するヒューリスティック法
def heuristic_initial_solution(all_vegetables_df, box_num, skip_threshold=30):
    boxes = initialize_boxes(box_num)
    step = 0

    while all_vegetables_df["Stock"].sum() > 0:  # 在庫がある限り繰り返す
        step += 1
        all_vegetables_df = all_vegetables_df.sort_values(by="Price", ascending=False).reset_index(drop=True)
        min_box_value = min([box["Box_Value"] for box in boxes])  # 箱の最低価値

        for idx in range(len(boxes)):
            for i, row in all_vegetables_df.iterrows():
                if row["Stock"] > 0:
                    # スキップ条件: 最低価値の箱との差が閾値を超える場合
                    if (boxes[idx]["Box_Value"] - min_box_value) >= skip_threshold:
                        continue

                    # 割り付け処理
                    boxes[idx]["Box_Value"] += row["Price"]
                    boxes[idx]["Vegetables"].append(row["No."])
                    boxes[idx]["Total_Count"] += 1
                    all_vegetables_df.loc[i, "Stock"] -= 1  # 修正済み
                    break  # 1回割り付けたら次の箱へ

        # 1週ごとに箱をソート（価値が小さい順）
        boxes = sorted(boxes, key=lambda x: x["Box_Value"])

    # 初期解を個体として生成
    individual = []
    for box_idx, box in enumerate(boxes):
        for vegetable in box["Vegetables"]:
            vegetable_idx = all_vegetables_df[all_vegetables_df["No."] == vegetable].index[0]
            individual.append(box_idx)

    return individual, boxes

# 評価関数
def evaluate(individual, data, box_num):
    boxes = initialize_boxes(box_num)
    stock_tracker = data["Stock"].copy()
    vegetable_index = 0

    for box_index in individual:
        while vegetable_index < len(data) and stock_tracker.iloc[vegetable_index] == 0:
            vegetable_index += 1

        if vegetable_index < len(data) and box_index < box_num:
            vegetable = data.iloc[vegetable_index]
            boxes[box_index]["Box_Value"] += vegetable["Price"]
            boxes[box_index]["Vegetables"].append(vegetable["No."])
            boxes[box_index]["Total_Count"] += 1
            stock_tracker.iloc[vegetable_index] -= 1

    box_values = [box["Box_Value"] for box in boxes]
    cucumber_counts = [sum(1 for v in box["Vegetables"] if "Cucumber" in v) for box in boxes]
    tomato_counts = [sum(1 for v in box["Vegetables"] if "Tomato" in v) for box in boxes]
    carrot_counts = [sum(1 for v in box["Vegetables"] if "Carrot" in v) for box in boxes]

    value_variance = np.var(box_values)
    cucumber_variance = np.var(cucumber_counts)
    tomato_variance = np.var(tomato_counts)
    carrot_variance = np.var(carrot_counts)

    # 分散と構成均等性を評価
    return (value_variance + 0.1 * (cucumber_variance + tomato_variance + carrot_variance),)

# 修復処理
def repair_individual(individual, data, box_num):
    stock = data["Stock"].copy()
    repaired = []
    box_values = [0] * box_num

    # 野菜の優先順位付け（希少なものから割り付け）
    vegetable_priority = data.sort_values(by="Stock", ascending=True).reset_index()

    for i, row in vegetable_priority.iterrows():
        vegetable_idx = row["index"]
        while stock.iloc[vegetable_idx] > 0:
            min_box_index = box_values.index(min(box_values))
            repaired.append(min_box_index)
            box_values[min_box_index] += row["Price"]
            stock.iloc[vegetable_idx] -= 1

    return repaired

# 遺伝的アルゴリズムで改善
def improve_solution_with_ga(initial_individual, data, box_num, generations=50):
    toolbox = base.Toolbox()

    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: initial_individual.copy())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=box_num - 1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, data=data, box_num=box_num)

    population = toolbox.population(n=200)
    evaluation_progress = []  # 評価関数の進捗を記録

    for gen in range(generations):
        current_mutpb = 0.02 + 0.08 * (gen / generations)  # 突然変異率を徐々に上げる
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=current_mutpb)

        for ind in offspring:
            repaired = repair_individual(ind, data, box_num)
            ind[:] = repaired

        elite = tools.selBest(population, k=5)
        population[:] = toolbox.select(offspring + elite, len(population))

        best_ind = tools.selBest(population, k=1)[0]
        best_eval = evaluate(best_ind, data, box_num)[0]
        evaluation_progress.append(best_eval)
        print(f"Generation {gen}: Best Variance: {best_eval}")

    # 評価関数の進捗をプロット
    plot_evaluation_progress(evaluation_progress)

    return tools.selBest(population, k=1)[0]

# 評価関数の進捗をプロット
def plot_evaluation_progress(evaluation_progress):
    generations = list(range(len(evaluation_progress)))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, evaluation_progress, marker='o', linestyle='-', color='b', label="Best Evaluation Function")
    plt.title("GA Performance Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Evaluation Function (Variance)")
    plt.grid(True)
    plt.legend()
    plt.show()

# メイン処理
def main():
    total_stock_price = all_vegetables_df["Price"].dot(all_vegetables_df["Stock"])
    box_num = calculate_box_count_by_target(total_stock_price, target_price)

    # ヒューリスティック法による初期解
    initial_individual, heuristic_boxes = heuristic_initial_solution(all_vegetables_df, box_num)

    # 初期解の分散を計算
    print("Heuristic Initial Solution Variance:", np.var([box["Box_Value"] for box in heuristic_boxes]))

    # 遺伝的アルゴリズムで解を改善
    best_individual = improve_solution_with_ga(initial_individual, all_vegetables_df, box_num)

    # 最良解をボックスに反映
    best_boxes = initialize_boxes(box_num)
    stock_tracker = all_vegetables_df["Stock"].copy()
    vegetable_index = 0

    for box_index in best_individual:
        while vegetable_index < len(all_vegetables_df) and stock_tracker.iloc[vegetable_index] == 0:
            vegetable_index += 1

        if vegetable_index < len(all_vegetables_df) and box_index < box_num:
            vegetable = all_vegetables_df.iloc[vegetable_index]
            best_boxes[box_index]["Box_Value"] += vegetable["Price"]
            best_boxes[box_index]["Vegetables"].append(vegetable["No."])
            best_boxes[box_index]["Total_Count"] += 1
            stock_tracker.iloc[vegetable_index] -= 1

    # 結果の表示
    result_df = pd.DataFrame([
        {
            "Box_Number": box["Box_Number"],
            "Total_Value": box["Box_Value"],
            "Cucumber_Count": sum(1 for v in box["Vegetables"] if "Cucumber" in v),
            "Tomato_Count": sum(1 for v in box["Vegetables"] if "Tomato" in v),
            "Carrot_Count": sum(1 for v in box["Vegetables"] if "Carrot" in v),
            "Details": ", ".join(box["Vegetables"])
        }
        for box in best_boxes
    ])
    result_df["Total_Amount"] = result_df["Total_Value"]

    print("\n各箱の内容:")
    print(result_df)
    print("\n合計金額の分散:", np.var(result_df["Total_Amount"]))

if __name__ == "__main__":
    main()
