def print_detailed_possibilities(corrupted, detailed_possibilities):
    for possibility in detailed_possibilities:
        # 提取分配和概率部分
        assignments = possibility[:-1]
        probability = possibility[-1]

        # 构造输出字符串
        result_str = "Pr["
        for i in range(len(corrupted)):
            if i > 0:
                result_str += ", "
            result_str += f"O(P{corrupted[i]})=P{assignments[i]}"
        result_str += f"] = {probability}"

        # 打印结果
        print(result_str)

# 示例输入
corrupted = [1, 2, 3, 6, 8, 9]
detailed_possibilities = [
    [1, 2, 3, 4, 5, 6, 0.00030864197530864197],
    # 你可以在此处添加更多详细的可能性
]

# 调用函数打印结果
print_detailed_possibilities(corrupted, detailed_possibilities)