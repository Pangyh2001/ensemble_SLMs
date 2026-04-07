import math

def calculate_std(numbers, ddof=1):
    """
    计算标准差
    
    参数:
    numbers: 包含数字的列表
    ddof: 自由度调整，ddof=0计算总体标准差，ddof=1计算样本标准差(默认)
    
    返回:
    标准差
    """
    if len(numbers) == 0:
        return 0.0
    
    # 计算均值
    mean = sum(numbers) / len(numbers)
    
    # 计算方差
    variance = sum((x - mean) ** 2 for x in numbers) / (len(numbers) - ddof)
    
    # 计算标准差
    std = math.sqrt(variance)
    
    return std

def main():
    """主函数"""
    print("=" * 40)
    print("三 个 数 标 准 差 计 算 器")
    print("=" * 40)
    
    # 获取三个数
    numbers = []
    for i in range(3):
        while True:
            try:
                value = float(input(f"请输入第 {i+1} 个数: "))
                numbers.append(value)
                break
            except ValueError:
                print("输入无效，请输入一个数字！")
    
    print("\n" + "-" * 40)
    print(f"输入的数据: {numbers}")
    print(f"数据个数: {len(numbers)}")
    print(f"总和: {sum(numbers):.4f}")
    print(f"均值: {sum(numbers)/len(numbers):.4f}")
    
    # 计算两种标准差
    std_population = calculate_std(numbers, ddof=0)  # 总体标准差
    std_sample = calculate_std(numbers, ddof=1)      # 样本标准差
    
    print("\n" + "-" * 40)
    print("标 准 差 结 果")
    print("-" * 40)
    print(f"总体标准差 (σ, ddof=0): {std_population:.6f}")
    print(f"样本标准差 (s, ddof=1): {std_sample:.6f}")
    
    # 计算过程展示
    print("\n" + "-" * 40)
    print("计 算 过 程")
    print("-" * 40)
    
    mean = sum(numbers) / len(numbers)
    print(f"1. 计算均值 μ = {numbers[0]:.4f} + {numbers[1]:.4f} + {numbers[2]:.4f} / 3")
    print(f"             = {sum(numbers):.4f} / 3")
    print(f"             = {mean:.4f}")
    
    print(f"\n2. 计算每个数与均值的差:")
    for i, x in enumerate(numbers):
        diff = x - mean
        print(f"   x{i+1} - μ = {x:.4f} - {mean:.4f} = {diff:.6f}")
    
    print(f"\n3. 计算差的平方:")
    squares = []
    for i, x in enumerate(numbers):
        diff = x - mean
        square = diff ** 2
        squares.append(square)
        print(f"   (x{i+1} - μ)² = ({x:.4f} - {mean:.4f})² = {square:.8f}")
    
    sum_squares = sum(squares)
    print(f"\n4. 平方和 Σ(xᵢ - μ)² = {sum_squares:.8f}")
    
    print(f"\n5. 计算方差:")
    print(f"   总体方差 = Σ(xᵢ - μ)² / N = {sum_squares:.8f} / {len(numbers)}")
    print(f"             = {sum_squares/len(numbers):.8f}")
    print(f"\n   样本方差 = Σ(xᵢ - μ)² / (N-1) = {sum_squares:.8f} / {len(numbers)-1}")
    print(f"             = {sum_squares/(len(numbers)-1):.8f}")
    
    print(f"\n6. 计算标准差 (开平方根):")
    print(f"   总体标准差 = √({sum_squares/len(numbers):.8f}) = {std_population:.6f}")
    print(f"   样本标准差 = √({sum_squares/(len(numbers)-1):.8f}) = {std_sample:.6f}")
    
    print("\n" + "=" * 40)
    print(f"最终结果: {mean:.4f} ± {std_sample:.6f} (样本标准差)")
    print("=" * 40)

if __name__ == "__main__":
    main()