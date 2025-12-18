import pickle
import numpy as np


def print_pkl_detail(data, indent=0, max_items=5, prefix="数据"):
    """
    递归打印PKL文件的所有层级内容（自动适配字典、列表、元组、NumPy数组等）
    :param data: 要打印的PKL数据
    :param indent: 缩进量（区分层级）
    :param max_items: 每个层级最多打印的元素数（避免刷屏）
    :param prefix: 打印前缀（区分不同文件/字段）
    """
    # 缩进符号（用空格体现层级）
    indent_str = "  " * indent

    # 1. 处理NumPy数组
    if isinstance(data, np.ndarray):
        print(f"{indent_str}{prefix} → 类型：NumPy数组 | 形状：{data.shape} | 数据类型：{data.dtype}")
        # 短数组打印全部，长数组打印前max_items个
        print(f"{indent_str}  内容预览：{data if data.size <= max_items else data[:max_items]}")
        return

    # 2. 处理字典（键值对形式）
    if isinstance(data, dict):
        print(f"{indent_str}{prefix} → 类型：字典 | 键数量：{len(data)}")
        for i, (key, value) in enumerate(data.items()):
            if i >= max_items:
                print(f"{indent_str}  键...（省略剩余 {len(data) - max_items} 个键）")
                break
            print_pkl_detail(value, indent + 2, max_items, prefix=f"键'{key}'的值")
        return

    # 3. 处理列表/元组（序列形式）
    if isinstance(data, (list, tuple)):
        type_name = "列表" if isinstance(data, list) else "元组"
        print(f"{indent_str}{prefix} → 类型：{type_name} | 元素数量：{len(data)}")
        for i, item in enumerate(data):
            if i >= max_items:
                print(f"{indent_str}  元素...（省略剩余 {len(data) - max_items} 个元素）")
                break
            print_pkl_detail(item, indent + 2, max_items, prefix=f"第{i + 1}个元素")
        return

    # 4. 处理简单类型（字符串、数字、布尔等，直接打印）
    print(f"{indent_str}{prefix} → 类型：{type(data).__name__} | 内容：{data}")


# ------------------- 加载并打印PKL文件 -------------------
pkl_files = [
    "sdd_test.pkl",
    "sdd_train.pkl"
]

for file_path in pkl_files:
    print(f"\n==================================================")
    print(f"正在读取文件：{file_path}")
    print(f"==================================================")
    try:
        with open(file_path, 'rb') as f:
            pkl_data = pickle.load(f)
        print_pkl_detail(pkl_data, prefix="根数据")
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件！请检查路径是否正确：{file_path}")
    except Exception as e:
        print(f"❌ 读取失败：{str(e)}")