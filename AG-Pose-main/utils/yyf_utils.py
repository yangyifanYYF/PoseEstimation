import pickle

# 打开.pkl文件并加载数据
with open('/workspace/code/data/PoseData/gts/real_test/results_real_test_scene_1_0000.pkl', 'rb') as file:  # 'rb' 表示以二进制读取模式打开文件
    data = pickle.load(file)

# 打印加载的数据
print(data)
