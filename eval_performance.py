# coding: utf-8

import json

def load_set(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        result = set()
        for line in f:
            line = line.strip()
            if line:
                # 如果包含分隔符，说明是新格式，取第一部分作为节点ID
                if ' | ' in line:
                    node_id = line.split(' | ')[0].strip()
                else:
                    # 旧格式，整行就是节点ID
                    node_id = line
                result.add(node_id)
        return result

def load_attack_connections(test_file):
    """加载攻击连接信息，包括源节点和目标节点"""
    attack_connections = set()
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('type') == 'udp attack':
                    src_node = data.get('src_ip_port', '')
                    dest_node = data.get('dest_ip_port', '')
                    if src_node and dest_node:
                        attack_connections.add((src_node, dest_node))
            except json.JSONDecodeError:
                continue
    return attack_connections

# 文件路径
real_attack_file = 'dataset/mydata-flow/net_attack.txt'
alarm_file = 'dataset/mydata-flow/net_alarms.txt'
test_file = 'dataset/mydata-flow/test_conn_23_15-16.json'

# 加载集合
real_attack = load_set(real_attack_file)
alarms = load_set(alarm_file)
attack_connections = load_attack_connections(test_file)

# 创建攻击相关的目标节点集合
attack_target_nodes = set()
for src, dest in attack_connections:
    attack_target_nodes.add(dest)

# 计算指标（改进版）
TP = len(real_attack & alarms)  # 源节点正确检测
FP = len(alarms - real_attack - attack_target_nodes)  # 排除攻击相关的目标节点
FN = len(real_attack - alarms)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"真实攻击节点数: {len(real_attack)}")
print(f"检测为异常节点数: {len(alarms)}")
print(f"攻击相关的目标节点数: {len(attack_target_nodes)}")
print(f"真正(TP): {TP}")
print(f"假正(FP): {FP}")
print(f"假负(FN): {FN}")
print(f"精度(Precision): {precision:.4f}")
print(f"召回率(Recall): {recall:.4f}")
print(f"F1分数(F1-score): {f1:.4f}")

attack_targets_detected = len(alarms & attack_target_nodes)
#print(f"攻击目标节点被检测数: {attack_targets_detected}")
#print(f"攻击目标节点总数: {len(attack_target_nodes)}")