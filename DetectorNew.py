import xml.etree.ElementTree as ET

# 读取网络文件
net_file = 'network.net.xml'
# 输出的检测器配置文件
output_file = 'TrafficSim/detectors.add.xml'

# 解析网络XML文件
tree = ET.parse(net_file)
root = tree.getroot()

# 创建'additional'根节点
add_root = ET.Element('additional')

# 遍历所有的lane节点
for edge in root.findall('edge'):
    if edge.get('function') == 'internal':
        continue  # 跳过内部边（internal edges）

    for lane in edge.findall('lane'):
        lane_id = lane.get('id')
        detector_id = "detector_" + lane_id

        # 创建inductionLoop节点
        induction_loop = ET.SubElement(add_root, 'inductionLoop', {
            'id': detector_id,
            'lane': lane_id,
            'pos': '42',  # 检测器的位置，这里是硬编码示例
            'freq': '3600',  # 检测周期
            'file': 'out.xml'  # 输出文件名
        })

# 格式化并写入到输出文件
tree = ET.ElementTree(add_root)
tree.write(output_file, encoding='utf-8', xml_declaration=True)

print(f"Detectors have been written to {output_file}")