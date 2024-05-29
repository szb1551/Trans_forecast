import os
import sys
import optparse
import time
import traci  # 导入TraCI库
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

prior_pro = 0.5 # 初始生成车的概率
num_vehicles_per_route = 50 #每条道上的初始车辆

def generate_routine(step):
    """
    每10个仿真步在id为"1"的路上生成一辆车
    """
    if step % 10 == 0:
        vehicle_id = "vehicle_%i" % (step/10)
        traci.vehicle.add(vehicle_id, routeID="route1")
        print(f"Generated a vehicle with id: {vehicle_id}")

def generate_vechicle_number(length, max_len):
    probability = 1 - length/max_len + 1e-9
    """
        这个函数根据给定的概率生成一个随机数，概率越高，生成接近于0.3的随机数的可能性越大。

        :param probability: 概率，介于0和1之间
        :param size: 生成随机数的数量
        :return: 生成的随机数数组
        """
    # 根据概率调整beta分布的参数

    a = probability * 5  # 'a'越大，分布越偏向于右侧
    b = (1 - probability) * 5  # 'b'越大，分布越偏向于左侧

    # 生成beta分布随机数，然后调整到[-0.3, 0.3]的范围内
    random_numbers = np.random.beta(a, b, size=1)  # 生成的是[0,1]之间的数
    random_numbers = random_numbers * 0.6 - 0.3  # 转换到[-0.3, 0.3]
    return random_numbers

def run():
    # 仿真循环
    # 获取所有道路的列表
    route_ids = traci.route.getIDList()

    print("Available routes:", route_ids)

    vec_all = np.zeros(len(route_ids), dtype=int) * num_vehicles_per_route
    edge_length_all = np.zeros(len(route_ids))
    max_len = 0
    route_to_lane = dict()
    for i, route_id in enumerate(route_ids):
        edges = traci.route.getEdges(route_id)
        edge_length_all[i] = sum(traci.lane.getLength(edge_id+"_0") for edge_id in edges)
        route_to_lane[route_id] = edges[0] + "_0"
        print(f"The length of route {route_id} is: {edge_length_all[i]} meters")
        max_len = max(max_len, edge_length_all[i])

    print(max_len)
    # 可以在此处基于路线生成车辆
    # 例如，为每个路线生成一个车辆
    # 执行一定时间步的仿真，可根据实际需求调整
    for step in tqdm(range(7*24*3600), desc="Processing"):
        for i, route_id in enumerate(route_ids):
            now_pro = generate_vechicle_number(edge_length_all[i], max_len)
            if np.random.rand()<=now_pro:
                vehicle_id = f"veh_{i}_{vec_all[i]}"
                traci.vehicle.add(vehicle_id, route_id, depart=1)  # depart参数可以根据需求调整
                vec_all[i] = vec_all[i] + 1
            # vehicle_num = traci.lane.getLastStepVehicleNumber(route_to_lane[route_id])
            # print(f"{route_id}上的车辆有{vehicle_num}辆")
        traci.simulationStep()


if __name__ == "__main__":
    sumoBinary = "sumo"  # 或者使用 "sumo" 作为无图形界面模式运行
    sumoCmd = [sumoBinary, "-c", "F:\我的文件\EV充电需求\Transformer_Forcast\TrafficSim\hello.sumocfg", "--no-warnings"]

    # 启动SUMO进程和TraCI控制
    traci.start(sumoCmd)

    run()
    traci.close()  # 关闭TraCI
