import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import isomorphism
import json
from queue import Queue
import random
from itertools import islice
import concurrent.futures
from tqdm import tqdm
from collections import deque
import signal


class TimeoutException(Exception):
    pass


class Generation:
    #problem 0:cycle_detection 1:Connectivity 2:Bipartite_graph_check 3:Topological_sort 4:Shortest_path 5:Maximum_triangle_sum 6:Maximum_flow 7:Hamilton_path 8:Subgraph_matching
    def __init__(self,problem,G,G2=None,source=-1,target=-1,result={}):
        self.problem=problem
        self.G=G
        self.G2=G2
        self.source=source
        self.target=target
        self.result=result
        
    def hamiltonian_cycles(self, graph):  
        all_hamiltonian_cycles = []
        
        nodes = list(graph.nodes())
        
        def search(current_node, visited, path):
            path.append(current_node)
            visited.add(current_node)
            
            if len(path) == len(nodes) and path[-1] in graph.neighbors(path[0]):
                all_hamiltonian_cycles.append(path[:])
            else:
                for neighbor in graph.neighbors(current_node):
                    if neighbor not in visited:
                        search(neighbor, visited.copy(), path.copy())
            
            path.pop()
            visited.remove(current_node)
        
        for node in nodes:
            search(node, set(), [])
        
        return all_hamiltonian_cycles

    def limited_simple_paths(self, graph, source, target, max_paths=3):
        def dfs(current_path):
            current_node = current_path[-1]
            
            if current_node == target:
                yield list(current_path)
                return
              
            for neighbor in graph.neighbors(current_node):
                if neighbor not in current_path:  # 避免环路
                    current_path.append(neighbor)
                    yield from dfs(current_path)
                    current_path.pop()

        # 调用深度优先搜索生成器
        return islice(dfs([source]), max_paths)
    

    def find_k_cycles(self, G, k):
        def dfs_cycle(path, visited):
            current = path[-1]
            for neighbor in G[current]:
                if neighbor == path[0] and len(path) > 2:
                    cycles.append(path[:])
                    if len(cycles) >= k: 
                        return True
                elif neighbor not in visited:  
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs_cycle(path, visited): 
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False

        cycles = []
        for node in G.nodes:
            if dfs_cycle([node], {node}):
                break 
        return cycles
    

    def find_topological_sorts_with_limit(self, G, limit, timeout=1):
        def dfs_sort(current_sort, visited):
            if len(current_sort) == len(G):  # 如果当前排序已经包含所有节点
                yield current_sort[:]
                return
            for node in G.nodes:
                if node not in visited and all(pred in visited for pred in G.predecessors(node)):
                    # 当前节点未访问，且所有前驱节点已经访问
                    visited.add(node)
                    current_sort.append(node)
                    yield from dfs_sort(current_sort, visited)  # 使用递归生成器
                    current_sort.pop()
                    visited.remove(node)

        def run_dfs_sort():
            return islice(dfs_sort([], set()), limit)  # 用 islice 限制生成器输出

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_dfs_sort)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print("Function call timed out")
                return []
    
    import networkx as nx

    def find_multiple_maximum_flows(self, G, source, target, num_solutions=3):
        # List to store results
        results = []
        
        for i in range(num_solutions):
            # Compute the maximum flow
            flow_value, flow_dict = nx.maximum_flow(G, source, target)
            results.append({'flow_value': flow_value, 'flow_dict': flow_dict})
            
            print(f"Solution {i+1}:")
            print("Flow Value:", flow_value)
            print("Flow Distribution:", flow_dict)
            print()
            
            # Modify the graph to find a new solution
            # Reduce the capacity or remove edges with non-zero flow to force a new path
            for u in flow_dict:
                for v, flow in flow_dict[u].items():
                    if flow > 0:
                        if G.has_edge(u, v):
                            G[u][v]['capacity'] -= flow
                            if G[u][v]['capacity'] <= 0:
                                G.remove_edge(u, v)
        
        return results
    

    def is_bipartite(self, graph):
        color = {}  # Dictionary to store the color of each node (0 or 1)
        set1, set2 = set(), set()  # To store the two sets of the bipartite graph
        conflict_pairs = set()  # To store conflicting node pairs
        flag = True  # To indicate if the graph is bipartite

        for node in graph.nodes():  # Iterate through all nodes, handling disconnected graphs
            if node not in color:
                queue = deque([node])  # BFS queue
                color[node] = 0  # Start by coloring the first node with 0

                while queue:
                    current = queue.popleft()
                    current_color = color[current]

                    # Add the current node to the appropriate set based on its color
                    if current_color == 0:
                        set1.add(current)
                    else:
                        set2.add(current)

                    # Iterate through the neighbors of the current node
                    for neighbor in graph.neighbors(current):
                        if neighbor not in color:  # If the neighbor hasn't been colored yet
                            color[neighbor] = 1 - current_color  # Assign the opposite color
                            queue.append(neighbor)
                        elif color[neighbor] == current_color:  # If the neighbor has the same color
                            # Record the conflicting nodes
                            if current < neighbor:
                                conflict_pairs.add((current, neighbor))
                            else:
                                conflict_pairs.add((neighbor, current))
                            flag = False  # The graph is not bipartite
                            if len(conflict_pairs) >= 3:  # Limit to recording 3 conflict pairs
                                return False, list(conflict_pairs)

        # Return the result
        if flag:
            return True, (list(set1), list(set2))  # Return the two sets if the graph is bipartite
        else:
            return False, list(conflict_pairs)  # Return the conflict pairs if not bipartite
    
    def timeout_handler(signum, frame):
        raise TimeoutException
    
    signal.signal(signal.SIGALRM, timeout_handler)

    def generate_path(self):
        problem=self.problem

        if problem == 0:  # cycle detection
            try:
                signal.alarm(3)  # 设置3秒的闹钟
                cycles = self.find_k_cycles(self.G, 3)
                signal.alarm(0)  # 取消闹钟
                cycles_list = [cycle for cycle in cycles]
                self.result['paths'] = cycles_list
            except TimeoutException:
                self.result['paths'] = -1
                print("Error: find_k_cycles function timed out after 3 seconds")
            except nx.NetworkXNoPath:
                self.result['paths'] = []
            finally:
                signal.alarm(0)  # 确保闹钟被取消

            return self.result
       
        if problem==1:      #conectivity     
            source = self.source
            target = self.target
            try:
                # paths = nx.all_simple_paths(self.G,source=source,target=target)
                signal.alarm(3)  # 设置3秒的闹钟
                paths = self.limited_simple_paths(self.G, source, target, max_paths=3)
                signal.alarm(0)  # 取消闹钟
                paths_list = [path for path in paths]
                print(paths_list)
                self.result['source'] = source
                self.result['target'] = target
                self.result['paths'] = paths_list
            except nx.NetworkXNoPath:
                self.result['source'] = source
                self.result['target'] = target
                self.result['paths'] = []
            except TimeoutException:
                self.result['paths'] = -1
                print("Error: function timed out after 3 seconds")
            finally:
                signal.alarm(0)  # 确保闹钟被取消
            return self.result
        
        if problem==2: #bipartite graph check
            answer = self.is_bipartite(self.G)
            self.result['is_bipartite'] = answer[0]
            self.result['paths'] = answer[1]
            return self.result

        
        # if problem==2: #bipartite graph check
        #     if bipartite.is_bipartite(self.G):
        #         print("The graph is bipartite.")
        #         partitions=nx.bipartite.sets(self.G)
        #         partition1, partition2 = partitions
                
        #         set1=[]
        #         set2=[]
                
        #         for i in partition1:
        #             set1.append(i)
        #         for i in partition2:
        #             set2.append(i)
                
        #         print(partition1)
        #         print(partition2)
                
        #         self.result['Bipartite_graph']="Yes"
        #         self.result['set1']=set1
        #         self.result['set2']=set2 
        #         return self.result
            
        #     else: 
        #         print("The graph is not bipartite.")
        #         color={}
        #         flag=False
        #         for node in self.G.nodes():
        #             if node not in color:
        #                 q=Queue()
        #                 q.put(node)
        #                 color[node]=1
        #                 while q:
        #                     curr=q.get()
        #                     for temp in self.G.neighbors(curr):
        #                         if temp not in color:
        #                             q.put(temp)
        #                             color[temp]=color[curr]^1
        #                         elif color[temp]==color[curr]:
        #                             self.result['Bipartite_graph']="No"
        #                             path_list=[]
        #                             path_list.append(curr)
        #                             path_list.append(temp)
        #                             print(f"controdictary nodes:{curr} and {temp}")
        #                             self.result['Nodes_in_the_same_set']=path_list
        #                             return self.result
                                   
                            
        if problem==3: #toplogical sort
            try:
                #如果运行时间超过3秒，就停止
                # toplogical_orders = nx.all_topological_sorts(self.G)
                toplogical_orders = self.find_topological_sorts_with_limit(self.G, 3)
                toplogical_orders_list = [toplogical_order for toplogical_order in toplogical_orders]
                print(toplogical_orders_list)
                self.result['paths'] = toplogical_orders_list
            except nx.NetworkXNoPath:
                self.result['paths'] = []
            return result


        
        if problem==4: #shortest path
            source = self.source
            target = self.target
            try:
                shortest_paths = nx.all_shortest_paths(self.G, source, target, weight='weight')
                shortest_length = nx.shortest_path_length(self.G, source, target, weight='weight')
                paths_list = [path for path in shortest_paths]
                self.result['source'] = source
                self.result['target'] = target
                if len(paths_list) > 3:
                    paths_list = paths_list[:3]
                self.result['paths'] = paths_list
                self.result['length']=shortest_length
            except nx.NetworkXNoPath:
                self.result['source'] = source
                self.result['target'] = target
                self.result['paths'] = []
            
            return self.result

           
        
        if problem==5: #maximum triangle sum
            
            cycles = nx.simple_cycles(self.G,length_bound=3)
            triangles=[]
            for cycle in cycles:
                if len(cycle)==3:
                    triangles.append(cycle)


            max_weight_sum = float('-inf')
            for triangle in triangles:
                weight_sum = sum(self.G.nodes[triangle[i]]['weight'] for i in range(len(triangle)))
                max_weight_sum = max(max_weight_sum, weight_sum)

            # 保留所有权重之和最大的三元环
            max_weight_triangles = [triangle for triangle in triangles if sum(G.nodes[triangle[i]]['weight'] for i in range(len(triangle))) == max_weight_sum]

            # 输出所有权重之和最大的三元环
            print("All triangles with maximum edge weight sum:")
            # for triangle in max_weight_triangles:
            #     print(triangle)

            if len(max_weight_triangles) > 3:
                max_weight_triangles = max_weight_triangles[:3]
            
            self.result['max_weight_sum']=max_weight_sum  
            self.result['paths']=max_weight_triangles
            
            return self.result


        
        if problem==6: #maximum flow
            source = self.source
            target = self.target
            try:
                flow_value, flow_dict = nx.maximum_flow(G, source, target)
                #flow_results = self.find_multiple_maximum_flows(self.G, source, target, num_solutions=3)
                self.result['flow_value'] = flow_value
                self.result['paths']= flow_dict
                self.result['source'] = source
                self.result['target'] = target
            except nx.NetworkXNoPath:
                self.result['paths'] = []
            
            return self.result

        
        if problem==7: #hamiltonian path
            all_hamiltonian_cycles = []
            all_hamiltonian_cycles= self.hamiltonian_cycles(self.G)                 
            
            if len(all_hamiltonian_cycles) > 3:
                all_hamiltonian_cycles = all_hamiltonian_cycles[:3]

            if len(all_hamiltonian_cycles) == 0:
                self.result['paths'] = []
                self.result['is_hamiltonian'] = False
            else:
                self.result['is_hamiltonian'] = True
                self.result['paths']=all_hamiltonian_cycles    
            print("All Hamiliton_paths:")
            return self.result
            
        if problem==8: #isochromatic
            GM = isomorphism.GraphMatcher(self.G, self.G2)
            self.result['is_isochormatic']=GM.is_isomorphic()
            return self.result

def generate_random_er_graphs(min_nodes, max_nodes, directed=False):
    num_nodes = random.randint(min_nodes, max_nodes)
    p = random.uniform(0, 0.5)  # 随机生成边的概率
    if directed:
        G = nx.erdos_renyi_graph(num_nodes, p, directed=True)
    else:
        G = nx.erdos_renyi_graph(num_nodes, p)
    return G

def generate_dag(min_nodes, max_nodes):
    num_nodes = random.randint(min_nodes, max_nodes)
    p = random.uniform(0, 0.5)  # 随机生成边的概率
    G = nx.DiGraph()
    
    # 添加节点
    G.add_nodes_from(range(num_nodes))
    
    # 添加边，基于随机生成的概率，确保没有环
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u != v and random.random() < p and not nx.has_path(G, v, u):  # 确保没有环
                G.add_edge(u, v)
    
    return G

def generate_weighted_graph(min_nodes, max_nodes):
    num_nodes = random.randint(min_nodes, max_nodes)
    edge_probability = random.uniform(0, 0.3)
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                weight = random.randint(1, 10)  # 随机生成权重
                G.add_edge(i, j, weight=weight)
    return G

def generate_weighted_digraph(min_nodes, max_nodes):
    num_nodes = random.randint(min_nodes, max_nodes)
    edge_probability = random.uniform(0, 0.3)
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < edge_probability:
                capacity = random.randint(1, 10)  # 随机生成权重
                G.add_edge(i, j, capacity=capacity)
    return G

def generate_node_weighted_graph(min_nodes, max_nodes):
    num_nodes = random.randint(min_nodes, max_nodes)
    edge_probability = random.uniform(0, 0.3)
    G = nx.Graph()
    for i in range(num_nodes):
        weight = random.randint(1, 10)  # 随机生成顶点权重
        G.add_node(i, weight=weight)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                G.add_edge(i, j)
    return G
     
   
            
if __name__ == "__main__":
    #problem = 0 #problem 0:cycle_detection 1:Connectivity 2:Bipartite_graph_check 3:Topological_sort 4:Shortest_path 5:Maximum_triangle_sum 6:Maximum_flow 7:Hamilton_path 8:Subgraph_matching
    #随机生成1000个图，每个图的节点在5-30之间

    problem_list = [7]
    for problem in tqdm(problem_list, desc="Processing problems"):
        task_list = ['Cycle_detection', 'Connectivity', 'Bipartite_graph_check', 'Topological_sort', 'Shortest_path', 'Maximum_triangle_sum', 'Maximum_flow', 'Hamilton_path', 'Subgraph_matching']
        edge_list = []
        ans_list = []
        if problem == 5:
            node_list =[]
        while len(edge_list) < 20000:
            for _ in tqdm(range(20000 - len(edge_list)), desc="Generating graphs"):
                if problem in [0, 1]:
                    G = generate_random_er_graphs(5, 25)
                elif problem == 2:
                    # if len(edge_list) %2 == 0:
                    #     G = generate_random_er_graphs(1, 3)
                    # else:
                    G = generate_random_er_graphs(5, 25, directed=True)
                elif problem == 3:
                    G = generate_dag(5, 25)
                elif problem == 4:
                    G = generate_weighted_graph(5, 25)
                elif problem == 5:
                    G = generate_node_weighted_graph(5, 25)
                elif problem == 6:
                    G = generate_weighted_digraph(5, 25)
                elif problem == 7:
                    G = generate_random_er_graphs(5, 10)

                G2 = generate_random_er_graphs(5, 25)
                source=random.randint(0, G.number_of_nodes()-1)
                target = source
                while target == source:
                    target=random.randint(0, G.number_of_nodes()-1)
                
                if problem == 1 and not nx.has_path(G, source, target) and G.number_of_nodes() > 10:
                    continue



                result={}

                Generator=Generation(problem=problem,G=G,G2=G2,source=source,target=target,result=result)
                result=Generator.generate_path()
                # if problem == 2 and result['is_bipartite'] == True and (len(result['paths'][0])  == 0 or len(result['paths'][1]) == 0):
                #     continue

                if problem == 6 and result['flow_value'] == 0:
                    continue

                if problem in [0,1]:
                    if not isinstance(result['paths'],list) and result['paths'] == -1:
                        continue
                    elif len(result['paths']) > 0:
                        print(result)
                        edge_list.append(list(G.edges()))
                        ans_list.append(result)
                    elif len(result['paths']) == 0:
                        if problem == 0:
                            edge_list.append(list(G.edges()))
                            ans_list.append({'paths' :[-1]})
                        if problem == 1:
                            edge_list.append(list(G.edges()))
                            ans_list.append({'source': source, 'target': target, 'paths' :[-1]})
                elif problem == 4:
                    if len(result['paths']) > 0:
                        print(result)
                        edge_list.append([(u, v, G[u][v]['weight']) for u, v in G.edges()])
                        ans_list.append(result)
                elif problem == 5:
                    if len(result['paths']) > 0:
                        print(result)
                        edge_list.append(list(G.edges()))
                        ans_list.append(result)
                        node_list.append([(u, G.nodes[u]['weight']) for u in G.nodes()])
                elif problem == 6:
                    if len(result['paths']) > 0:
                        print(result)
                        edge_list.append([(u, v, G[u][v]['capacity']) for u, v in G.edges()])
                        ans_list.append(result)
                # elif problem == 7:
                #         edge_list.append(list(G.edges()))
                #         ans_list.append(result)
                else:
                    if len(result['paths']) > 0:
                        print(result)
                        edge_list.append(list(G.edges()))
                        ans_list.append(result)

        datas = []
        for i in range(len(edge_list)):
            if problem == 0:
                datas.append({
                    "node_num": G.number_of_nodes(),
                    "edges": edge_list[i],
                    "result": ans_list[i]['paths'],
                    "directed": False,
                    "task": task_list[problem]
                })


            if problem == 1:
                datas.append({
                    "node_num": G.number_of_nodes(),
                    "edges": edge_list[i],
                    "source": ans_list[i]['source'],
                    "target": ans_list[i]['target'],
                    "result": ans_list[i]['paths'],
                    "directed": False,
                    "task": task_list[problem]
                })


            
            if problem == 2:
                datas.append({
                    "node_num": G.number_of_nodes(),
                    "edges": edge_list[i],
                    "result": ans_list[i]['is_bipartite'],
                    "path": ans_list[i]['paths'],
                    "directed": False,
                    "task": task_list[problem]
                })

            if problem == 3:
                datas.append({
                    "node_num": G.number_of_nodes(),
                    "edges": edge_list[i],
                    "result": ans_list[i]['paths'],
                    "directed": True,
                    "task": task_list[problem]
                })


            if problem == 4:
                datas.append({
                    "node_num": G.number_of_nodes(),
                    "edges": edge_list[i],
                    "source": ans_list[i]['source'],
                    "target": ans_list[i]['target'],
                    "result": ans_list[i]['paths'],
                    "length": ans_list[i]['length'],
                    "directed": False,
                    "task": task_list[problem]
                })


            if problem == 5:
                datas.append({
                    "node_num": G.number_of_nodes(),
                    "edges": edge_list[i],
                    "node_weight": node_list[i],
                    "result": ans_list[i]['paths'],
                    "weight": ans_list[i]['max_weight_sum'],
                    "directed": False,
                    "task": task_list[problem]
                })

            if problem == 6:
                datas.append({
                    "node_num": G.number_of_nodes(),
                    "edges": edge_list[i],
                    "source": ans_list[i]['source'],
                    "target": ans_list[i]['target'],
                    "result": ans_list[i]['paths'],
                    "flow_value": ans_list[i]['flow_value'],
                    "directed": True,
                    "task": task_list[problem]
                })
            
            if problem == 7:
                datas.append({
                    "node_num": G.number_of_nodes(),
                    "edges": edge_list[i],
                    "is_hamiltonian": ans_list[i]['is_hamiltonian'],
                    "result": ans_list[i]['paths'],
                    "directed": False,
                    "task": task_list[problem]
                })

        with open(f'{task_list[problem]}.json', 'w') as f:
            json.dump(datas, f)

    # G.add_edges_from([(0,1),(1,2),(2,3),(0,3),(1,4),(2,4)])#G.add_edges_from([(1,2,{'capacity':6}),(1,3,{'capacity':4}),(2,3,{'capacity':1}),(2,4,{'capacity':2}),(2,5,{'capacity':3}),(3,5,{'capacity':2}),(4,6,{'capacity':7}),(5,4,{'capacity':3}),(5,6,{'capacity':2})])
    # G2=nx.Graph()

    


            
            
            
            
        

            
            

        
        
                   
        


                                


                

            
        
            
                



         












