'''
深度优先搜索算法实现
'''
from collections import deque


def depth_first_search(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start)  # 访问节点
    for neighbor in graph[start]:
        if neighbor not in visited:
            depth_first_search(graph, neighbor, visited)
    return visited


# 示例图表示为邻接表
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# 从节点 'A' 开始进行深度优先搜索
depth_first_search(graph, 'A')

'''
实现深度优先搜索查找图中路径
'''


def depth_first_search_path(graph, start, goal, path=None):
    if path is None:
        path = []
    path = path + [start]

    if start == goal:
        return path

    for neighbor in graph[start]:
        if neighbor not in path:
            new_path = depth_first_search_path(graph, neighbor, goal, path)
            if new_path:
                return new_path

    return None


# 示例图表示为邻接表
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# 从节点 'A' 到节点 'F' 查找路径
path = depth_first_search_path(graph, 'A', 'F')
print(f"Path from A to F: {path}")

'''
广度优先搜索算法实现

'''


def bfs(graph, start):
    # 创建一个队列用于存储待处理的节点
    queue = deque([start])
    # 创建一个集合用于存储已访问的节点
    visited = set()

    while queue:
        # 从队列中取出一个节点
        node = queue.popleft()

        if node not in visited:
            # 打印当前节点
            print(node, end=' ')
            # 将节点标记为已访问
            visited.add(node)

            # 将所有未访问的邻居节点添加到队列中
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)


# 示例图（邻接表表示）
graph = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': ['G'],
    'D': [],
    'E': [],
    'F': ['G'],
    'G': []
}

# 从节点 'A' 开始进行广度优先搜索
bfs(graph, 'A')
'''
广度优先搜索实现查找图的路径
'''

from collections import deque


def bfs(graph, start, goal):
    # 创建一个队列来存储待访问的节点及其路径
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        (vertex, path) = queue.popleft()
        if vertex not in visited:
            if vertex == goal:
                return path
            visited.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return None


# 示例图（邻接表表示）
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# 使用广度优先搜索查找从 'A' 到 'F' 的路径
start_node = 'A'
goal_node = 'F'
path = bfs(graph, start_node, goal_node)
print(f"Path from {start_node} to {goal_node}: {path}")

'''
深度优先搜索处理判断是否是二分图
'''


def is_bipartite(graph):
    # 用于记录每个节点的颜色，0 表示未访问，1 和 -1 表示两种不同的颜色
    color = {}

    def dfs(node, c):
        color[node] = c
        for neighbor in graph[node]:
            if neighbor not in color:
                if not dfs(neighbor, -c):
                    return False
            elif color[neighbor] == color[node]:
                return False
        return True

    # 遍历所有节点，处理可能的非连通图
    for node in graph:
        if node not in color:
            if not dfs(node, 1):
                return False

    return True


# 示例图表示为邻接表
graph = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [0, 2]
}

print(is_bipartite(graph))  # 输出: True

'''
深度优先搜索：有向图的可达性
'''


def dfs_reach(graph, start, end):
    # 用于记录每个节点是否被访问过
    visited = set()
    # 用于记录从起始节点到当前节点的路径
    path = []

    def dfs(node):
        # 标记当前节点已被访问
        visited.add(node)
        # 记录当前节点到起始节点的路径
        path.append(node)
        # 如果当前节点是终止节点，则返回路径
        if node == end:
            return path
        # 遍历当前节点的邻居节点
        for neighbor in graph[node]:
            # 如果邻居节点未被访问过，则递归调用
            if neighbor not in visited:
                new_path = dfs(neighbor)
                # 如果递归调用返回路径，则返回路径
                if new_path:
                    return new_path
        # 如果当前节点的所有邻居都已被访问过，则从路径中删除当前节点
        path.pop()
        return None

    # 从起始节点开始递归调用
    return dfs(start)


# 示例图表示为邻接表
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': ['E'],
    'E': ['F'],
    'F': []
}

# 从节点 'A' 到节点 'F' 查找可达路径
path = dfs_reach(graph, 'A', 'F')
print(f"Path from A to F: {path}")

'''
寻找一副有向图的中的有向环
'''


def find_cycle(graph):
    # 用于记录每个节点是否被访问过
    visited = set()
    # 用于记录从起始节点到当前节点的路径
    path = []

    def dfs(node):
        if node in path:  # 检查环
            cycle_start = path.index(node)
            return path[cycle_start:] + [node]
        if node in visited:  # 如果节点已经访问过，返回空
            return []
        visited.add(node)
        path.append(node)  # 将当前节点添加到路径
        for neighbor in graph[node]:  # 遍历邻接节点
            cycle = dfs(neighbor)
            if cycle:  # 如果找到环，返回它
                return cycle
        path.pop()  # 如果当前路径没有找到环，回溯
        return []

    # 遍历所有节点，查找有向环
    for node in graph:
        if node not in visited:
            cycle = dfs(node)
            if cycle:  # 如果找到环，返回它
                return cycle
    return None  # 如果没有找到环，返回 None


# 示例图表示为邻接表
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': ['E'],
    'E': ['F'],
    'F': ['A']
}

# 查找有向环
cycle = find_cycle(graph)
if cycle:
    print(f"Cycle found: {cycle}")
else:
    print("No cycle found")

'''
拓扑排序
'''
from collections import defaultdict


class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # 使用字典列表来存储图
        self.V = vertices  # 图中顶点数量

    # 添加边
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def dfs_util(self, v, visited, stack):
        visited[v] = True
        for i in self.graph[v]:
            if visited[i] == False:
                self.dfs_util(i, visited, stack)

        # 将当前顶点的所有邻居都访问完后，将当前顶点压入栈中
        stack.append(v)

    # 拓扑排序
    def topologicalSort(self):
        # 标记所有顶点为未访问
        visited = [False] * self.V
        stack = []
        # 对所有未访问的顶点调用dfs
        for i in range(self.V):
            if not visited[i]:
                self.dfs_util(i, visited, stack)

        # 由于最后入栈的顶点应排在最前面，最终需要反转栈
        return stack[::-1]


# 示例图
if __name__ == "__main__":
    g = Graph(6)
    g.addEdge(5, 2),
    g.addEdge(5, 0),
    g.addEdge(4, 0),
    g.addEdge(4, 1),
    g.addEdge(2, 3),
    g.addEdge(3, 1)

# 输出拓扑排序
print(g.topologicalSort())  # 输出：[5, 4, 2, 3, 1, 0]

'''
最小生成树 ：  带权重的边的数据类型
'''


class WeightedEdge:
    def __init__(self, u, v, weight):
        """
        初始化带权重的边。

        :param u: 边的起点
        :param v: 边的终点
        :param weight: 边的权重
        """
        self.u = u
        self.v = v
        self.weight = weight

    def __repr__(self):
        """
        返回边的字符串表示。
        """
        return f"({self.u}, {self.v}, {self.weight})"

    def __lt__(self, other):
        """
        比较两条边的权重，用于排序或最小生成树算法中。
        """
        return self.weight < other.weight

    def __eq__(self, other):
        """
        判断两条边是否相等。
        """
        return self.u == other.u and self.v == other.v and self.weight == other.weight

    def __hash__(self):
        """
        为了能够在集合或字典中使用，提供哈希函数。
        """
        return hash((self.u, self.v, self.weight))


'''
最小生成树：
  加权无向图的数据类型
'''


class WeightedUndirectedGraph:
    def __init__(self):
        """
        初始化一个空的加权无向图。
        """
        self.adj_list = {}

    def add_edge(self, u, v, weight):
        """
        添加一条带权重的无向边。

        :param u: 边的一个顶点
        :param v: 边的另一个顶点
        :param weight: 边的权重
        """
        edge = WeightedEdge(u, v, weight)

        if u not in self.adj_list:
            self.adj_list[u] = []
        if v not in self.adj_list:
            self.adj_list[v] = []

        # 因为是无向图，所以我们需要在两个方向都添加这条边
        self.adj_list[u].append(edge)
        self.adj_list[v].append(WeightedEdge(v, u, weight))  # 这里是反向的边，保持无向图的对称性

    def get_edges(self):
        """
        返回图中的所有边。
        """
        edges = set()
        for u in self.adj_list:
            for edge in self.adj_list[u]:
                # 因为是无向图，所以我们只添加一个方向的边，避免重复
                if (edge.u, edge.v, edge.weight) not in edges and (edge.v, edge.u, edge.weight) not in edges:
                    edges.add((edge.u, edge.v, edge.weight))
        return list(edges)

    def get_vertices(self):
        """
        返回图中的所有顶点。
        """
        return list(self.adj_list.keys())

    def __repr__(self):
        """
        返回图的字符串表示，方便打印。
        """
        result = ""
        for u in self.adj_list:
            result += f"{u}: {[str(edge) for edge in self.adj_list[u]]}\n"
        return result


# 创建一个图
graph = WeightedUndirectedGraph()

# 添加一些边
graph.add_edge('A', 'B', 10)
graph.add_edge('A', 'C', 5)
graph.add_edge('B', 'C', 2)
graph.add_edge('B', 'D', 7)

# 打印图的邻接表
print(graph)

# 获取所有顶点
vertices = graph.get_vertices()
print("Vertices:", vertices)

# 获取所有边
edges = graph.get_edges()
print("Edges:", edges)

'''
    最小生成树： prim算法 （延时版本）
'''
import heapq


class LazyPrimMST:
    def __init__(self, graph):
        """
        初始化 Prim 算法的延时实现。

        :param graph: 加权无向图的实例
        """
        self.graph = graph
        self.mst = []  # 最小生成树中的边
        self.visited = set()  # 已经访问的顶点
        self.min_heap = []  # 最小优先队列（最小堆）
        self.total_weight = 0  # MST的总权重

        # 开始算法，假设从图中的第一个顶点开始
        start_vertex = next(iter(graph.get_vertices()))  # 选择任意一个顶点作为起点
        self.visit(start_vertex)

        while self.min_heap and len(self.mst) < len(graph.get_vertices()) - 1:
            weight, u, v = heapq.heappop(self.min_heap)
            if v in self.visited:
                continue

            self.mst.append((u, v, weight))
            self.total_weight += weight
            self.visit(v)

    def visit(self, vertex):
        """
        标记顶点并将与之关联的边放入优先队列。

        :param vertex: 当前访问的顶点
        """
        self.visited.add(vertex)
        for edge in self.graph.adj_list[vertex]:
            if edge.v not in self.visited:
                heapq.heappush(self.min_heap, (edge.weight, edge.u, edge.v))

    def get_mst(self):
        """
        返回最小生成树的边集。
        """
        return self.mst

    def get_total_weight(self):
        """
        返回最小生成树的总权重。
        """
        return self.total_weight

    def __repr__(self):
        """
        返回 MST 的字符串表示。
        """
        return f"MST Edges: {self.mst}\nTotal Weight: {self.total_weight}"


# 创建一个图
graph = WeightedUndirectedGraph()

# 添加一些边
graph.add_edge('A', 'B', 10)
graph.add_edge('A', 'C', 5)
graph.add_edge('B', 'C', 2)
graph.add_edge('B', 'D', 7)
graph.add_edge('C', 'D', 3)
graph.add_edge('C', 'E', 8)
graph.add_edge('D', 'E', 1)

# 使用 Lazy Prim 算法计算最小生成树
prim_mst = LazyPrimMST(graph)

# 打印最小生成树的边和总权重
print(prim_mst)

'''
最小生成树的prim算法： 即时版本

'''

import heapq


class EagerPrimMST:
    def __init__(self, graph):
        """
        初始化 Prim 算法的即时版本。

        :param graph: 加权无向图的实例
        """
        self.graph = graph
        self.mst = []  # 最小生成树中的边
        self.total_weight = 0  # MST的总权重
        self.edge_to = {}  # 存储连接每个顶点的最小权重边
        self.dist_to = {}  # 存储每个顶点的最小权重
        self.pq = []  # 最小优先队列（最小堆）
        self.visited = set()  # 记录已经访问的顶点

        # 初始化 dist_to 字典，所有距离初始化为正无穷大
        for vertex in graph.get_vertices():
            self.dist_to[vertex] = float('inf')

        # 假设从图中的第一个顶点开始
        start_vertex = next(iter(graph.get_vertices()))
        self.dist_to[start_vertex] = 0
        heapq.heappush(self.pq, (0, start_vertex))

        # 开始构建 MST
        while self.pq:
            _, v = heapq.heappop(self.pq)
            if v in self.visited:
                continue
            self.visit(v)

    def visit(self, v):
        """
        标记顶点v并更新与之连接的所有未访问顶点的权重。

        :param v: 当前访问的顶点
        """
        self.visited.add(v)
        for edge in self.graph.adj_list[v]:
            w = edge.v
            if w in self.visited:
                continue
            if edge.weight < self.dist_to[w]:
                # 更新连接到w的最小权重边
                self.edge_to[w] = edge
                self.dist_to[w] = edge.weight
                heapq.heappush(self.pq, (edge.weight, w))

        # 将与v相连的边加入 MST
        if v in self.edge_to:
            self.mst.append((self.edge_to[v].u, self.edge_to[v].v, self.edge_to[v].weight))
            self.total_weight += self.edge_to[v].weight

    def get_mst(self):
        """
        返回最小生成树的边集。
        """
        return self.mst

    def get_total_weight(self):
        """
        返回最小生成树的总权重。
        """
        return self.total_weight

    def __repr__(self):
        """
        返回 MST 的字符串表示。
        """
        return f"MST Edges: {self.mst}\nTotal Weight: {self.total_weight}"


# 创建一个图
graph = WeightedUndirectedGraph()

# 添加一些边
graph.add_edge('A', 'B', 10)
graph.add_edge('A', 'C', 5)
graph.add_edge('B', 'C', 2)
graph.add_edge('B', 'D', 7)
graph.add_edge('C', 'D', 3)
graph.add_edge('C', 'E', 8)
graph.add_edge('D', 'E', 1)

# 使用 Eager Prim 算法计算最小生成树
eager_prim_mst = EagerPrimMST(graph)

# 打印最小生成树的边和总权重
print(eager_prim_mst)

'''
最小生成树： kruskal算法 
'''


class UnionFind:
    def __init__(self, vertices):
        """
        初始化并查集数据结构。

        :param vertices: 图中的所有顶点
        """
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, vertex):
        """
        查找顶点的根，使用路径压缩优化。

        :param vertex: 顶点
        :return: 顶点所属的根
        """
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, vertex1, vertex2):
        """
        将两个顶点的连通分量合并，使用按秩合并优化。

        :param vertex1: 顶点1
        :param vertex2: 顶点2
        """
        root1 = self.find(vertex1)
        root2 = self.find(vertex2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


class KruskalMST:
    def __init__(self, graph):
        """
        初始化 Kruskal 算法。

        :param graph: 加权无向图的实例
        """
        self.graph = graph
        self.mst = []  # 最小生成树中的边
        self.total_weight = 0  # 最小生成树的总权重

        # 获取图中的所有边，并按权重排序
        edges = sorted(graph.get_edges(), key=lambda x: x[2])

        # 初始化并查集
        union_find = UnionFind(graph.get_vertices())

        # 遍历所有边，并尝试将其加入 MST
        for u, v, weight in edges:
            # 检查 u 和 v 是否属于不同的连通分量
            if union_find.find(u) != union_find.find(v):
                # 如果 u 和 v 不在同一个连通分量中，则加入这条边到 MST 中
                self.mst.append((u, v, weight))
                self.total_weight += weight
                union_find.union(u, v)

                # 如果 MST 已经有 V-1 条边，算法可以结束
                if len(self.mst) == len(graph.get_vertices()) - 1:
                    break

    def get_mst(self):
        """
        返回最小生成树的边集。
        """
        return self.mst

    def get_total_weight(self):
        """
        返回最小生成树的总权重。
        """
        return self.total_weight

    def __repr__(self):
        """
        返回 MST 的字符串表示。
        """
        return f"MST Edges: {self.mst}\nTotal Weight: {self.total_weight}"


# 创建一个图
graph = WeightedUndirectedGraph()

# 添加一些边
graph.add_edge('A', 'B', 10)
graph.add_edge('A', 'C', 5)
graph.add_edge('B', 'C', 2)
graph.add_edge('B', 'D', 7)
graph.add_edge('C', 'D', 3)
graph.add_edge('C', 'E', 8)
graph.add_edge('D', 'E', 1)

# 使用 Kruskal 算法计算最小生成树
kruskal_mst = KruskalMST(graph)

# 打印最小生成树的边和总权重
print(kruskal_mst)

'''
 最短路径： 加权有向边的数据类型
 
'''


class DirectedEdge:
    def __init__(self, v, w, weight):
        """
        Initializes a directed edge from vertex v to vertex w with the given weight.
        :param v: The starting vertex of the directed edge.
        :param w: The ending vertex of the directed edge.
        :param weight: The weight of the directed edge.
        """
        self._v = v
        self._w = w
        self._weight = weight

    def from_vertex(self):
        """
        Returns the starting vertex of the directed edge.
        :return: The starting vertex (v).
        """
        return self._v

    def to_vertex(self):
        """
        Returns the ending vertex of the directed edge.
        :return: The ending vertex (w).
        """
        return self._w

    def weight(self):
        """
        Returns the weight of the directed edge.
        :return: The weight of the edge.
        """
        return self._weight

    def __str__(self):
        """
        Returns a string representation of the directed edge.
        :return: A string representing the edge in the form "v -> w with weight".
        """
        return f"{self._v} -> {self._w} with weight {self._weight}"


'''
 最短路径： 加权有向图的数据类型
 
'''


class EdgeWeightedDigraph:
    def __init__(self, V):
        """
        初始化一个带权重的有向图，包含 V 个顶点。
        :param V: 图中的顶点数。
        """
        self._V = V  # 顶点数
        self._E = 0  # 边数
        self._adj = [[] for _ in range(V)]  # 邻接表

    def V(self):
        """
        返回图中的顶点数。
        :return: 顶点数 (V)。
        """
        return self._V

    def E(self):
        """
        返回图中的边数。
        :return: 边数 (E)。
        """
        return self._E

    def add_edge(self, edge):
        """
        向图中添加一条有向边。
        :param edge: 要添加的有向边。
        """
        v = edge.from_vertex()
        self._adj[v].append(edge)
        self._E += 1

    def adj(self, v):
        """
        返回与顶点 v 相邻的边。
        :param v: 顶点 v。
        :return: 与顶点 v 相邻的有向边列表。
        """
        return self._adj[v]

    def edges(self):
        """
        返回图中的所有边。
        :return: 图中所有边的列表。
        """
        all_edges = []
        for v in range(self._V):
            all_edges.extend(self._adj[v])
        return all_edges

    def __str__(self):
        """
        返回图的字符串表示形式。
        :return: 显示所有顶点及其相邻边的字符串表示形式。
        """
        result = []
        for v in range(self._V):
            result.append(f"{v}: " + ", ".join(str(e) for e in self._adj[v]))
        return "\n".join(result)


'''
    最短路径 ： 无环加权有向图的最短路径算法
'''

from collections import defaultdict, deque


# 定义图的类
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    # 添加边到图中
    def add_edge(self, u, v, weight):
        self.graph[u].append((v, weight))

    # 帮助函数：拓扑排序
    def topological_sort_util(self, v, visited, stack):
        visited[v] = True
        for i, weight in self.graph[v]:
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)
        stack.append(v)

    # 执行拓扑排序
    def topological_sort(self):
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)

        return stack[::-1]  # 返回拓扑排序结果

    # 计算无环加权有向图的最短路径
    def shortest_path(self, src):
        stack = self.topological_sort()
        dist = [float('inf')] * self.V
        dist[src] = 0

        while stack:
            i = stack.pop()

            if dist[i] != float('inf'):
                for node, weight in self.graph[i]:
                    if dist[node] > dist[i] + weight:
                        dist[node] = dist[i] + weight

        return dist


# 示例用法
g = Graph(6)
g.add_edge(0, 1, 5)
g.add_edge(0, 2, 3)
g.add_edge(1, 3, 6)
g.add_edge(1, 2, 2)
g.add_edge(2, 4, 4)
g.add_edge(2, 5, 2)
g.add_edge(2, 3, 7)
g.add_edge(3, 4, -1)
g.add_edge(4, 5, -2)

source = 1
distances = g.shortest_path(source)

print(f"从节点 {source} 到其他节点的最短路径距离:")
for i in range(len(distances)):
    print(f"到节点 {i} 的距离: {distances[i]}")

'''
    最短路径：  基于队列的Bellman-Ford算法
    
'''

from collections import deque, defaultdict


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def add_edge(self, u, v, weight):
        self.graph[u].append((v, weight))

    def bellman_ford_queue(self, src):
        # 初始化距离为正无穷
        dist = [float('inf')] * self.V
        dist[src] = 0

        # 队列初始化，包含源节点
        queue = deque([src])
        in_queue = [False] * self.V
        in_queue[src] = True

        # 记录每个节点被松弛的次数，用于检测负环
        count = [0] * self.V

        while queue:
            u = queue.popleft()
            in_queue[u] = False

            # 遍历u的所有邻接边
            for v, weight in self.graph[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
                        count[v] += 1

                        # 如果某个顶点被松弛超过V次，说明存在负权重环路
                        if count[v] > self.V - 1:
                            print("图中包含负权重环路")
                            return None

        return dist


# 示例用法
g = Graph(5)
g.add_edge(0, 1, -1)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 3)
g.add_edge(1, 3, 2)
g.add_edge(1, 4, 2)
g.add_edge(3, 2, 5)
g.add_edge(3, 1, 1)
g.add_edge(4, 3, -3)

source = 0
distances = g.bellman_ford_queue(source)

if distances:
    print(f"从节点 {source} 到其他节点的最短路径距离:")
    for i in range(len(distances)):
        print(f"到节点 {i} 的距离: {distances[i]}")

'''
    最短路径： 货币兑换中的套汇问题
    
'''

import math
from collections import defaultdict


class CurrencyGraph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, rate):
        # 将汇率取对数并取负值
        weight = -math.log(rate)
        self.graph.append((u, v, weight))

    def bellman_ford_arbitrage(self, src):
        dist = [float('inf')] * self.V
        dist[src] = 0

        # 进行 V-1 次松弛操作
        for _ in range(self.V - 1):
            for u, v, weight in self.graph:
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight

        # 第 V 次松弛操作，用于检测负权重环路
        for u, v, weight in self.graph:
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                print("存在套汇机会！")
                return True

        print("不存在套汇机会。")
        return False


# 示例用法
currencies = ["USD", "EUR", "GBP", "JPY"]
currency_map = {currency: idx for idx, currency in enumerate(currencies)}

g = CurrencyGraph(len(currencies))
g.add_edge(currency_map["USD"], currency_map["EUR"], 0.9)
g.add_edge(currency_map["EUR"], currency_map["GBP"], 0.8)
g.add_edge(currency_map["GBP"], currency_map["JPY"], 150.0)
g.add_edge(currency_map["JPY"], currency_map["USD"], 0.007)

source = currency_map["USD"]
g.bellman_ford_arbitrage(source)
