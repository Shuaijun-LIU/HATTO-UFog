"""Graph utilities for waypoint routing."""
from __future__ import annotations

from typing import List, Optional
import heapq
import math


class GraphRoutingError(RuntimeError):
    pass


def shortest_path(nodes: List[tuple], edges: List[List[int]], start: int, goal: int) -> Optional[List[int]]:
    """Dijkstra shortest path with Euclidean edge cost."""
    if start == goal:
        return [start]
    n = len(nodes)
    dist = [math.inf] * n
    prev = [-1] * n
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            break
        if d > dist[u]:
            continue
        for v in edges[u]:
            du = nodes[u]
            dv = nodes[v]
            w = math.sqrt((du[0] - dv[0]) ** 2 + (du[1] - dv[1]) ** 2 + (du[2] - dv[2]) ** 2)
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if prev[goal] == -1:
        return None
    path = [goal]
    cur = goal
    while cur != start and cur != -1:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path
