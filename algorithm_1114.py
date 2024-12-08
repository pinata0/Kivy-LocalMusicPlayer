from collections import defaultdict
from typing import List

class Query:
    def __init__(self, t: int, p: int, s: int):
        self.t = t
        self.p = p
        self.s = s

    def __lt__(self, other):
        return self.t < other.t

def dfs(cur: int, adj: List[List[int]], in_time: List[int], out_time: List[int], sz: List[int], cnt: List[int]) -> int:
    cnt[0] += 1
    in_time[cur] = cnt[0]
    sz[cur] = 1
    for nxt in adj[cur]:
        sz[cur] += dfs(nxt, adj, in_time, out_time, sz, cnt)
    out_time[cur] = cnt[0]
    return sz[cur]

def update(tree: List[int], x: int, v: int, n: int):
    while x <= n:
        tree[x] += v
        x += x & -x

def range_update(tree: List[int], l: int, r: int, v: int, n: int):
    update(tree, l, v, n)
    update(tree, r + 1, -v, n)

def point_query(tree: List[int], x: int) -> int:
    ret = 0
    while x:
        ret += tree[x]
        x -= x & -x
    return ret

def algorithm(n: int, k: int, j: int, parents: List[int], singers: List[int], queries: List[Query]) -> List[int]:
    adj = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        adj[parents[i - 2]].append(i)
    
    in_time = [0] * (n + 1)
    out_time = [0] * (n + 1)
    sz = [0] * (n + 1)
    cnt = [0]
    
    dfs(1, adj, in_time, out_time, sz, cnt)

    nums = [0] * (n + 1)
    songs_by_singer = defaultdict(list)
    for i in range(1, n + 1):
        nums[singers[i - 1]] += 1
        songs_by_singer[singers[i - 1]].append(i)

    lo = [0] * (n + 1)
    hi = [k + 1] * (n + 1)
    ans = [0] * (n + 1)

    queries.sort()

    while True:
        flag = True
        qry_list = defaultdict(list)
        for i in range(1, n + 1):
            if nums[i] == 0 or lo[i] + 1 == hi[i]:
                continue
            flag = False
            mid = (lo[i] + hi[i]) >> 1
            qry_list[mid].append(i)

        if flag:
            break

        tree = [0] * (n + 1)
        for i in range(1, k + 1):
            t, p, s = queries[i - 1].t, queries[i - 1].p, queries[i - 1].s
            range_update(tree, in_time[p], out_time[p], s // sz[p], n)

            for singer in qry_list[i]:
                total = 0
                for song in songs_by_singer[singer]:
                    total += point_query(tree, in_time[song])

                if total > nums[singer] * j:
                    hi[singer] = i
                    ans[singer] = t
                else:
                    lo[singer] = i

    result = []
    for i in range(1, n + 1):
        result.append(ans[singers[i - 1]] if ans[singers[i - 1]] else -1)
    
    return result

# n = 5  # 곡 수 (노드 개수)
# k = 3  # 데이터의 수 (쿼리 개수)
# j = 52  # 목표 점수
# parents = [1, 1, 2, 2]  # 부모 노드 되는 곡 번호
# singers = [1, 1, 3, 1, 2]  # 가수 번호
# queries = [Query(1, 1, 90), Query(2, 2, 100), Query(3, 1, 30)]

import sys
input = sys.stdin.readline
n, k, j = map(int, input().split())
parents = list(map(int, input().split()))
singers = list(map(int, input().split()))
queries = [Query(*map(int, input().split())) for _ in range(k)]

for i in algorithm(n, k, j, parents, singers, queries): print(i)
