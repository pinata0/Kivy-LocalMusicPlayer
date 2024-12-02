from collections import defaultdict
from typing import List

# Query 클래스: 쿼리 정보를 저장하는 클래스
class Query:
    def __init__(self, t: int, p: int, s: int):
        self.t = t  # 쿼리의 시간 t
        self.p = p  # 쿼리 대상의 사람 번호 p
        self.s = s  # 쿼리의 값 s (예: 음악의 인기도나 점수 등)

    # 쿼리의 시간 t를 기준으로 오름차순 정렬을 위해 __lt__ 메서드 정의
    def __lt__(self, other):
        return self.t < other.t

# DFS 함수: 트리 탐색을 수행하고 각 노드의 in_time, out_time, 서브트리 크기(sz) 계산
def dfs(cur: int, adj: List[List[int]], visited: List[bool], in_time: List[int], out_time: List[int], sz: List[int], cnt: List[int]) -> int:
    visited[cur] = True  # 현재 노드를 방문 처리
    cnt[0] += 1  # 방문한 노드 수 증가
    in_time[cur] = cnt[0]  # in_time 기록 (탐색 시작 시간)
    sz[cur] = 1  # 현재 노드의 서브트리 크기 초기화
    for nxt in adj[cur]:  # 현재 노드의 모든 인접 노드 탐색
        if not visited[nxt]:
            sz[cur] += dfs(nxt, adj, visited, in_time, out_time, sz, cnt)  # 서브트리 크기 계산
    out_time[cur] = cnt[0]  # out_time 기록 (탐색 종료 시간)
    return sz[cur]

# 트리의 값 업데이트 함수 (Fenwick Tree의 update)
def update(tree: List[int], x: int, v: int, n: int):
    while x <= n:
        tree[x] += v  # x번째 위치에 v를 더함
        x += x & -x  # 다음 인덱스로 이동

# 구간 업데이트 함수 (Fenwick Tree의 range update)
def range_update(tree: List[int], l: int, r: int, v: int, n: int):
    update(tree, l, v, n)  # 구간 [l, r]에 대해 v만큼 업데이트
    update(tree, r + 1, -v, n)  # r+1 위치부터 -v 업데이트

# 포인트 쿼리 함수 (Fenwick Tree의 point query)
def point_query(tree: List[int], x: int) -> int:
    ret = 0
    while x:
        ret += tree[x]  # x번째 위치까지의 값을 더함
        x -= x & -x  # 부모 노드로 이동
    return ret

# 알고리즘 메인 함수
def algorithm(n: int, k: int, j: int, edges: List[List[int]], singers: List[int], queries: List[Query]) -> List[int]:
    adj = [[] for _ in range(n + 1)]  # 그래프의 인접 리스트 초기화
    # 그래프의 간선 정보 읽어들이기
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)  # 무방향 그래프이므로 양쪽에 간선 추가
    
    # dfs를 위한 배열 초기화
    in_time = [0] * (n + 1)
    out_time = [0] * (n + 1)
    sz = [0] * (n + 1)
    visited = [False] * (n + 1)
    cnt = [0]

    # DFS로 트리 탐색하고, in_time, out_time, 서브트리 크기(sz) 계산
    dfs(1, adj, visited, in_time, out_time, sz, cnt)

    # 각 가수의 노래 수 및 가수별 노래 인덱스를 저장할 자료구조
    nums = [0] * (n + 1)
    songs_by_singer = defaultdict(list)
    for i in range(1, n + 1):
        nums[singers[i - 1]] += 1  # 각 가수별 노래 수 업데이트
        songs_by_singer[singers[i - 1]].append(i)  # 가수별 노래 번호 저장

    # 구간에 대한 lower, upper bound를 기록할 배열
    lo = [0] * (n + 1)
    hi = [k + 1] * (n + 1)
    ans = [0] * (n + 1)  # 각 가수에 대한 답을 기록할 배열

    # 쿼리를 시간 순으로 정렬
    queries.sort()

    # 쿼리를 처리하는 메인 루프
    while True:
        flag = True  # 종료 조건 체크
        qry_list = defaultdict(list)
        # 각 가수에 대해 현재 가능한 구간을 찾기
        for i in range(1, n + 1):
            if nums[i] == 0 or lo[i] + 1 == hi[i]:
                continue
            flag = False  # 아직 가능한 구간이 있다는 의미로 flag = False
            mid = (lo[i] + hi[i]) >> 1  # 중간값 계산
            qry_list[mid].append(i)  # mid값에 해당하는 가수들을 저장

        if flag:
            break  # 종료 조건 만족 시 종료

        tree = [0] * (n + 1)  # Fenwick Tree 초기화
        # 각 쿼리를 처리
        for i in range(1, k + 1):
            t, p, s = queries[i - 1].t, queries[i - 1].p, queries[i - 1].s
            # 트리에 구간 업데이트
            range_update(tree, in_time[p], out_time[p], s // sz[p], n)

            # 현재 중간값에 대해 쿼리 처리
            for singer in qry_list[i]:
                total = 0
                # 가수의 노래에 대해 트리 쿼리
                for song in songs_by_singer[singer]:
                    total += point_query(tree, in_time[song])

                # 가수의 노래 점수가 조건을 만족하면 upper bound 갱신
                if total > nums[singer] * j:
                    hi[singer] = i
                    ans[singer] = t
                else:
                    lo[singer] = i

    # 각 가수의 결과를 반환
    result = []
    for i in range(1, n + 1):
        result.append(ans[singers[i - 1]] if ans[singers[i - 1]] else -1)
    
    return result


# 입력 형태
n = 5  # 곡 수 (노드 개수)
k = 3  # 데이터의 수 (쿼리 개수)
j = 20  # 목표 점수
edges = [(1, 2), (1, 3), (2, 4), (2, 5)]  # 그래프의 간선들
singers = [1, 1, 3, 1, 2]  # 가수 번호
queries = [Query(1, 1, 90), Query(2, 2, 100), Query(3, 1, 30)]

# import sys
# input = sys.stdin.readline
# n, k, j = map(int, input().split())
# edges = [tuple(map(int, input().split())) for _ in range(n - 1)]
# singers = list(map(int, input().split()))
# queries = [Query(*map(int, input().split())) for _ in range(k)]

for i in algorithm(n, k, j, edges, singers, queries): 
    print(i)
