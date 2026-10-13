"""Topological sort and parallel group extraction for CellProfiler dependency graphs.

Uses the existing get_dependency_graph() output to identify which modules
can run in parallel (no data dependencies between them).
"""

from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Any


def topological_sort(
    module_numbers: List[int],
    dependencies: List[Tuple[int, int]],
) -> List[int]:
    """Kahn's algorithm: topological sort of modules given dependency edges.

    Args:
        module_numbers: List of module numbers in the pipeline.
        dependencies: List of (producer_module_num, consumer_module_num) edges,
                      meaning the consumer depends on the producer.

    Returns:
        Modules in topological order (producers before consumers).
        Returns the original order if a cycle is detected.
    """
    # Build adjacency list and in-degree counts
    graph: Dict[int, List[int]] = defaultdict(list)
    in_degree: Dict[int, int] = {m: 0 for m in module_numbers}

    for producer, consumer in dependencies:
        if producer in in_degree and consumer in in_degree:
            graph[producer].append(consumer)
            in_degree[consumer] += 1

    # Start with nodes that have no dependencies
    queue = deque([m for m in module_numbers if in_degree[m] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Cycle detected — fall back to original order
    if len(result) != len(module_numbers):
        return list(module_numbers)

    return result


def get_parallel_groups(
    module_numbers: List[int],
    dependencies: List[Tuple[int, int]],
) -> List[Set[int]]:
    """Group modules into parallel execution levels.

    A level contains all modules whose dependencies have been satisfied
    by previous levels. Modules within the same level can run in parallel.

    Args:
        module_numbers: List of module numbers.
        dependencies: List of (producer, consumer) pairs.

    Returns:
        List of sets, each set containing module numbers that can run
        in parallel at that level.
    """
    # Build adjacency and in-degree
    graph: Dict[int, List[int]] = defaultdict(list)
    in_degree: Dict[int, int] = {m: 0 for m in module_numbers}

    for producer, consumer in dependencies:
        if producer in in_degree and consumer in in_degree:
            graph[producer].append(consumer)
            in_degree[consumer] += 1

    # Track which level each module belongs to
    level: Dict[int, int] = {}
    queue = deque([(m, 0) for m in module_numbers if in_degree[m] == 0])

    while queue:
        node, lvl = queue.popleft()
        level[node] = max(level.get(node, 0), lvl)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append((neighbor, lvl + 1))

    # Group by level
    groups: Dict[int, Set[int]] = defaultdict(set)
    for module_num, lvl in level.items():
        groups[lvl].add(module_num)

    # Modules not in any level are in their own serial level
    for m in module_numbers:
        if m not in level:
            groups[max(groups.keys(), default=0) + 1].add(m)

    return [groups[lvl] for lvl in sorted(groups.keys())]


def analyze_parallelism(
    module_names: Dict[int, str],
    dependencies: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Analyze a pipeline's dependency graph and report parallelism opportunities.

    Args:
        module_names: Dict mapping module_number -> module_name.
        dependencies: List of (producer, consumer) dependency pairs.

    Returns:
        Dict with analysis results including parallel groups, serial depth,
        and estimated speedup.
    """
    module_numbers = list(module_names.keys())
    groups = get_parallel_groups(module_numbers, dependencies)

    n_serial = len(groups)
    n_modules = len(module_numbers)
    max_parallel = max(len(g) for g in groups) if groups else 0

    # Estimate: if all groups were equal size, speedup ≈ n_modules / n_serial
    estimated_speedup = n_modules / n_serial if n_serial > 0 else 1.0

    return {
        "parallel_groups": [
            {m: module_names[m] for m in sorted(g)} for g in groups
        ],
        "serial_depth": n_serial,
        "total_modules": n_modules,
        "max_parallel_width": max_parallel,
        "estimated_speedup": round(estimated_speedup, 2),
        "bottleneck": "None" if max_parallel > 1 else "All modules serially dependent",
    }
