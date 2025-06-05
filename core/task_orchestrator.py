from typing import List, Dict, Any, Set
from datetime import datetime
import logging
from collections import defaultdict
import networkx as nx
from core.utils import suggest_tags
from core.goal_router import GoalRouter
from core.memory import Memory
from core.models import Task

logger = logging.getLogger(__name__)

class TaskOrchestrator:
    def __init__(self, memory: Memory):
        self.memory = memory
        self.router = GoalRouter()
        
    def cluster_tasks(self, tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group related tasks into clusters for batch processing"""
        if not tasks:
            return []
            
        # Build similarity graph
        G = nx.Graph()
        for i, task1 in enumerate(tasks):
            G.add_node(i, task=task1)
            for j, task2 in enumerate(tasks[i+1:], i+1):
                similarity = self._calculate_task_similarity(task1, task2)
                if similarity > 0.5:  # Threshold for considering tasks related
                    G.add_edge(i, j, weight=similarity)
        
        # Extract clusters using community detection
        clusters = list(nx.community.greedy_modularity_communities(G))
        
        # Convert node indices back to tasks
        task_clusters = []
        for cluster in clusters:
            task_cluster = [G.nodes[i]['task'] for i in cluster]
            task_cluster = self._order_cluster(task_cluster)
            task_clusters.append(task_cluster)
            
        return task_clusters
        
    def chain_tasks(self, tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create execution chains of dependent tasks"""
        if not tasks:
            return []
            
        # Build dependency graph - ensure tasks have metadata
        G = nx.DiGraph()
        for task in tasks:
            # Initialize metadata if missing
            if not isinstance(task, dict):
                logger.warning(f"Invalid task format: {task}")
                continue
            
            if 'id' not in task:
                logger.warning(f"Task missing ID: {task}")
                continue
            
            # Add node with safe metadata access
            G.add_node(task['id'], task=task)
        
        # Add dependency edges with safe metadata access
        for task1 in tasks:
            for task2 in tasks:
                if not isinstance(task1, dict) or not isinstance(task2, dict):
                    continue
                
                if 'id' not in task1 or 'id' not in task2:
                    continue
                
                if task1['id'] != task2['id']:
                    if self._is_dependent(task1, task2):
                        G.add_edge(task2['id'], task1['id'])
        
        # Find execution chains (paths in graph)
        chains = []
        start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        for start in start_nodes:
            try:
                paths = nx.single_source_shortest_path(G, start)
                for end in paths:
                    if len(paths[end]) > 1:  # Only include non-trivial paths
                        chain = [G.nodes[n]['task'] for n in paths[end]]
                        chains.append(chain)
            except nx.NetworkXError as e:
                logger.error(f"Error processing path from {start}: {e}")
                continue
        
        return chains
        
    def _calculate_task_similarity(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """Calculate similarity score between two tasks"""
        score = 0.0
        
        # Tag overlap - safely access metadata
        tags1 = set(task1.get('metadata', {}).get('tags', []))
        tags2 = set(task2.get('metadata', {}).get('tags', []))
        tag_overlap = len(tags1.intersection(tags2)) / max(len(tags1.union(tags2)), 1)
        score += tag_overlap * 0.4
        
        # File overlap - safely access metadata
        files1 = set(task1.get('metadata', {}).get('target_files', []))
        files2 = set(task2.get('metadata', {}).get('target_files', []))
        file_overlap = len(files1.intersection(files2)) / max(len(files1.union(files2)), 1)
        score += file_overlap * 0.4
        
        # Goal text similarity (using memory's vector similarity)
        text_similarity = self.memory.calculate_similarity(task1.get('goal', ''), task2.get('goal', ''))
        score += text_similarity * 0.2
        
        return score
        
    def _is_dependent(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> bool:
        """Check if task1 depends on task2"""
        # Check for explicit dependencies - safely access metadata
        if task1.get('metadata', {}).get('depends_on') == task2.get('id'):
            return True
            
        # Check for implicit dependencies based on files and goals
        files1 = set(task1.get('metadata', {}).get('target_files', []))
        files2 = set(task2.get('metadata', {}).get('target_files', []))
        
        if files1.intersection(files2):
            # If they share files, check if one is a prerequisite for the other
            prereq_patterns = [
                (r'fix|repair', r'improve|enhance'),
                (r'add|create', r'update|modify'),
                (r'refactor', r'optimize')
            ]
            
            goal1 = task1.get('goal', '').lower()
            goal2 = task2.get('goal', '').lower()
            
            for prereq, dependent in prereq_patterns:
                if (any(p in goal2 for p in prereq.split('|')) and
                    any(d in goal1 for d in dependent.split('|'))):
                    return True
                    
        return False
        
    def _order_cluster(self, cluster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order tasks within a cluster based on dependencies"""
        G = nx.DiGraph()
        
        # Add all tasks to graph
        for task in cluster:
            G.add_node(task['id'], task=task)
            
        # Add dependency edges
        for task1 in cluster:
            for task2 in cluster:
                if task1['id'] != task2['id'] and self._is_dependent(task1, task2):
                    G.add_edge(task2['id'], task1['id'])
        
        # Topologically sort tasks
        try:
            ordered_ids = list(nx.topological_sort(G))
            return [G.nodes[id]['task'] for id in ordered_ids]
        except nx.NetworkXUnfeasible:
            # If there's a cycle, fall back to original order
            return cluster 

    def fetch_pending_tasks(self, batch: int = 5) -> List[Task]:
        """Fetch a batch of pending tasks from the task queue"""
        return [Task.from_dict(r) for r in self._read_open_tasks()[:batch]] 