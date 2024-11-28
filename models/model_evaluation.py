
from typing import Dict, List
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import ast
import logging
from concurrent.futures import ThreadPoolExecutor

class CodeModelEvaluator:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = {}
        
    def evaluate_code_generation(self, test_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate code generation metrics."""
        self.model.eval()
        metrics = {
            'perplexity': [],
            'code_validity': [],
            'semantic_similarity': []
        }
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                loss = self._compute_loss(outputs, labels)
                
                generated_code = self.model.generate(input_ids)
                decoded_code = self.tokenizer.decode(generated_code[0])
                
                metrics['perplexity'].append(torch.exp(loss).item())
                metrics['code_validity'].append(self._check_code_validity(decoded_code))
                metrics['semantic_similarity'].append(
                    self._compute_semantic_similarity(decoded_code, batch['reference'][0])
                )
                
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _check_code_validity(self, code: str) -> float:
        """Check if generated code is syntactically valid."""
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.0
            
    def _compute_semantic_similarity(self, generated: str, reference: str) -> float:
        """Compute semantic similarity between generated and reference code."""
        try:
            gen_tree = ast.parse(generated)
            ref_tree = ast.parse(reference)
            return self._tree_similarity(gen_tree, ref_tree)
        except:
            return 0.0
            
    
    def _tree_similarity(self, tree1: ast.AST, tree2: ast.AST) -> float:
            """Compute similarity between two AST trees.
            
            Returns:
                float: Similarity score between 0 and 1
            """
            def get_node_type_counts(tree: ast.AST) -> Dict[str, int]:
                """Count occurrence of each node type in the tree."""
                counts = {}
                for node in ast.walk(tree):
                    node_type = type(node).__name__
                    counts[node_type] = counts.get(node_type, 0) + 1
                return counts
            
            def structural_similarity(node1: ast.AST, node2: ast.AST, depth: int = 0) -> float:
                """Compute structural similarity recursively."""
                if type(node1) != type(node2):
                    return 0.0
                    
                # Get fields that are AST nodes or lists of AST nodes
                fields1 = {name: getattr(node1, name) for name in ast.iter_fields(node1)}
                fields2 = {name: getattr(node2, name) for name in ast.iter_fields(node2)}
                
                if not fields1 and not fields2:
                    return 1.0
                    
                similarities = []
                
                for name in set(fields1.keys()) & set(fields2.keys()):
                    f1, f2 = fields1[name], fields2[name]
                    
                    # Handle lists of nodes
                    if isinstance(f1, list) and isinstance(f2, list):
                        if len(f1) == 0 and len(f2) == 0:
                            similarities.append(1.0)
                        elif len(f1) == 0 or len(f2) == 0:
                            similarities.append(0.0)
                        else:
                            # Compare corresponding nodes in lists
                            list_sim = sum(structural_similarity(n1, n2, depth + 1) 
                                         for n1, n2 in zip(f1[:min(len(f1), len(f2))]))
                            similarities.append(list_sim / max(len(f1), len(f2)))
                    
                    # Handle individual nodes
                    elif isinstance(f1, ast.AST) and isinstance(f2, ast.AST):
                        similarities.append(structural_similarity(f1, f2, depth + 1))
                    
                    # Handle primitive values
                    elif f1 == f2:
                        similarities.append(1.0)
                    else:
                        similarities.append(0.0)
                
                return np.mean(similarities) if similarities else 0.0
    
            # Get node type distributions
            counts1 = get_node_type_counts(tree1)
            counts2 = get_node_type_counts(tree2)
            
            # Calculate node type similarity
            all_types = set(counts1.keys()) | set(counts2.keys())
            type_similarity = sum(min(counts1.get(t, 0), counts2.get(t, 0)) 
                                for t in all_types) / sum(max(counts1.get(t, 0), counts2.get(t, 0)) 
                                for t in all_types)
            
            # Calculate structural similarity
            struct_similarity = structural_similarity(tree1, tree2)
            
            # Combine both metrics with weights
            return 0.4 * type_similarity + 0.6 * struct_similarity
        
    def analyze_code_quality(self, code: str) -> Dict[str, float]:
        """Analyze code quality metrics."""
        metrics = {}
        
        try:
            tree = ast.parse(code)
            
            # Complexity metrics
            metrics['cyclomatic_complexity'] = self._compute_complexity(tree)
            metrics['maintainability_index'] = self._compute_maintainability(tree)
            
            # Code style metrics
            metrics['line_length_score'] = self._analyze_line_lengths(code)
            metrics['naming_convention_score'] = self._analyze_naming(tree)
            
        except Exception as e:
            logging.error(f"Error analyzing code: {str(e)}")
            metrics = {k: 0.0 for k in ['cyclomatic_complexity', 'maintainability_index', 
                                      'line_length_score', 'naming_convention_score']}
            
        return metrics