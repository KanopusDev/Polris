import requests
from typing import List, Dict, Union
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import load_dataset
import json

class CodeDataLoader:
    def __init__(self, token: str = None, languages: List[str] = ['python']):
        self.headers = {'Authorization': f'token {token}'} if token else {}
        self.languages = languages
        self.base_url = "https://api.github.com"
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, sources: List[str] = ['github', 'codeparrot', 'codenet']) -> List[str]:
        """Load code data from multiple sources."""
        all_code_data = []
        
        for source in sources:
            try:
                if source == 'github':
                    data = self._load_github_data()
                elif source == 'codeparrot':
                    data = self._load_codeparrot_data()
                elif source == 'codenet':
                    data = self._load_codenet_data()
                else:
                    self.logger.warning(f"Unknown source: {source}")
                    continue
                
                all_code_data.extend(data)
                self.logger.info(f"Loaded {len(data)} samples from {source}")
                
            except Exception as e:
                self.logger.error(f"Error loading data from {source}: {str(e)}")
                continue
        
        if not all_code_data:
            raise ValueError("No data could be loaded from any source")
        
        self.logger.info(f"Total samples collected: {len(all_code_data)}")
        return all_code_data

    def _load_github_data(self) -> List[str]:
        """Load data from GitHub."""
        repos = self.fetch_repositories()
        return self.fetch_code_content(repos)
        
    def _load_codeparrot_data(self, subset_size: int = 5000) -> List[str]:
        """Load data from CodeParrot dataset with fallback."""
        try:
            # Try loading from HuggingFace hub first
            dataset = load_dataset(
                "codeparrot/codeparrot-clean-train",
                split="train",
                streaming=True
            )
            
            code_samples = []
            for sample in dataset.take(subset_size):
                if self._validate_code_quality(sample['content']):
                    code_samples.append(sample['content'])
                    
            if code_samples:
                return code_samples
                
        except Exception as e:
            self.logger.warning(f"Error loading from HuggingFace hub: {str(e)}")
            
        # Fallback to local dataset if available
        try:
            local_path = "data/codeparrot_sample.jsonl"
            if Path(local_path).exists():
                with open(local_path, 'r') as f:
                    samples = [json.loads(line.strip())['content'] 
                             for line in f.readlines()[:subset_size]]
                return [s for s in samples if self._validate_code_quality(s)]
        except Exception as e:
            self.logger.warning(f"Error loading from local file: {str(e)}")
            
        return []

    def _load_codenet_data(self, subset_size: int = 5000) -> List[str]:
        """Load data from CodeNet with fallback options."""
        try:
            # Try alternative dataset sources
            alternative_datasets = [
                "codenet/python-algorithms",
                "codenet/java-algorithms",
                "microsoft/CodeXGLUE"
            ]
            
            for dataset_name in alternative_datasets:
                try:
                    dataset = load_dataset(dataset_name, split="train")
                    if dataset:
                        code_samples = []
                        for sample in dataset:
                            if 'code' in sample and self._validate_code_quality(sample['code']):
                                code_samples.append(sample['code'])
                        if code_samples:
                            return code_samples[:subset_size]
                except:
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error loading CodeNet alternatives: {str(e)}")
            
        return []

    def _validate_code_quality(self, code: str) -> bool:
        """Validate code quality before including in dataset."""
        try:
            if len(code.strip()) < 50:  # Skip very short snippets
                return False
                
            # Check for common code quality indicators
            quality_indicators = [
                'def ',  # Has function definitions
                'class ',  # Has class definitions
                '"""',  # Has docstrings
                'import ',  # Has imports
                'return '  # Has return statements
            ]
            
            if not any(indicator in code for indicator in quality_indicators):
                return False
                
            # Check for proper indentation
            lines = code.split('\n')
            if not any(line.startswith('    ') for line in lines):
                return False
                
            # Additional code quality checks can be added here
                
            return True
            
        except Exception:
            return False

    def fetch_repositories(self, min_stars: int = 1000) -> List[Dict]:
        """Fetch high-quality repositories based on strict criteria."""
        repos = []
        for lang in self.languages:
            # Fix query syntax by using proper GitHub search syntax
            query = f"language:{lang} stars:>={min_stars} is:public archived:false fork:false"
            
            try:
                headers = {
                    **self.headers,
                    'Accept': 'application/vnd.github.v3+json'
                }
                
                response = requests.get(
                    f"{self.base_url}/search/repositories",
                    params={
                        'q': query,
                        'sort': 'stars',
                        'order': 'desc',
                        'per_page': 100
                    },
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                if 'items' in data:
                    # Relax criteria to ensure we get some results
                    quality_repos = [
                        repo for repo in data['items']
                        if (repo.get('watchers_count', 0) >= 100 and  # Reduced from 500
                            not repo.get('fork', True) and
                            repo.get('size', 0) > 100)  # Reduced from 1000
                    ]
                    repos.extend(quality_repos)
                    
                self.logger.info(f"Found {len(quality_repos)} quality repositories for {lang}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching repositories for {lang}: {str(e)}")
                continue
                
        if not repos:
            self.logger.warning("No repositories found with strict criteria, falling back to basic search")
            return self._fallback_repository_search()
            
        return repos

    def _fallback_repository_search(self) -> List[Dict]:
        """Fallback method for repository search with relaxed criteria."""
        repos = []
        for lang in self.languages:
            try:
                # Simplified query with minimal criteria
                query = f"language:{lang} stars:>50"
                
                response = requests.get(
                    f"{self.base_url}/search/repositories",
                    params={'q': query, 'sort': 'stars', 'per_page': 50},
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()
                
                if 'items' in data:
                    repos.extend(data['items'])
                    
            except Exception as e:
                self.logger.error(f"Error in fallback repository search: {str(e)}")
                continue
                
        return repos

    def fetch_code_content(self, repos: List[Dict]) -> List[str]:
        code_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for repo in repos:
                futures.append(
                    executor.submit(self._fetch_repo_contents, repo['full_name'])
                )
            
            for future in tqdm(futures, desc="Fetching code content"):
                result = future.result()
                if result:  # Only extend if we got data
                    code_data.extend(result)
                    
        if not code_data:
            raise ValueError("No code content found in any repository")
        
        logging.info(f"Successfully fetched {len(code_data)} code files")
        return code_data

    def _fetch_repo_contents(self, repo_name: str) -> List[str]:
        contents_url = f"{self.base_url}/repos/{repo_name}/contents"
        try:
            response = requests.get(contents_url, headers=self.headers)
            files = response.json()
            return [self._get_file_content(f['download_url']) 
                   for f in files if self._is_valid_file(f)]
        except Exception as e:
            logging.error(f"Error fetching {repo_name}: {str(e)}")
            return []

    def _is_valid_file(self, file_info: Dict) -> bool:
        return (file_info['type'] == 'file' and 
                any(file_info['name'].endswith(f'.{lang}') 
                    for lang in self.languages))

    def _get_file_content(self, url: str) -> str:
        try:
            response = requests.get(url)
            return response.text
        except:
            return ""