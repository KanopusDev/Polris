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
        """Validate code quality with stricter criteria."""
        try:
            if not isinstance(code, str) or len(code.strip()) < 100:  # Increased minimum length
                return False
            
            lines = code.split('\n')
            if len(lines) < 5:  # Require at least 5 lines
                return False
            
            # Check for common code quality indicators
            quality_indicators = [
                'def ',      # Has function definitions
                'class ',    # Has class definitions
                'import ',   # Has imports
                '    ',      # Has proper indentation
                '"""',       # Has docstrings
                'return ',   # Has return statements
                '(',         # Has function calls or definitions
                ':'         # Has control structures
            ]
            
            # Require at least 3 quality indicators
            matches = sum(1 for indicator in quality_indicators if indicator in code)
            if matches < 3:
                return False
            
            # Check for proper indentation structure
            has_structure = False
            for line in lines:
                if line.startswith('    '):
                    has_structure = True
                    break
            
            if not has_structure:
                return False
            
            return True
            
        except Exception:
            return False

    def fetch_repositories(self, min_stars: int = 1000) -> List[Dict]:
        """Fetch high-quality repositories based on strict criteria."""
        repos = []
        for lang in self.languages:
            # Reduce initial criteria to get more repos
            query = f"language:{lang} stars:>=10 is:public fork:false"
            
            try:
                headers = {
                    **self.headers,
                    'Accept': 'application/vnd.github.v3+json'
                }
                
                # Make multiple requests to get more repositories
                for page in range(1, 6):  # Get up to 500 repos (5 pages * 100 per page)
                    response = requests.get(
                        f"{self.base_url}/search/repositories",
                        params={
                            'q': query,
                            'sort': 'stars',
                            'order': 'desc',
                            'per_page': 100,
                            'page': page
                        },
                        headers=headers
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'items' in data:
                        repos.extend([
                            repo for repo in data['items']
                            if repo.get('size', 0) > 50  # Reduced size threshold
                        ])
                        
                    self.logger.info(f"Found {len(repos)} repositories so far for {lang}")
                    
                    if len(repos) >= 200:  # Stop if we have enough repos
                        break
                        
            except Exception as e:
                self.logger.error(f"Error fetching repositories for {lang}: {str(e)}")
                continue
        
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
        """Fetch code content from repositories with better error handling."""
        code_data = []
        
        # Process repos in parallel with progress bar
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for repo in repos:
                futures.append(
                    executor.submit(self._fetch_repo_contents_recursive, 
                                  repo['full_name'], '', 0)
                )
            
            for future in tqdm(futures, desc="Fetching code content"):
                try:
                    result = future.result(timeout=60)  # Add timeout
                    if result:
                        code_data.extend(result)
                        self.logger.info(f"Current code samples: {len(code_data)}")
                except Exception as e:
                    self.logger.warning(f"Error processing repository: {str(e)}")
                    continue
                
                if len(code_data) >= 1000:  # Stop if we have enough samples
                    break
        
        if len(code_data) < 2:
            raise ValueError(f"Insufficient code samples found: {len(code_data)}")
            
        self.logger.info(f"Successfully fetched {len(code_data)} code files")
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

    def _fetch_repo_contents_recursive(self, repo_name: str, path: str = '', depth: int = 0) -> List[str]:
        """Recursively fetch repository contents with depth limit."""
        if depth > 3:  # Limit recursion depth
            return []
            
        contents_url = f"{self.base_url}/repos/{repo_name}/contents/{path}"
        code_files = []
        
        try:
            response = requests.get(contents_url, headers=self.headers)
            response.raise_for_status()
            items = response.json()
            
            # Handle single file response
            if not isinstance(items, list):
                items = [items]
            
            for item in items:
                try:
                    if item['type'] == 'file' and self._is_valid_file(item):
                        content = self._get_file_content(item['download_url'])
                        if content and self._validate_code_quality(content):
                            code_files.append(content)
                    elif item['type'] == 'dir' and depth < 3:  # Only recurse if not too deep
                        code_files.extend(
                            self._fetch_repo_contents_recursive(
                                repo_name, item['path'], depth + 1
                            )
                        )
                except Exception as e:
                    self.logger.warning(f"Error processing item in {repo_name}: {str(e)}")
                    continue
                    
                if len(code_files) >= 50:  # Limit files per repository
                    break
                    
        except Exception as e:
            self.logger.error(f"Error in recursive fetch for {repo_name}/{path}: {str(e)}")
            
        return code_files

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