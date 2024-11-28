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
        """Load data from CodeParrot dataset."""
        try:
            dataset = load_dataset("codeparrot/codeparrot-clean", split="train[:10000]")
            code_samples = [
                sample['content'] 
                for sample in dataset 
                if any(lang in sample.get('lang', '').lower() 
                      for lang in self.languages)
            ][:subset_size]
            return code_samples
        except Exception as e:
            self.logger.error(f"Error loading CodeParrot data: {str(e)}")
            return []

    def _load_codenet_data(self, subset_size: int = 5000) -> List[str]:
        """Load data from Project CodeNet dataset."""
        try:
            dataset = load_dataset("codenet", split="train[:10000]")
            code_samples = [
                sample['code'] 
                for sample in dataset 
                if any(lang in sample.get('language', '').lower() 
                      for lang in self.languages)
            ][:subset_size]
            return code_samples
        except Exception as e:
            self.logger.error(f"Error loading CodeNet data: {str(e)}")
            return []

    def fetch_repositories(self, min_stars: int = 100) -> List[Dict]:
        repos = []
        for lang in self.languages:
            query = f"language:{lang} stars:>={min_stars}"
            url = f"{self.base_url}/search/repositories?q={query}&sort=stars"
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()  # Raise exception for bad status codes
                data = response.json()
                if 'items' not in data:
                    logging.error(f"No 'items' found in response for {lang}: {data}")
                    continue
                repos.extend(data['items'])
                logging.info(f"Found {len(data['items'])} repositories for {lang}")
            except Exception as e:
                logging.error(f"Error fetching repositories for {lang}: {str(e)}")
        
        if not repos:
            raise ValueError("No repositories found. Check your GitHub token and rate limits.")
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