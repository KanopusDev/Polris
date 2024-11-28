import requests
from typing import List, Dict
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class GitHubDataLoader:
    def __init__(self, token: str, languages: List[str] = ['python']):
        self.headers = {'Authorization': f'token {token}'}
        self.languages = languages
        self.base_url = "https://api.github.com"
        logging.basicConfig(level=logging.INFO)
        
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