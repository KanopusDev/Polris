
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
            response = requests.get(url, headers=self.headers)
            repos.extend(response.json()['items'])
        return repos

    def fetch_code_content(self, repos: List[Dict]) -> List[str]:
        code_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for repo in repos:
                futures.append(
                    executor.submit(self._fetch_repo_contents, repo['full_name'])
                )
            for future in tqdm(futures):
                code_data.extend(future.result())
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