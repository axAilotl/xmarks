"""
HuggingFace Likes Processor - Fetches liked repositories and creates summaries
Downloads READMEs and generates LLM summaries for liked HuggingFace repositories
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from core.data_models import ProcessingStats
from core.config import config
from core.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceRepo:
    """HuggingFace repository data structure"""
    id: str
    name: str
    full_name: str
    description: Optional[str]
    html_url: str
    likes: int
    downloads: int
    repo_type: str  # 'model', 'dataset', 'space'
    tags: List[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    license: Optional[str]
    library: Optional[str]
    readme_content: Optional[str] = None
    llm_summary: Optional[str] = None
    
    @classmethod
    def from_repo_info(cls, repo_info, repo_type: str = 'model') -> 'HuggingFaceRepo':
        """Create HuggingFaceRepo from HuggingFace Hub repository info"""
        return cls(
            id=repo_info.id,
            name=repo_info.id.split('/')[-1] if '/' in repo_info.id else repo_info.id,
            full_name=repo_info.id,
            description=getattr(repo_info, 'description', None),
            html_url=f"https://huggingface.co/{repo_info.id}",
            likes=getattr(repo_info, 'likes', 0),
            downloads=getattr(repo_info, 'downloads', 0),
            repo_type=repo_type,
            tags=getattr(repo_info, 'tags', []),
            created_at=getattr(repo_info, 'created_at', None),
            updated_at=getattr(repo_info, 'last_modified', None),
            license=getattr(repo_info, 'license', None),
            library=getattr(repo_info, 'library_name', None)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'full_name': self.full_name,
            'description': self.description,
            'html_url': self.html_url,
            'likes': self.likes,
            'downloads': self.downloads,
            'repo_type': self.repo_type,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'license': self.license,
            'library': self.library,
            'readme_content': self.readme_content,
            'llm_summary': self.llm_summary
        }


class HuggingFaceLikesProcessor:
    """Processes HuggingFace liked repositories"""
    
    def __init__(self, vault_path: str = None):
        self.vault_path = Path(vault_path or config.get('vault_dir', 'knowledge_vault'))
        self.stars_dir = self.vault_path / 'stars'
        self.repos_dir = self.vault_path / 'repos'
        
        # Create directories
        self.stars_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        
        # Get HuggingFace username
        self.hf_user = os.getenv('HF_USER')
        if not self.hf_user:
            raise ValueError("HF_USER environment variable is required")
        
        # Import HuggingFace Hub
        try:
            from huggingface_hub import list_liked_repos, hf_hub_download, repo_info
            from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
            self.list_liked_repos = list_liked_repos
            self.hf_hub_download = hf_hub_download
            self.repo_info = repo_info
            self.RepositoryNotFoundError = RepositoryNotFoundError
            self.EntryNotFoundError = EntryNotFoundError
        except ImportError:
            raise ImportError("huggingface_hub package required. Install with: pip install huggingface_hub")
        
        # Initialize LLM interface for summaries
        self.llm_interface = None
        llm_config = config.get('llm', {})
        if llm_config.get('tasks', {}).get('summary', {}).get('enabled', False):
            try:
                self.llm_interface = LLMInterface(llm_config)
                logger.info("LLM interface initialized for HuggingFace likes summaries")
            except Exception as e:
                logger.warning(f"Could not initialize LLM interface: {e}")
    
    async def fetch_and_process_liked_repos(self, limit: int = None, resume: bool = True, 
                                          include_models: bool = True, include_datasets: bool = True, 
                                          include_spaces: bool = True) -> ProcessingStats:
        """Fetch liked repositories and process them"""
        stats = ProcessingStats()
        
        try:
            # Fetch liked repositories
            logger.info(f"📡 Fetching liked repositories for user: {self.hf_user}")
            repos = await self._fetch_liked_repos(limit, include_models, include_datasets, include_spaces)
            
            if not repos:
                logger.warning("No liked repositories found")
                return stats
            
            logger.info(f"📦 Found {len(repos)} liked repositories")
            
            # Process repositories
            for repo in repos:
                try:
                    processed = await self._process_single_repo(repo, resume)
                    if processed:
                        stats.updated += 1
                    else:
                        stats.skipped += 1
                    
                    stats.total_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing repo {repo.full_name}: {e}")
                    stats.errors += 1
            
            # Save repository index
            await self._save_repos_index(repos, "huggingface")
            
            logger.info(f"🤗 HuggingFace likes processing complete: {stats.updated} processed, {stats.skipped} skipped, {stats.errors} errors")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to fetch liked repositories: {e}")
            stats.errors = 1
            return stats
    
    async def _fetch_liked_repos(self, limit: int = None, include_models: bool = True, 
                                include_datasets: bool = True, include_spaces: bool = True) -> List[HuggingFaceRepo]:
        """Fetch liked repositories from HuggingFace Hub"""
        repos = []
        
        try:
            # Get liked repos using HuggingFace Hub
            likes = self.list_liked_repos(self.hf_user)
            
            # Process models
            if include_models and hasattr(likes, 'models'):
                for model_id in likes.models:
                    try:
                        repo_data = self.repo_info(model_id, repo_type='model')
                        repo = HuggingFaceRepo.from_repo_info(repo_data, 'model')
                        repos.append(repo)
                        logger.debug(f"Found liked model: {repo.full_name}")
                    except Exception as e:
                        logger.warning(f"Could not fetch model info for {model_id}: {e}")
            
            # Process datasets  
            if include_datasets and hasattr(likes, 'datasets'):
                for dataset_id in likes.datasets:
                    try:
                        repo_data = self.repo_info(dataset_id, repo_type='dataset')
                        repo = HuggingFaceRepo.from_repo_info(repo_data, 'dataset')
                        repos.append(repo)
                        logger.debug(f"Found liked dataset: {repo.full_name}")
                    except Exception as e:
                        logger.warning(f"Could not fetch dataset info for {dataset_id}: {e}")
            
            # Process spaces
            if include_spaces and hasattr(likes, 'spaces'):
                for space_id in likes.spaces:
                    try:
                        repo_data = self.repo_info(space_id, repo_type='space')
                        repo = HuggingFaceRepo.from_repo_info(repo_data, 'space')
                        repos.append(repo)
                        logger.debug(f"Found liked space: {repo.full_name}")
                    except Exception as e:
                        logger.warning(f"Could not fetch space info for {space_id}: {e}")
            
            # Sort by likes (descending)
            repos.sort(key=lambda r: r.likes, reverse=True)
            
            # Apply limit
            if limit and len(repos) > limit:
                repos = repos[:limit]
            
            logger.info(f"Fetched {len(repos)} liked repositories")
            return repos
            
        except Exception as e:
            logger.error(f"Error fetching liked repos: {e}")
            return []
    
    async def _process_single_repo(self, repo: HuggingFaceRepo, resume: bool) -> bool:
        """Process a single repository"""
        try:
            # Create summary filename (using HuggingFace prefix to distinguish from GitHub)
            safe_name = f"hf_{repo.full_name.replace('/', '_')}"
            summary_file = self.stars_dir / f"{safe_name}_summary.md"
            readme_file = self.repos_dir / f"{safe_name}_README.md"
            
            # Skip if resume enabled and files exist
            if resume and summary_file.exists() and readme_file.exists():
                logger.debug(f"Skipping existing repo: {repo.full_name}")
                return False
            
            # Download README
            readme_content = await self._download_readme(repo)
            if readme_content:
                repo.readme_content = readme_content
                
                # Save README to repos folder
                with open(readme_file, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
                # Generate LLM summary if available
                if self.llm_interface:
                    summary = await self._generate_repo_summary(repo)
                    repo.llm_summary = summary
                
                # Create summary markdown file
                await self._create_summary_file(repo, summary_file)
                
                logger.info(f"✅ Processed HF repo: {repo.full_name} ({repo.likes} ❤️, {repo.repo_type})")
                return True
            else:
                logger.warning(f"Could not download README for {repo.full_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing repo {repo.full_name}: {e}")
            raise
    
    async def _download_readme(self, repo: HuggingFaceRepo) -> Optional[str]:
        """Download README content for a HuggingFace repository"""
        try:
            # Try different README filenames
            readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
            
            for readme_file in readme_files:
                try:
                    # Use HuggingFace Hub to download README
                    local_readme = self.hf_hub_download(
                        repo_id=repo.full_name,
                        filename=readme_file,
                        repo_type=repo.repo_type,
                        local_files_only=False,
                        cache_dir=None
                    )
                    
                    # Read the downloaded file
                    with open(local_readme, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    logger.debug(f"Downloaded {readme_file} for {repo.full_name}")
                    return content
                    
                except (self.RepositoryNotFoundError, self.EntryNotFoundError, FileNotFoundError):
                    continue
                except Exception as e:
                    logger.debug(f"Could not download {readme_file} for {repo.full_name}: {e}")
                    continue
            
            logger.debug(f"No README found for {repo.full_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error downloading README for {repo.full_name}: {e}")
            return None
    
    async def _generate_repo_summary(self, repo: HuggingFaceRepo) -> Optional[str]:
        """Generate LLM summary for repository"""
        try:
            if not self.llm_interface or not repo.readme_content:
                return None
            
            # Create prompt for repository summarization
            repo_type_context = {
                'model': 'machine learning model',
                'dataset': 'dataset', 
                'space': 'application/demo'
            }
            
            context_type = repo_type_context.get(repo.repo_type, 'repository')
            
            prompt_template = f"Provide a concise 2-3 sentence summary of this HuggingFace {context_type}, focusing on what it does and its key features."
            
            context = f"""
Repository: {repo.full_name}
Type: {repo.repo_type}
Description: {repo.description or 'No description'}
Library: {repo.library or 'Unknown'}
Likes: {repo.likes}
Downloads: {repo.downloads}
Tags: {', '.join(repo.tags) if repo.tags else 'None'}

README Content:
{repo.readme_content[:3000]}...  # Limit content to prevent token overflow
"""
            
            summary = await self.llm_interface.summarize_content(context, "readme")
            if summary and not summary.startswith("Summary unavailable"):
                return summary.strip()
            logger.warning(f"LLM summary generation failed for {repo.full_name}")
            return None
                
        except Exception as e:
            logger.error(f"Error generating summary for {repo.full_name}: {e}")
            return None
    
    async def _create_summary_file(self, repo: HuggingFaceRepo, summary_file: Path):
        """Create summary markdown file"""
        try:
            # Create emoji mapping for repo types
            type_emoji = {
                'model': '🤖',
                'dataset': '📊', 
                'space': '🚀'
            }
            
            emoji = type_emoji.get(repo.repo_type, '📦')
            
            # Create comprehensive summary markdown
            content = f"""---
type: huggingface_like
repo_id: {repo.id}
full_name: {repo.full_name}
repo_type: {repo.repo_type}
likes: {repo.likes}
downloads: {repo.downloads}
library: {repo.library or 'Unknown'}
created_at: {repo.created_at}
updated_at: {repo.updated_at}
processed_at: {datetime.now().isoformat()}
---

# {emoji} {repo.full_name}

## 📊 Repository Stats
- **Type**: {repo.repo_type.title()} {emoji}
- **Likes**: {repo.likes} ❤️
- **Downloads**: {repo.downloads:,} ⬇️
- **Library**: {repo.library or 'Unknown'}
- **License**: {repo.license or 'Unknown'}

## 📝 Description
{repo.description or 'No description provided'}

## 🏷️ Tags
{', '.join([f'`{tag}`' for tag in repo.tags]) if repo.tags else 'No tags'}

## 🔗 Links
- **Repository**: [{repo.full_name}]({repo.html_url})
- **README**: [[hf_{repo.full_name.replace('/', '_')}_README.md]]

"""

            # Add LLM summary if available
            if repo.llm_summary:
                content += f"""## 🤖 AI Summary
{repo.llm_summary}

"""

            # Add tags
            tags = [f"#{tag.replace('-', '_').replace(' ', '_').lower()}" for tag in repo.tags[:5]]
            tags.extend([f"#huggingface", f"#liked", f"#{repo.repo_type}", f"#{repo.library.lower()}" if repo.library else "#unknown"])
            
            if repo.likes >= 1000:
                tags.append("#popular")
            elif repo.likes >= 100:
                tags.append("#trending")
            
            if repo.downloads >= 100000:
                tags.append("#widely_used")
            
            content += ' '.join(tags)
            
            # Write summary file
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.debug(f"Created summary file: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error creating summary file for {repo.full_name}: {e}")
            raise
    
    async def _save_repos_index(self, repos: List[HuggingFaceRepo], source: str):
        """Save repository index for reference"""
        try:
            index_file = self.stars_dir / f"{source}_liked_repos_index.json"
            
            index_data = {
                "source": source,
                "user": self.hf_user,
                "generated_at": datetime.now().isoformat(),
                "total_repos": len(repos),
                "repositories": [repo.to_dict() for repo in repos]
            }
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, default=str)
            
            logger.info(f"💾 Saved repository index: {index_file}")
            
        except Exception as e:
            logger.error(f"Error saving repository index: {e}")
    
    def get_stats_summary(self, stats: ProcessingStats) -> str:
        """Generate formatted stats summary"""
        return (f"HuggingFace Likes processing: {stats.updated} processed, "
                f"{stats.skipped} skipped, {stats.errors} errors")
