"""
Markdown Generator - Shared utility for generating markdown sections
Consolidates duplicate markdown generation logic from ContentProcessor and ThreadProcessor
"""

import time
from typing import List, Optional, Dict, Any, Union


class MarkdownGenerator:
    """Shared utility for generating markdown sections used by both tweet and thread processors"""
    
    @staticmethod
    def generate_frontmatter(metadata: Dict[str, Any]) -> List[str]:
        """Generate YAML frontmatter for markdown files"""
        lines = ["---"]
        
        for key, value in metadata.items():
            if isinstance(value, str):
                lines.append(f"{key}: \"{value}\"")
            else:
                lines.append(f"{key}: {value}")
        
        lines.append("---")
        lines.append("")
        return lines
    
    @staticmethod
    def generate_arxiv_section(papers: List[Any], detailed: bool = True) -> List[str]:
        """Generate ArXiv papers section for markdown files
        
        Args:
            papers: List of ArXiv papers
            detailed: If True, include full details; if False, use compact format
        """
        if not papers:
            return []
        
        lines = []
        section_title = "## ArXiv Papers" if detailed else "### ArXiv Papers"
        lines.append(section_title)
        
        for i, paper in enumerate(papers, 1):
            if detailed:
                lines.append(f"### Paper {i}: {paper.title}")
                lines.append(f"- **ArXiv ID**: [{paper.arxiv_id}]({paper.abs_url})")
                lines.append(f"- **Abstract**: {paper.abstract}")
            else:
                lines.append(f"**Paper {i}: {paper.title}**")
                lines.append(f"- **ArXiv ID**: [{paper.arxiv_id}]({paper.abs_url})")
                lines.append(f"- **Abstract**: {paper.abstract}")
            
            if paper.downloaded and paper.filename:
                lines.append(f"- **PDF**: [[{paper.filename}]]")
            lines.append("")
        
        return lines
    
    @staticmethod
    def generate_pdf_section(pdf_links: List[Any], detailed: bool = True) -> List[str]:
        """Generate PDF documents section for markdown files
        
        Args:
            pdf_links: List of PDF links
            detailed: If True, include full details; if False, use compact format
        """
        if not pdf_links:
            return []
        
        lines = []
        section_title = "## PDF Documents" if detailed else "### PDF Documents"
        lines.append(section_title)
        
        for i, pdf_link in enumerate(pdf_links, 1):
            if detailed:
                lines.append(f"### Document {i}: {pdf_link.title}")
                lines.append(f"- **URL**: [{pdf_link.url}]({pdf_link.url})")
            else:
                lines.append(f"**Document {i}: {pdf_link.title}**")
                lines.append(f"- **URL**: [{pdf_link.url}]({pdf_link.url})")
            
            if pdf_link.downloaded and pdf_link.filename:
                lines.append(f"- **PDF**: [[{pdf_link.filename}]]")
            lines.append("")
        
        return lines
    
    @staticmethod
    def generate_repo_section(repo_links: List[Any], detailed: bool = True) -> List[str]:
        """Generate repository READMEs section for markdown files
        
        Args:
            repo_links: List of repository links
            detailed: If True, include full details; if False, use compact format
        """
        if not repo_links:
            return []
        
        lines = []
        section_title = "## Repository READMEs" if detailed else "### Repository READMEs"
        lines.append(section_title)
        
        for repo_link in repo_links:
            platform_name = repo_link.platform.upper()
            if detailed:
                lines.append(f"### {platform_name} REPO: {repo_link.repo_name}")
            else:
                lines.append(f"**{platform_name} REPO: {repo_link.repo_name}**")
            
            # Use LLM summary if available, otherwise placeholder
            if hasattr(repo_link, 'llm_summary') and repo_link.llm_summary:
                lines.append(f"- **Summary**: {repo_link.llm_summary}")
            else:
                lines.append(f"- **Summary**: PLACEHOLDER")
            
            if repo_link.downloaded and repo_link.filename:
                lines.append(f"- **README**: [[{repo_link.filename}]]")
            lines.append("")
        
        return lines
    
    @staticmethod
    def generate_youtube_section(youtube_videos: List[Any], detailed: bool = True) -> List[str]:
        """Generate YouTube videos section for markdown files
        
        Args:
            youtube_videos: List of YouTube video objects
            detailed: If True, include full details; if False, use compact format
        """
        if not youtube_videos:
            return []
        
        lines = []
        section_title = "## YouTube Videos" if detailed else "### YouTube Videos"
        lines.append(section_title)
        
        for i, video in enumerate(youtube_videos, 1):
            if detailed:
                lines.append(f"### Video {i}: {video.title}")
                lines.append(f"- **Channel**: {video.channel_title}")
                lines.append(f"- **Duration**: {video.duration or 'Unknown'}")
                lines.append(f"- **Views**: {f'{video.view_count:,}' if video.view_count else 'Unknown'}")
                lines.append(f"- **YouTube Link**: [https://youtu.be/{video.video_id}](https://youtu.be/{video.video_id})")
                
                # Embed video if embeddings enabled
                from core.config import config
                if config.get('youtube.enable_embeddings', True):
                    lines.append(f"- **Video**: ![YouTube Video](https://youtu.be/{video.video_id})")
                
                # Include transcript link if available
                if video.transcript:
                    safe_title = f"youtube_{video.video_id}_{MarkdownGenerator._sanitize_filename(video.title)}"
                    lines.append(f"- **Transcript**: [[{safe_title}.md]]")
            else:
                lines.append(f"**Video {i}: {video.title}**")
                lines.append(f"- **Channel**: {video.channel_title}")
                lines.append(f"- **Link**: [https://youtu.be/{video.video_id}](https://youtu.be/{video.video_id})")
            
            lines.append("")
        
        return lines
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe filesystem usage"""
        import re
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces and special chars with underscores
        filename = re.sub(r'[\s\-\[\]()]+', '_', filename)
        # Remove multiple underscores
        filename = re.sub(r'_+', '_', filename)
        # Limit length
        filename = filename[:50]
        # Remove trailing underscores
        filename = filename.strip('_')
        return filename or 'untitled'
    
    @staticmethod
    def generate_media_embeds(media_items: List[Any]) -> List[str]:
        """Generate media embeds with Obsidian-style links"""
        if not media_items:
            return []
        
        lines = []
        for media in media_items:
            if media.media_type in ['video', 'animated_gif']:
                # For videos: show thumbnail but link to video file
                if media.filename and media.video_filename:
                    # Show thumbnail image but link to video
                    lines.append(f"[[{media.video_filename}|![[{media.filename}]]]]")
                elif media.filename:
                    # Fallback: just show thumbnail if video not downloaded
                    lines.append(f"![[{media.filename}]]")
                    lines.append("*Video file not available*")
            else:
                # For photos: embed directly
                if media.filename:
                    lines.append(f"![[{media.filename}]]")
            
            # Add alt text if available
            if hasattr(media, 'alt_text') and media.alt_text:
                lines.append(f"*{media.alt_text}*")
        
        return lines
    
    @staticmethod
    def generate_tags(tags: List[str], author: str) -> List[str]:
        """Generate a tag section with header and formatted tags (author first)"""
        if not tags:
            return []

        lines: List[str] = []
        lines.append("## Tags")

        # Ensure author tag is included and comes first
        tags_with_author = [f"@{author}"] + [tag for tag in tags if tag != f"@{author}"]
        tags_str = ' '.join([f"#{tag.replace(' ', '-').replace('@', '').lower()}" for tag in tags_with_author])
        lines.append(tags_str)
        lines.append("")
        return lines
    
    @staticmethod
    def generate_thread_info_section(tweet: Any, detailed: bool = True) -> List[str]:
        """Generate thread info section for tweets that are part of threads"""
        if tweet.display_type != 'SelfThread':
            return []
        
        lines = []
        section_title = "## Thread Info" if detailed else "### Thread Info"
        lines.append(section_title)
        lines.append("- **Type**: Self Thread")
        
        if tweet.thread_id:
            lines.append(f"- **Thread ID**: {tweet.thread_id}")
            lines.append(f"- **Thread File**: [[thread_{tweet.thread_id}_{tweet.screen_name}]]")
        
        lines.append("")
        return lines
    
    @staticmethod
    def generate_summary_section(summary: str, detailed: bool = True) -> List[str]:
        """Generate summary section if summary exists"""
        if not summary:
            return []
        
        lines = []
        section_title = "## Summary" if detailed else "### Summary"
        lines.append(section_title)
        lines.append(summary)
        lines.append("")
        return lines
    
    @staticmethod
    def should_summarize_tweet(tweet: Any) -> bool:
        """Check if tweet should be summarized (512+ characters)"""
        if not tweet.full_text:
            return False
        return len(tweet.full_text) > 512