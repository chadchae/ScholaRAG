"""
ScholaRAG File Conventions
==========================

Centralized file naming conventions for all pipeline stages.
Ensures consistency between stages and prevents file path mismatches.

Usage:
    from core.file_conventions import FileConventions
    conventions = FileConventions(project_path)
    relevant_file = conventions.get_relevant_papers_path()
"""

from pathlib import Path
from typing import Optional


class FileConventions:
    """
    Centralized file naming conventions for ScholaRAG pipeline.

    This class ensures all stages use consistent file paths,
    preventing the critical bug where Stage 3 outputs auto_included.csv
    but Stage 4/7 expects relevant_papers.csv.
    """

    # Stage 1: Identification (01_fetch_papers.py)
    IDENTIFICATION_DIR = "data/01_identification"
    SEMANTIC_SCHOLAR_RESULTS = "semantic_scholar_results.csv"
    OPENALEX_RESULTS = "openalex_results.csv"
    ARXIV_RESULTS = "arxiv_results.csv"
    SCOPUS_RESULTS = "scopus_results.csv"
    WOS_RESULTS = "wos_results.csv"
    DEDUPLICATED = "deduplicated.csv"

    # Stage 2: Screening (03_screen_papers.py)
    SCREENING_DIR = "data/02_screening"

    # v1.2.6: Unified output file names
    # Stage 3 now outputs BOTH auto_included.csv (legacy) AND relevant_papers.csv (standard)
    # This ensures backward compatibility while fixing the pipeline
    RELEVANT_PAPERS = "relevant_papers.csv"  # PRIMARY - used by Stage 4, 7
    EXCLUDED_PAPERS = "excluded_papers.csv"  # PRIMARY - used by Stage 7

    # Legacy names (for backward compatibility)
    AUTO_INCLUDED = "auto_included.csv"      # LEGACY - kept for compatibility
    AUTO_EXCLUDED = "auto_excluded.csv"      # LEGACY - kept for compatibility
    HUMAN_REVIEW_QUEUE = "human_review_queue.csv"
    ALL_SCREENED = "all_screened_papers.csv"
    SCREENING_PROGRESS = "screening_progress.csv"

    # Stage 3: PDF Download (04_download_pdfs.py)
    PDF_DIR = "data/03_pdfs"
    DOWNLOAD_LOG = "download_log.csv"
    PAPERS_METADATA = "papers_metadata.csv"

    # Stage 4: RAG (05_build_rag.py)
    RAG_DIR = "data/04_rag"
    CHROMA_DB = "chroma_db"
    RAG_CONFIG = "rag_config.json"

    # Stage 5: Outputs (07_generate_prisma.py)
    OUTPUTS_DIR = "outputs"
    PRISMA_DIAGRAM_PNG = "prisma_diagram.png"
    PRISMA_DIAGRAM_PDF = "prisma_diagram.pdf"
    STATISTICS_REPORT = "statistics_report.md"
    STATISTICS_JSON = "statistics.json"

    def __init__(self, project_path: str | Path):
        """
        Initialize file conventions for a project.

        Args:
            project_path: Path to the project directory
        """
        self.project_path = Path(project_path)

    # ========== Identification Stage ==========

    def get_identification_dir(self) -> Path:
        """Get identification stage directory"""
        return self.project_path / self.IDENTIFICATION_DIR

    def get_semantic_scholar_path(self) -> Path:
        """Get Semantic Scholar results file path"""
        return self.get_identification_dir() / self.SEMANTIC_SCHOLAR_RESULTS

    def get_openalex_path(self) -> Path:
        """Get OpenAlex results file path"""
        return self.get_identification_dir() / self.OPENALEX_RESULTS

    def get_arxiv_path(self) -> Path:
        """Get arXiv results file path"""
        return self.get_identification_dir() / self.ARXIV_RESULTS

    def get_scopus_path(self) -> Path:
        """Get Scopus results file path"""
        return self.get_identification_dir() / self.SCOPUS_RESULTS

    def get_wos_path(self) -> Path:
        """Get Web of Science results file path"""
        return self.get_identification_dir() / self.WOS_RESULTS

    def get_deduplicated_path(self) -> Path:
        """Get deduplicated papers file path"""
        return self.get_identification_dir() / self.DEDUPLICATED

    # ========== Screening Stage ==========

    def get_screening_dir(self) -> Path:
        """Get screening stage directory"""
        return self.project_path / self.SCREENING_DIR

    def get_relevant_papers_path(self) -> Path:
        """
        Get relevant papers file path (PRIMARY).

        This is the standard name used by Stage 4 (download) and Stage 7 (PRISMA).
        """
        return self.get_screening_dir() / self.RELEVANT_PAPERS

    def get_excluded_papers_path(self) -> Path:
        """
        Get excluded papers file path (PRIMARY).

        This is the standard name used by Stage 7 (PRISMA).
        """
        return self.get_screening_dir() / self.EXCLUDED_PAPERS

    def get_auto_included_path(self) -> Path:
        """Get auto-included papers file path (LEGACY)"""
        return self.get_screening_dir() / self.AUTO_INCLUDED

    def get_auto_excluded_path(self) -> Path:
        """Get auto-excluded papers file path (LEGACY)"""
        return self.get_screening_dir() / self.AUTO_EXCLUDED

    def get_human_review_path(self) -> Path:
        """Get human review queue file path"""
        return self.get_screening_dir() / self.HUMAN_REVIEW_QUEUE

    def get_all_screened_path(self) -> Path:
        """Get all screened papers file path"""
        return self.get_screening_dir() / self.ALL_SCREENED

    def get_screening_progress_path(self) -> Path:
        """Get screening progress file path"""
        return self.get_screening_dir() / self.SCREENING_PROGRESS

    # ========== PDF Stage ==========

    def get_pdf_dir(self) -> Path:
        """Get PDF download directory"""
        return self.project_path / self.PDF_DIR

    def get_download_log_path(self) -> Path:
        """Get download log file path"""
        return self.get_pdf_dir() / self.DOWNLOAD_LOG

    def get_papers_metadata_path(self) -> Path:
        """Get papers metadata file path"""
        return self.get_pdf_dir() / self.PAPERS_METADATA

    # ========== RAG Stage ==========

    def get_rag_dir(self) -> Path:
        """Get RAG directory"""
        return self.project_path / self.RAG_DIR

    def get_chroma_db_path(self) -> Path:
        """Get ChromaDB directory path"""
        return self.get_rag_dir() / self.CHROMA_DB

    def get_rag_config_path(self) -> Path:
        """Get RAG config file path"""
        return self.get_rag_dir() / self.RAG_CONFIG

    # ========== Outputs Stage ==========

    def get_outputs_dir(self) -> Path:
        """Get outputs directory"""
        return self.project_path / self.OUTPUTS_DIR

    def get_prisma_png_path(self) -> Path:
        """Get PRISMA diagram PNG path"""
        return self.get_outputs_dir() / self.PRISMA_DIAGRAM_PNG

    def get_prisma_pdf_path(self) -> Path:
        """Get PRISMA diagram PDF path"""
        return self.get_outputs_dir() / self.PRISMA_DIAGRAM_PDF

    def get_statistics_report_path(self) -> Path:
        """Get statistics report path"""
        return self.get_outputs_dir() / self.STATISTICS_REPORT

    def get_statistics_json_path(self) -> Path:
        """Get statistics JSON path"""
        return self.get_outputs_dir() / self.STATISTICS_JSON

    # ========== Utility Methods ==========

    def ensure_directories(self):
        """Create all pipeline directories if they don't exist"""
        directories = [
            self.get_identification_dir(),
            self.get_screening_dir(),
            self.get_pdf_dir(),
            self.get_rag_dir(),
            self.get_outputs_dir(),
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_all_database_result_paths(self) -> dict[str, Path]:
        """
        Get all database result file paths.

        Returns:
            Dictionary mapping database name to file path
        """
        return {
            'semantic_scholar': self.get_semantic_scholar_path(),
            'openalex': self.get_openalex_path(),
            'arxiv': self.get_arxiv_path(),
            'scopus': self.get_scopus_path(),
            'wos': self.get_wos_path(),
        }


# Convenience function for quick access
def get_conventions(project_path: str | Path) -> FileConventions:
    """
    Get file conventions for a project.

    Args:
        project_path: Path to the project directory

    Returns:
        FileConventions instance
    """
    return FileConventions(project_path)
