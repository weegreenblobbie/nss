import os
import pytest
from unittest.mock import patch
from nss.mru import MruManager

def mock_isdir(path: str) -> bool:
    # Treat paths ending with file extensions as files (not directories)
    if path.lower().endswith(('.tif', '.tiff', '.txt', '.png', '.jpg')):
        return False
    return True

def test_mru_manager_flow() -> None:
    # Normalize paths so they are formatted consistently on POSIX and Windows using relative base paths
    dir1 = os.path.abspath("dummy/existing/dir1")
    dir2 = os.path.abspath("dummy/existing/dir2")
    dir3 = os.path.abspath("dummy/existing/dir3")
    
    # Under mock, QSettings is initialized with default ["/dummy/existing/dir1", "/dummy/existing/dir2"] normalized
    # Patch os.path.isdir to return True only for mock directories
    with patch('os.path.isdir', side_effect=mock_isdir):
        mru = MruManager()
        
        # Load verified MRU directories and verify
        dirs = mru.get_mru_directories()
        assert dirs == [dir1, dir2]
        
        # Verify default recent directory
        assert mru.get_most_recent_directory() == dir1
        
        # Add a third directory path from a file
        mru.add_path(os.path.join(dir3, "image.tif"))
        
        # Verify it was added to front and persisted in the mock settings store
        assert mru.get_mru_directories() == [dir3, dir1, dir2]

def test_mru_manager_limit_and_deduplicate() -> None:
    dir1 = os.path.abspath("dummy/dir1")
    dir2 = os.path.abspath("dummy/dir2")
    dir3 = os.path.abspath("dummy/dir3")
    dir4 = os.path.abspath("dummy/dir4")

    with patch('os.path.isdir', side_effect=mock_isdir):
        mru = MruManager()
        # Explicitly set store using normalized paths
        mru.settings._store["mru_directories"] = [dir1, dir2, dir3]
        
        # Add a duplicate path (should move dir3 to the front)
        mru.add_path(dir3)
        assert mru.get_mru_directories() == [dir3, dir1, dir2]
        
        # Add a 4th new path (should slide dir2 out because limit is 3)
        mru.add_path(dir4)
        assert mru.get_mru_directories() == [dir4, dir3, dir1]
