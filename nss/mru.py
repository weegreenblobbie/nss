import os
from typing import List
from PyQt6.QtCore import QSettings

class MruManager:
    """
    Manages and persists a list of up to 3 Most Recently Used (MRU) directories
    using QSettings.
    """
    def __init__(self, organization: str = "NSS", application: str = "ColorGradingExplorer") -> None:
        # QSettings handles platform-native configuration persistence (Registry on Win, plist on macOS, ini on Linux)
        self.settings = QSettings(organization, application)
        self._mru_list: List[str] = self._load_mru()

    def _load_mru(self) -> List[str]:
        """
        Loads the MRU directory list from persistent storage, verifying existence.
        """
        saved = self.settings.value("mru_directories", [])
        if saved is None:
            return []
        if isinstance(saved, str):
            saved = [saved]
        
        # Only keep valid, existing directories on the system
        valid_dirs: List[str] = []
        for path in saved:
            if isinstance(path, str) and os.path.isdir(path):
                valid_dirs.append(os.path.abspath(path))
        return valid_dirs[:3]

    def _save_mru(self) -> None:
        """
        Saves the MRU list back to persistent storage.
        """
        self.settings.setValue("mru_directories", self._mru_list)

    def get_mru_directories(self) -> List[str]:
        """
        Returns the list of up to 3 MRU directories.
        """
        self._mru_list = self._load_mru()
        return self._mru_list

    def get_most_recent_directory(self) -> str:
        """
        Returns the most recent directory, or the user's home directory if empty.
        """
        dirs = self.get_mru_directories()
        if dirs:
            return dirs[0]
        return os.path.expanduser("~")

    def add_path(self, filepath: str) -> None:
        """
        Extracts the directory from the file path, moves it to the front
        of the MRU stack, and saves the list.
        """
        if not filepath:
            return

        if os.path.isdir(filepath):
            dir_path = os.path.abspath(filepath)
        else:
            dir_path = os.path.abspath(os.path.dirname(filepath))

        if not os.path.isdir(dir_path):
            return

        # Read latest list
        self._mru_list = self._load_mru()

        # Deduplicate and place at front of list
        if dir_path in self._mru_list:
            self._mru_list.remove(dir_path)
        
        self._mru_list.insert(0, dir_path)
        self._mru_list = self._mru_list[:3]
        
        self._save_mru()
