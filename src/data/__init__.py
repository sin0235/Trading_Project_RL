__all__ = ["DownloadData", "DataProcessor"]


def __getattr__(name):
    if name == "DownloadData":
        from .download_data import DownloadData
        return DownloadData
    if name == "DataProcessor":
        from .data_processor import DataProcessor
        return DataProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
