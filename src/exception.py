import sys
from types import TracebackType


def error_message_details(error: Exception, error_detail: sys) -> str:
    """
    Construct a descriptive error message with traceback details when available.
    """
    _, _, exc_tb = error_detail.exc_info()
    if not isinstance(exc_tb, TracebackType):
        return f"{error}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return (
        f"Error occurred in the python script name [{file_name}] "
        f"line number [{line_number}] with message [{error}]"
    )


class CustomException(Exception):
    def __init__(self, error: Exception, error_details: sys) -> None:
        super().__init__(error)
        self.error_message = error_message_details(error, error_detail=error_details)

    def __str__(self) -> str:
        return self.error_message
