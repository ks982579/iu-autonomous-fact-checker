# ./api/logging_to_file.py
"""
This is Logger created for development purposes. 
Very helpful in saving when and where things go wrong.

Currently removed from the final implementation but saving logic in case required again.
"""
from datetime import datetime
import json
from pathlib import Path
import re
import threading


class LoggingJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            # Resort to '__dict__'
            if hasattr(obj, "__dict__"):
                result = {}
                for k, v in obj.__dict__.items():
                    try:
                        # test first if serializable
                        json.dumps(v)
                        result[k] = v
                    except TypeError:
                        # else make into string
                        result[k] = str(v)
                return result
            else:
                return str(obj)


def stringify(obj, indent=2):
    return json.dumps(obj, indent=indent, cls=LoggingJsonEncoder)


class MyLogger:
    def __init__(self, active: bool, logpath=None, logfile=None):
        self.active = active
        if not self.active:
            return

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        this_file = Path(__file__).resolve()

        # should be getters and setters
        if logpath is None:
            # puts logger in same directory as main file
            self.logpath = Path(f'{this_file.parent}/logging/{self.timestamp}')
        elif logpath is Path:
            self.logpath = logpath
        else:
            raise TypeError("Parameter 'logpath' must be type Path")

        if logfile is None:
            # puts logger in same directory as main file
            self.logfile = Path("main.log")
        elif logfile is Path:
            self.logfile = logfile
        else:
            raise TypeError("Parameter 'logfile' must be type Path")

        self.logpath.mkdir(exist_ok=True, parents=True)
        # Setting up Log File
        self.write_log({
            "action": "creating this log file",
            "timestamp": datetime.now().isoformat()
        })

    def write_log(self, log_obj: object):
        if not self.active:
            return
        with open(self.logpath / self.logfile, 'a', encoding='utf-8') as file:
            try:
                content = stringify(log_obj, indent=2)
                file.write(content+"\n")
                print(content)
            except Exception as err:
                print(f"Logging Error: {err}")

    # parameters are tightly coupled
    def log_article(self, metadata, scraper_response):
        if not self.active:
            return
        try:
            # Logging article content too - different file
            # articlepath = self.logpath / Path(f"{''.join(metadata.source.site_name.casefold().split(sep=None))}/{''.join(metadata.title[:12].casefold().split(sep=None))}")
            articlepath = self.logpath / Path(f"{self._only_alphanum(metadata.source.site_name)}/{self._only_alphanum(metadata.title)[:12]}")
            articlepath.mkdir(exist_ok=True, parents=True)

            # Setting up Log File
            with open(articlepath / "article.log", 'w', encoding='utf-8') as file:
                first_log = {
                    "action": "creating this log file",
                    "timestamp": datetime.now().isoformat()
                }
                file.write(stringify(first_log)+"\n")
                if scraper_response.content is not None:
                    file.write(scraper_response.content)
        except Exception as err:
            print(f"Logging Error: {err}")

    #region Private Methods
    @staticmethod
    def _only_alphanum(content: str) -> str:
        """
        For logging articles with proper directory names.
        Some article titles have punctuation that should not be in directory name.

        Args:
            content (str): string to be made only lowercase alphanumeric

        Returns:
            str: lowercase alphanumeric content - also no whitespace
        """
        expression = r'[^A-Za-z0-9]+'
        return re.sub(expression, '', content.casefold())
    #end region

# -----------------------------------


class MyThreadSafeLogger:
    def __init__(self, active: bool, logpath=None, logfile=None):
        self.active = active
        if not self.active:
            return

        # TODO: private...
        # This will create file
        self.logger = MyLogger(active, logpath, logfile)
        self.lock = threading.Lock()

    def write_log(self, log_obj: object):
        if not self.active:
            return
        with self.lock:
            self.logger.write_log(log_obj)

    # parameters are tightly coupled
    def log_article(self, metadata, scraper_response):
        """
        This should be writing to a new location so shouldn't need lock.

        Args:
            metadata (_type_): _description_
            scraper_response (_type_): _description_
        """
        if not self.active:
            return
        self.logger.log_article(metadata, scraper_response)


# EXAMPLE SETUP!!!
def __main(test_post):

    # You can set logging outside of class
    LOGGING = True
    
    # Optional to direct where logs are saved
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    this_file = Path(__file__).resolve()
    logpath = Path(f'{this_file.parent}/logging/{timestamp}')
    logfile = Path("main.log")  # overall log filename

    logger = MyThreadSafeLogger(LOGGING, logpath=logpath, logfile=logfile)  # This will create file

    # Other Setup Things...

    logger.write_log({
        "action": "logging original post",
        "timestamp": datetime.now().isoformat(),
        "original_post": test_post,
    })

    # Other actions

    logger.write_log({
        "action": "separate into sentences",
        "timestamp": datetime.now().isoformat(),
        "data": {'variable': 'with data you want to see'},
    })
