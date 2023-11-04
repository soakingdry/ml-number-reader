from colorama import Fore, Style


def info(content: str, prefix="-"):
    color = Fore.LIGHTBLUE_EX
    print(f"{Style.BRIGHT}[{color}{prefix}{Fore.RESET}]: {content} {Style.RESET_ALL}")


def error(content: str, prefix="-"):
    color = Fore.LIGHTRED_EX
    print(f"{Style.BRIGHT}[{color}{prefix}{Fore.RESET}]: {content} {Style.RESET_ALL}")


def warn(content: str, prefix="-"):
    color = Fore.LIGHTYELLOW_EX
    print(f"{Style.BRIGHT}[{color}{prefix}{Fore.RESET}]: {content} {Style.RESET_ALL}")


def success(content: str, prefix="-"):
    color = Fore.LIGHTGREEN_EX
    print(f"{Style.BRIGHT}[{color}{prefix}{Fore.RESET}]: {content} {Style.RESET_ALL}")
