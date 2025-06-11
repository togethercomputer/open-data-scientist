import io
import os
from contextlib import redirect_stdout, redirect_stderr
from typing import Any
from session_manager import Session

import pandas as pd
import numpy as np
import matplotlib
import sklearn


class CodeExecutor:
    def __init__(self):
        self.safe_builtins = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "abs": abs,
                "max": max,
                "min": min,
                "sum": sum,
                "next": next,
                "exec": exec,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "type": type,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "dir": dir,
                "vars": vars,
                "globals": globals,
                "locals": locals,
                "round": round,
                "pow": pow,
                "divmod": divmod,
                "all": all,
                "any": any,
                "chr": chr,
                "ord": ord,
                "hex": hex,
                "oct": oct,
                "bin": bin,
                "os": os,
                "open": open,
                "__import__": __import__,
                "__build_class__": __build_class__,
                "__name__": "__main__",
            },
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "plt": matplotlib,
            "sklearn": sklearn,
        }

    async def execute(self, code: str, session: Session) -> Any:
        async with session.lock:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            session.namespace.update(self.safe_builtins)

            original_cwd = os.getcwd()
            os.chdir("/app/custom_data")
            try:
            
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = eval(
                        compile(code, "<string>", "eval"),
                        session.namespace,
                        session.namespace,
                    )

                output = stdout_capture.getvalue()
                if output:
                    return output.strip()
                return result

            except SyntaxError:
                try:
                    with (
                        redirect_stdout(stdout_capture),
                        redirect_stderr(stderr_capture),
                    ):
                        exec(
                            compile(code, "<string>", "exec"),
                            session.namespace,
                            session.namespace,
                        )

                    output = stdout_capture.getvalue()
                    error_output = stderr_capture.getvalue()

                    if error_output:
                        raise Exception(error_output.strip())

                    return output.strip() if output else None

                except Exception as e:
                    error_output = stderr_capture.getvalue()
                    if error_output:
                        raise Exception(error_output.strip())
                    raise e

            except Exception as e:
                error_output = stderr_capture.getvalue()
                if error_output:
                    raise Exception(error_output.strip())
                raise e
            finally:
                os.chdir(original_cwd)
