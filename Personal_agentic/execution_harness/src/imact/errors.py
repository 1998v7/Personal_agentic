class HarnessError(Exception):
    """可交给外层处理的稳定错误码，不包含密钥或上游原始错误页。"""

    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self):
        return {"code": self.code, "message": self.message}
