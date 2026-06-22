"""异常体系"""

class MyAgentsException(Exception):
    """MyAgents基础异常类"""
    pass

class LLMException(MyAgentsException):
    """LLM相关异常"""
    pass

class AgentException(MyAgentsException):
    """Agent相关异常"""
    pass

class ConfigException(MyAgentsException):
    """配置相关异常"""
    pass

class ToolException(MyAgentsException):
    """工具相关异常"""
    pass

