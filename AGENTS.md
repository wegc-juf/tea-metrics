## PyCharm MCP integration

When the PyCharm MCP server is available, use its IDE capabilities where they provide an advantage over native tools.

### Prefer PyCharm MCP for
- Code inspections and diagnostics (`get_file_problems`, `lint_files`).
- Symbol navigation, symbol information and call hierarchies (`search_symbol`, `get_symbol_info`, `analyze_calls`).
- Running existing PyCharm run configurations (`get_run_configurations`, `execute_run_configuration`).
- Applying IDE-specific code formatting (`reformat_file`).

### Prefer native OpenCode tools for
- Reading and editing ordinary project files.
- Git operations and diff inspection.
- Simple text searches and directory navigation.
- Running straightforward shell commands and tests that do not depend on PyCharm configuration.

### Safety and fallback
- Before using PyCharm MCP, verify that its project root matches the current working tree.
- Never use PyCharm MCP to modify or execute code in a different worktree.
- If MCP is unavailable or targets another project, continue using native tools.
- Do not repeatedly retry unavailable MCP tools.
- After substantial code changes, run relevant tests and use PyCharm inspections when available.
- Do not execute destructive commands or perform broad refactorings without appropriate confirmation.
