# Contributing to ECG Arrhythmia Detection

This is a personal portfolio project. Contributions and feedback are welcome!

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install dev dependencies: `pip install -e ".[dev]"`

## Testing

Run the test suite with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Code Quality

Before submitting changes, ensure code quality:

```bash
black src/ api/ tests/
ruff check src/ api/ tests/
mypy src/ api/
```

## Pull Requests

- Ensure all tests pass
- Add tests for new features
- Update the README if needed
- Reference any related issues

## Reporting Issues

Please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs. actual behavior
