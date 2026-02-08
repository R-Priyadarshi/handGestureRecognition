# Contributing to Hand Gesture Recognition Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. We expect all participants to:

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing private information
- Other unethical or unprofessional conduct

## How to Contribute

### Reporting Bugs

Before submitting a bug report:

1. Check existing issues to avoid duplicates
2. Collect information about your environment
3. Try to reproduce the issue

When reporting a bug, include:

- **Description:** Clear description of the issue
- **Steps to Reproduce:** Minimal steps to reproduce
- **Expected Behavior:** What you expected to happen
- **Actual Behavior:** What actually happened
- **Environment:** OS, Python version, dependencies
- **Screenshots:** If applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Include:

- **Use Case:** Why is this enhancement needed?
- **Proposed Solution:** How would you implement it?
- **Alternatives:** Other approaches considered
- **Impact:** Who benefits from this?

### Contributing Code

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/handGestureRecognition.git
   cd handGestureRecognition
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make Changes**
   - Follow coding standards
   - Add tests for new features
   - Update documentation

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # or
   git commit -m "fix: resolve issue #123"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- (Optional) CUDA for GPU support

### Installation

```bash
# Clone repository
git clone https://github.com/R-Priyadarshi/handGestureRecognition.git
cd handGestureRecognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=training --cov-report=html

# Run specific test file
pytest tests/test_vision.py -v

# Run with markers
pytest -m "not slow"
```

### Code Quality Tools

```bash
# Format code
black core training scripts tests
isort core training scripts tests

# Lint code
flake8 core training scripts tests

# Type checking
mypy core training --ignore-missing-imports

# Run all checks
pre-commit run --all-files
```

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length:** 100 characters (not 79)
- **Imports:** Sorted with isort
- **Formatting:** Black formatter
- **Type Hints:** Use for function signatures
- **Docstrings:** Google style

### Example

```python
"""
Module docstring.

Describes what this module does.
"""

from typing import List, Optional
import numpy as np


class ExampleClass:
    """
    Class docstring.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    """
    
    def __init__(self, param1: str, param2: int = 0):
        self.param1 = param1
        self.param2 = param2
    
    def example_method(self, input_data: np.ndarray) -> List[float]:
        """
        Method docstring.
        
        Args:
            input_data: Input array
            
        Returns:
            List of processed values
            
        Raises:
            ValueError: If input_data is empty
        """
        if len(input_data) == 0:
            raise ValueError("Input data cannot be empty")
        
        return input_data.tolist()
```

### Documentation

- **Code Comments:** Explain *why*, not *what*
- **Docstrings:** Required for all public functions/classes
- **README Updates:** Update README for new features
- **Type Hints:** Use for better IDE support

### Git Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(inference): add WebGPU backend support
fix(vision): resolve landmark normalization bug
docs(readme): update installation instructions
test(temporal): add transformer model tests
```

## Testing

### Test Organization

```
tests/
├── test_vision.py          # Vision module tests
├── test_landmarks.py       # Landmarks module tests
├── test_temporal.py        # Temporal module tests
├── test_inference.py       # Inference module tests
└── test_integration.py     # Integration tests
```

### Writing Tests

```python
import pytest
from your_module import YourClass


@pytest.fixture
def sample_data():
    """Fixture providing sample test data."""
    return {"key": "value"}


def test_basic_functionality(sample_data):
    """Test basic functionality."""
    instance = YourClass()
    result = instance.process(sample_data)
    
    assert result is not None
    assert isinstance(result, dict)


def test_error_handling():
    """Test error handling."""
    instance = YourClass()
    
    with pytest.raises(ValueError):
        instance.process(None)
```

### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths
- Include edge cases
- Test error conditions

## Submitting Changes

### Pull Request Process

1. **Update Documentation**
   - Update README if needed
   - Add docstrings
   - Update CHANGELOG

2. **Ensure Tests Pass**
   ```bash
   pytest
   flake8 core training
   black --check core training
   ```

3. **Create Pull Request**
   - Descriptive title
   - Detailed description
   - Link related issues
   - Request reviews

4. **Address Feedback**
   - Respond to comments
   - Make requested changes
   - Keep discussion focused

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Areas for Contribution

### High Priority

- [ ] Additional gesture classes
- [ ] Mobile app development (TFLite)
- [ ] Performance benchmarks
- [ ] Model quantization improvements
- [ ] Web app enhancements

### Medium Priority

- [ ] Additional model architectures
- [ ] Data augmentation strategies
- [ ] Temporal gesture recognition
- [ ] Multi-hand gesture support
- [ ] Accessibility features

### Documentation

- [ ] Tutorial notebooks
- [ ] Video demonstrations
- [ ] API documentation
- [ ] Deployment guides
- [ ] Translation to other languages

## Questions?

- **GitHub Issues:** For bug reports and feature requests
- **Discussions:** For questions and general discussion
- **Email:** Contact repository maintainers

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in documentation

Thank you for contributing! 🙌
