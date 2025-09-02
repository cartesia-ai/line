# Contributing to Line SDK

Welcome to the Line Voice Agents SDK repository. This file contains guidelines for new contributors.

## Repository Structure
- **Source code:** `line/` contains the implementation
- **Tests:** `tests/` contains the unit tests with a guide to writing tests below.
- **Examples:** `examples/` contains examples of how to use the SDK.
- **Example Integrations:** `example_integrations/` contains examples of how 3rd party providers manage to integrate with our Line SDK.

  > [!note]
  > These examples are developed and maintained by our partners.


## Python Version Support
Line SDK supports Python 3.9+. All code should be written to be compatible with Python 3.9+.

## Coding Style
We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with some modifications (listed below).
Respect code consistency in the repo, including but not limited to naming conventions and code structure.
If you have not already, we recommend setting up pre-commit, which runs some of the linters/formatters prior to each commit.
The same checks will be required to pass in CI, so this will help make the development process smoother.

### Modifications to the Google Python Style Guide
Arguments in the docstring should not have their types mentioned. All typing should happen in the signature of the function or method.

```
# Do this.
def my_function(arg1: int, arg2: str) -> bool:
    """
    My function description.

    Args:
        arg1: My arg1 description.
        arg2: My arg2 description.

    Returns:
        My return description.
    """
    return True

# Don't do this.
def my_function(arg1: int, arg2: str) -> bool:
    """
    My function description.

    Args:
        arg1 (int): My arg1 description.
        arg2 (str): My arg2 description.
    """
    return True
```

### Type Annotations
Functions and methods must be annotated with types.

## Testing
All new logic should be covered by unit tests. Design classes and functions to be unit testable.

If you are changing existing code, tests should be added to cover both the existing and the new behavior. If you are applying a bugfix, tests should exercise the codepath triggering the bug to avoid regressions.

We use [pytest](https://docs.pytest.org/en/stable/) for testing.

## Local Workflow
1. Install dependencies: `pip install -e .`
2. Run tests: `pytest tests/`
3. All python commands can be run with `python` or `uv run python`
4. Run linters with pre-commit: `pre-commit run --all-files`

## Reviewing Process
Reviewers look for the following:
- Small, contained PRs
- Clear documentation and updates to the README
- Clear PR description and title
- Unittests covering the new code
- Consistent style: Use `pre-commit` to run linters and formatters.

## Contributing to Our Example Integrations
Our `example_integrations/` folder contains agents developed by our partners that utilizes their services. If you're interested in adding your own example integration, your PR must satisfy the following requirements:
- Must be deployable on the Line platform
  - You can connect an existing Git repository with the agent code via our [Github Integration](https://docs.cartesia.ai/line/integrations/github) to test the agent
- Must have a README with clear documentation around the usecase with a link to your API docs
  - README should include Prerequisites, Environment Variables, File Overview, and any other information that might be relevant to modifying or using the example agent
  - See our [basic agent README](./examples/basic_chat/README.md) or [form filling README](./examples/form-filling/README.md) for examples
- Must follow our repositories style guide

  > [!note]
  > Partners are responsible for keeping their integration example up to date with Line changes
