# Example Integrations

This directory contains example integrations showing how third-party providers can integrate their services with the Cartesia Line SDK to build powerful voice agents.

## About Example Integrations

Example integrations demonstrate how to:
- Connect external APIs and services to Line voice agents
- Build specialized voice agents for specific use cases
- Implement industry-specific workflows and integrations
- Showcase partner services and capabilities

> [!NOTE]
> While Cartesia approves of these each examples, they are implemented and maintained by our partners.

## Contributing an Integration

Interested in adding your own example integration? See our [Contributing Guide](../CONTRIBUTING.md#contributing-to-our-example-integrations) for requirements.

## Getting Help

- **Documentation**: [Line Docs](https://docs.cartesia.ai/line/introduction)
- **Community**: [Discord community](https://discord.gg/cartesia)
- **More Examples**: Check out [../examples/](../examples/) for core SDK examples

## Structure

Each integration should follow this structure:
```
your_integration_name/
├── README.md         # Integration-specific documentation
├── main.py           # Main agent implementation
├── cartesia.toml     # Line platform configuration
└── requirements.txt  # Dependencies (or pyproject.toml)
```

Additional files may be included as needed for your specific integration (additional modules, configuration files, etc.).

## Disclaimer

All example integrations are provided "as is" without warranty of any kind. Cartesia assumes no liability for the use of these examples. All examples are subject to the license of this repository.
