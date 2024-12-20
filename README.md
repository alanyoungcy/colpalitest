# Data Processing and LLM OCR Tools

This repository contains various tools and scripts for data processing and Large Language Model (LLM) OCR tasks. The tools include implementations for document processing, vision-language models, and API integrations with different LLM services.

## Features

- **ColPali Document Processing**: Implementation of the ColPali model for document understanding and visual question answering
- **Grok API Integration**: Examples of using both the XAI SDK and OpenAI-compatible endpoints for Grok
- **Multi-Platform Support**: Support for different compute devices (CPU, MPS for Mac)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
Create a `.env` file in the root directory with:
```
XAI_API_KEY=your-xai-api-key-here
```

## Scripts

### llmtable.py
Document processing and visual question answering using the ColPali model. Supports:
- Image processing from various sources
- Query processing
- Embedding generation
- Retrieval evaluation

### grokapitest.py
Example implementation using the XAI SDK for text generation with Grok.

### grok2test.py
OpenAI-compatible implementation for interacting with Grok-2, including vision capabilities.

## Requirements

- Python 3.11+
- PyTorch
- transformers
- xai_sdk
- python-dotenv
- openai

## Usage

Each script can be run independently. For example:

```bash
python llmtable.py
python grokapitest.py
python grok2test.py
```

## Notes

- Make sure to keep your API keys secure and never commit them to version control
- Some features require specific hardware support (e.g., MPS for Mac)
- The ColPali implementation supports multiple input sources including datasets and PDFs

## Contributing

Feel free to submit issues and enhancement requests.

## License

[Your chosen license]
