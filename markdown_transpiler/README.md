# Markdown to HTML Transpiler

A lightweight, dependency-free Markdown to HTML transpiler written entirely from scratch in Python. This project translates standard Markdown into beautifully formatted HTML using a custom processing pipeline.

## Features Supported
- **Headings**: H1 through H6 (`#` to `######`)
- **Text Styles**: **Bold** (`**text**`), *Italic* (`*text*`), and ***Bold-Italic*** (`***text***`)
- **Lists**: Ordered (`1.`) and Unordered (`-` or `*`)
- **Code**: `Inline Code` and Fenced Code Blocks (```)
- **Quotes**: Blockquotes (`> text`)
- **Links & Images**: Standard Markdown syntax (`[text](url)` and `![alt](url)`)
- **Structure**: Paragraphs and Horizontal Rules (`---`)
- **Security**: Built-in HTML escaping for safe rendering.

## Architecture Pipeline
The transpiler follows a standard compiler front-end and back-end pipeline:
1. **Lexer (`lexer.py`)**: Reads the raw text, parses inline styles (regex-based), escapes HTML components, and handles line breaks.
2. **Parser (`parser.py`)**: Analyzes the lexical lines and groups them into logical block-level AST (Abstract Syntax Tree) structures like paragraphs, lists, and code blocks.
3. **Generator (`generator.py`)**: Walks the AST and emits valid HTML strings. Embeds a beautiful, modern default CSS stylesheet for instant visual feedback.

## How to Run

1. Open a terminal in the `markdown_transpiler` directory.
2. Place the markdown content you want to convert into `test.md`.
3. Run the main script:
   ```bash
   python main.py
   ```
4. The script will read `test.md` and generate an `output.html` file in the same directory.
5. Open `output.html` in your web browser to view the rendered result!

## Project Files
- `lexer.py`: Lexical analysis and inline parsing
- `parser.py`: Block-level parsing
- `generator.py`: HTML generation and styling
- `main.py`: Entry point to run the pipeline
- `test.md` & `output.html`: Input and Output files
