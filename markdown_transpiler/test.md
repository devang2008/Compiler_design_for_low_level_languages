# Markdown Transpiler — Test Document

This file exercises **every feature** supported by the transpiler.

## Headings

### This is an H3 heading

#### And this is H4

## Text Formatting

This is a normal paragraph with **bold text**, *italic text*,
and ***bold italic text*** all in one line.

You can also use `inline code` to highlight variables like `x = 42`.

## Links and Images

Visit [Python Official Site](https://www.python.org) for more info.

Here is an image example:

![Python Logo](https://www.python.org/static/community_logos/python-logo.png)

## Lists

### Unordered List

- First item with **bold**
- Second item with *italic*
- Third item with `code`

### Ordered List

1. Step one: install Python
2. Step two: write a lexer
3. Step three: build a parser
4. Step four: generate output

## Code Blocks

Here is a Python code block:

```python
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))  # Output: 55
```

And a generic code block:

```
$ python main.py test.md
[1/3] Lexing ...
[2/3] Parsing ...
[3/3] Generating HTML ...
```

## Blockquotes

> This is a blockquote. It can contain **bold** and *italic* text.
> It can also span multiple lines like this.

> Another separate blockquote with a [link](https://example.com).

## Horizontal Rules

Content above the rule.

---

Content below the rule.

## Combined Features

Here is a paragraph that combines a [link](https://example.com),
some **bold text**, an `inline code snippet`, and *italics* all together.

1. A list item with a [clickable link](https://example.com)
2. Another item with ***bold and italic***
3. Final item with `highlighted code`

---

*End of test document.*
