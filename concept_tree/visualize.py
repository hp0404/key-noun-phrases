"""Visualize CiteSpace concept tree XML output.

Usage:
    python visualize.py concept_tree.xml              # Text tree view
    python visualize.py concept_tree.xml --html       # HTML interactive view
    python visualize.py concept_tree.xml --graph      # Graphviz PNG (requires graphviz)
"""

from __future__ import annotations

import argparse
import html
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_tree(xml_path: Path) -> dict:
    """Parse TreeML XML into a nested dictionary structure."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def parse_node(elem) -> dict | None:
        """Parse a branch or leaf element."""
        name_attr = elem.find('attribute[@name="name"]')
        freq_attr = elem.find('attribute[@name="freq"]')
        context_attr = elem.find('attribute[@name="context"]')

        if name_attr is None:
            return None

        node = {
            "name": name_attr.get("value", ""),
            "freq": int(freq_attr.get("value", 0)) if freq_attr is not None else 0,
            "context": context_attr.get("value", "") if context_attr is not None else "",
            "children": [],
            "type": elem.tag,
        }

        # Parse children
        for child in elem:
            if child.tag in ("branch", "leaf"):
                child_node = parse_node(child)
                if child_node:
                    node["children"].append(child_node)

        return node

    # Find the root Concepts branch
    concepts_branch = root.find('.//branch')
    if concepts_branch is not None:
        return parse_node(concepts_branch)
    return {"name": "Empty", "freq": 0, "children": [], "type": "branch", "context": ""}


def print_text_tree(node: dict, prefix: str = "", is_last: bool = True, show_freq: bool = True) -> None:
    """Print tree as ASCII art."""
    connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "

    # Skip root "Concepts" node, just show its children
    if node["name"] == "Concepts":
        for i, child in enumerate(node["children"]):
            print_text_tree(child, "", i == len(node["children"]) - 1, show_freq)
        return

    # Node type indicator
    if node["type"] == "branch" and node["children"]:
        type_icon = "\u25cf"  # filled circle for branch
    else:
        type_icon = "\u25cb"  # empty circle for leaf

    # Format name with frequency
    freq_str = f" ({node['freq']})" if show_freq else ""
    print(f"{prefix}{connector}{type_icon} {node['name']}{freq_str}")

    # Print children
    new_prefix = prefix + ("    " if is_last else "\u2502   ")
    for i, child in enumerate(node["children"]):
        print_text_tree(child, new_prefix, i == len(node["children"]) - 1, show_freq)


def generate_html(node: dict, output_path: Path) -> None:
    """Generate an interactive HTML visualization."""

    def node_to_html(n: dict, depth: int = 0) -> str:
        """Convert node to HTML list item."""
        indent = "  " * depth

        if n["name"] == "Concepts":
            # Root node - just render children
            children_html = "\n".join(node_to_html(c, depth) for c in n["children"])
            return f'<ul class="tree">\n{children_html}\n</ul>'

        # Determine node class
        node_class = "branch" if n["children"] else "leaf"
        freq_badge = f'<span class="freq">{n["freq"]}</span>' if n["freq"] > 0 else ""

        # Context tooltip
        context = html.escape(n.get("context", ""))
        context_attr = f' title="{context[:200]}..."' if len(context) > 200 else f' title="{context}"'

        if n["children"]:
            children_html = "\n".join(node_to_html(c, depth + 1) for c in n["children"])
            return f'''{indent}<li class="{node_class}"{context_attr}>
{indent}  <span class="node-name">{html.escape(n["name"])}</span>{freq_badge}
{indent}  <ul>
{children_html}
{indent}  </ul>
{indent}</li>'''
        else:
            return f'{indent}<li class="{node_class}"{context_attr}><span class="node-name">{html.escape(n["name"])}</span>{freq_badge}</li>'

    tree_html = node_to_html(node)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Concept Tree Visualization</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4a90d9;
            padding-bottom: 10px;
        }}
        .tree {{
            list-style: none;
            padding-left: 0;
        }}
        .tree ul {{
            list-style: none;
            padding-left: 25px;
            border-left: 1px dashed #ccc;
            margin-left: 10px;
        }}
        .tree li {{
            margin: 8px 0;
            padding: 5px 10px;
            position: relative;
            cursor: pointer;
            border-radius: 4px;
            transition: background 0.2s;
        }}
        .tree li:hover {{
            background: #e8f4fc;
        }}
        .tree li.branch > .node-name {{
            font-weight: bold;
            color: #2c5282;
        }}
        .tree li.branch > .node-name::before {{
            content: "\\25BC ";
            font-size: 0.7em;
            color: #4a90d9;
        }}
        .tree li.leaf > .node-name {{
            color: #555;
        }}
        .tree li.leaf > .node-name::before {{
            content: "\\25CB ";
            font-size: 0.8em;
            color: #888;
        }}
        .freq {{
            background: #4a90d9;
            color: white;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.75em;
            margin-left: 8px;
        }}
        .collapsed > ul {{
            display: none;
        }}
        .collapsed > .node-name::before {{
            content: "\\25B6 " !important;
        }}
        .stats {{
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats span {{
            margin-right: 20px;
            color: #666;
        }}
        .stats strong {{
            color: #333;
        }}
    </style>
</head>
<body>
    <h1>Concept Tree Visualization</h1>
    <div class="stats">
        <span>Branches: <strong id="branch-count">0</strong></span>
        <span>Leaves: <strong id="leaf-count">0</strong></span>
        <span>Total concepts: <strong id="total-count">0</strong></span>
    </div>
    {tree_html}
    <script>
        // Toggle collapse on branch click
        document.querySelectorAll('.branch').forEach(function(el) {{
            el.addEventListener('click', function(e) {{
                if (e.target.closest('ul') === el.querySelector('ul')) return;
                el.classList.toggle('collapsed');
                e.stopPropagation();
            }});
        }});

        // Count stats
        document.getElementById('branch-count').textContent = document.querySelectorAll('.branch').length;
        document.getElementById('leaf-count').textContent = document.querySelectorAll('.leaf').length;
        document.getElementById('total-count').textContent =
            document.querySelectorAll('.branch').length + document.querySelectorAll('.leaf').length;
    </script>
</body>
</html>'''

    output_path.write_text(html_content, encoding="utf-8")
    print(f"HTML visualization saved to: {output_path}")


def generate_graphviz(node: dict, output_path: Path) -> None:
    """Generate a Graphviz DOT file and render to PNG."""
    try:
        import subprocess

        # Check if graphviz is installed
        result = subprocess.run(["dot", "-V"], capture_output=True)
        if result.returncode != 0:
            print("Error: Graphviz not installed. Install with: brew install graphviz")
            sys.exit(1)
    except FileNotFoundError:
        print("Error: Graphviz not installed. Install with: brew install graphviz")
        sys.exit(1)

    dot_lines = ['digraph ConceptTree {']
    dot_lines.append('    rankdir=LR;')
    dot_lines.append('    node [shape=box, style=rounded, fontname="Helvetica"];')
    dot_lines.append('    edge [color="#888888"];')

    node_id = [0]

    def add_node(n: dict, parent_id: str | None = None) -> str:
        """Add node to DOT graph."""
        current_id = f"n{node_id[0]}"
        node_id[0] += 1

        # Skip root
        if n["name"] == "Concepts":
            for child in n["children"]:
                add_node(child, None)
            return current_id

        # Node styling
        if n["children"]:
            style = 'fillcolor="#4a90d9", fontcolor="white", style="rounded,filled"'
        else:
            style = 'fillcolor="#e8f4fc", style="rounded,filled"'

        label = f'{n["name"]}\\n({n["freq"]})'
        dot_lines.append(f'    {current_id} [label="{label}", {style}];')

        if parent_id:
            dot_lines.append(f'    {parent_id} -> {current_id};')

        for child in n["children"]:
            add_node(child, current_id)

        return current_id

    add_node(node)
    dot_lines.append('}')

    dot_content = '\n'.join(dot_lines)
    dot_path = output_path.with_suffix('.dot')
    dot_path.write_text(dot_content, encoding="utf-8")

    # Render to PNG
    png_path = output_path.with_suffix('.png')
    subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)

    print(f"Graphviz DOT saved to: {dot_path}")
    print(f"PNG image saved to: {png_path}")


def print_stats(node: dict) -> None:
    """Print statistics about the concept tree."""
    branches = 0
    leaves = 0
    max_freq = 0
    top_concepts = []

    def count(n: dict):
        nonlocal branches, leaves, max_freq
        if n["name"] == "Concepts":
            for c in n["children"]:
                count(c)
            return

        if n["children"]:
            branches += 1
        else:
            leaves += 1

        if n["freq"] > 0:
            top_concepts.append((n["name"], n["freq"]))
        max_freq = max(max_freq, n["freq"])

        for c in n["children"]:
            count(c)

    count(node)
    top_concepts.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 50)
    print("CONCEPT TREE STATISTICS")
    print("=" * 50)
    print(f"Total branches (heads):  {branches}")
    print(f"Total leaves (concepts): {leaves}")
    print(f"Total nodes:             {branches + leaves}")
    print(f"Max frequency:           {max_freq}")
    print("\nTop concepts by frequency:")
    for name, freq in top_concepts[:10]:
        bar = "\u2588" * freq
        print(f"  {name:25} {freq:2} {bar}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CiteSpace concept tree XML output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py concept_tree.xml              # Text tree view
  python visualize.py concept_tree.xml --html       # HTML interactive view
  python visualize.py concept_tree.xml --graph      # Graphviz PNG
  python visualize.py concept_tree.xml --stats      # Show statistics only
        """
    )
    parser.add_argument("xml_file", type=Path, help="Path to TreeML XML file")
    parser.add_argument("--html", action="store_true", help="Generate interactive HTML")
    parser.add_argument("--graph", action="store_true", help="Generate Graphviz visualization")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--no-freq", action="store_true", help="Hide frequency numbers")
    parser.add_argument("-o", "--output", type=Path, help="Output file path")

    args = parser.parse_args()

    if not args.xml_file.exists():
        print(f"Error: File not found: {args.xml_file}")
        sys.exit(1)

    # Parse the XML
    tree_data = parse_tree(args.xml_file)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.xml_file.with_suffix("")

    # Generate requested output
    if args.stats:
        print_stats(tree_data)
    elif args.html:
        html_path = output_path.with_suffix(".html")
        generate_html(tree_data, html_path)
        print_stats(tree_data)
    elif args.graph:
        generate_graphviz(tree_data, output_path)
        print_stats(tree_data)
    else:
        # Default: text tree
        print("\nCONCEPT TREE")
        print("=" * 50)
        print_text_tree(tree_data, show_freq=not args.no_freq)
        print_stats(tree_data)


if __name__ == "__main__":
    main()
