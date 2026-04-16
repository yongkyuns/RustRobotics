project = "Rust Robotics"
copyright = "2026, Yongkyun Shin"
author = "Yongkyun Shin"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "Rust Robotics documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["sim_embed.js"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]

html_theme_options = {
    "source_repository": "https://github.com/yongkyuns/RustRobotics/",
    "source_branch": "master",
    "source_directory": "site_docs/",
}
