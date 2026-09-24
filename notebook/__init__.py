"""Notebook folder as a package.

Allows importing the module code under ``notebook/`` (e.g.
``notebook.models.relationship.model``) with the repository root on
``sys.path``.  The Jupyter notebooks themselves are unaffected: they
keep resolving ``models`` via their own working directory.
"""