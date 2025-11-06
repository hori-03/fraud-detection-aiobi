"""
utils package initializer.

An empty file is sufficient to ensure Python treats `utils` as a regular
package (not a namespace package or a conflicting module). This helps
imports such as `from utils.column_matcher import ColumnMatcher` work
consistently inside Docker/Railway where import resolution can differ.

Created automatically to fix ModuleNotFoundError: 'utils' is not a package
when the container booted.
"""

# package marker
