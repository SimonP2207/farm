"""
Any and all decorator definitions should be placed in this module
"""


def docstring_parameter(*sub):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec
