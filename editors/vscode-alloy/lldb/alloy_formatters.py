# LLDB formatters for Alloy native binaries (checked builds carry DWARF).
# Load in an LLDB session or CodeLLDB's "initCommands":
#     command script import path/to/alloy_formatters.py
#
# What they add on top of the plain DWARF view:
#   - slices show their length and preview u8 content as text
#   - heap arrays ('*[T]', a data pointer with the length stored at -8)
#     show their length instead of a bare pointer
#   - interface objects resolve the identity pointer to the implementing
#     type's symbol ('alloy.type.<Name>.<id>') and show <Name>

import lldb
import re


def slice_summary(value, _internal):
    try:
        data = value.GetChildMemberWithName("data")
        length_member = value.GetChildMemberWithName("length")
        if not data.IsValid() or not length_member.IsValid():
            return None
        length = length_member.GetValueAsUnsigned()
        base = data.GetValueAsUnsigned()
        if base == 0:
            return "empty slice"
        process = value.GetProcess()
        preview = ""
        element = data.GetType().GetPointeeType()
        if element.IsValid() and element.GetName() == "u8" and length > 0:
            error = lldb.SBError()
            raw = process.ReadMemory(base, min(length, 64), error)
            if error.Success():
                text = raw.decode("utf-8", "replace")
                preview = ' "%s"' % text
        return "len=%d%s" % (length, preview)
    except Exception:
        return None


def heap_array_summary(value, _internal):
    try:
        base = value.GetValueAsUnsigned()
        if base == 0:
            return "moved"
        process = value.GetProcess()
        error = lldb.SBError()
        length = process.ReadUnsignedFromMemory(base - 8, 8, error)
        if not error.Success():
            return None
        preview = ""
        element = value.GetType().GetTypedefedType().GetPointeeType()
        if element.IsValid() and element.GetName() == "u8" and length > 0:
            raw = process.ReadMemory(base, min(length, 64), error)
            if error.Success():
                preview = ' "%s"' % raw.decode("utf-8", "replace")
        return "len=%d%s" % (length, preview)
    except Exception:
        return None


def interface_summary(value, _internal):
    try:
        identity = value.GetChildMemberWithName("identity")
        data = value.GetChildMemberWithName("data")
        # only the interface fat pair carries exactly these members
        if not identity.IsValid() or not data.IsValid():
            return None
        address = identity.GetValueAsUnsigned()
        if address == 0:
            return "empty"
        target = value.GetTarget()
        resolved = target.ResolveLoadAddress(address)
        symbol = resolved.GetSymbol()
        if symbol and symbol.GetName():
            match = re.match(r"alloy\.type\.(.+)\.\d+$", symbol.GetName())
            if match:
                return "impl %s" % match.group(1)
            return symbol.GetName()
        return None
    except Exception:
        return None


def __lldb_init_module(debugger, _internal):
    category = debugger.GetDefaultCategory()
    module = "alloy_formatters"
    debugger.HandleCommand(
        'type summary add -x "^slice$" -F %s.slice_summary' % module
    )
    debugger.HandleCommand(
        'type summary add -x "^\\*\\[.*\\]$" -F %s.heap_array_summary' % module
    )
    # interface objects are 128-bit structures with data/identity members;
    # register per known interface name lazily is impossible, so match any
    # structure carrying an 'identity' member via the callback returning
    # None for non-matches
    debugger.HandleCommand(
        'type summary add -x "^[A-Z][A-Za-z0-9_]*$" -F %s.interface_summary' % module
    )
    _ = category
