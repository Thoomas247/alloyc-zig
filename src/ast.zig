//! Abstract syntax tree for the Alloy language, mirroring section 2
//! (Syntactic Grammar) of LANGUAGE_SPEC.md. All nodes are allocated in the
//! owning Ast's arena and reference tokens for names and source locations.

const std = @import("std");
const Token = @import("tokenizer.zig").Token;

pub const Ast = struct {
    arena: std.heap.ArenaAllocator,
    module: Module,

    pub fn init(backing_allocator: std.mem.Allocator) Ast {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .module = .{ .imports = &.{}, .definitions = &.{} },
        };
    }

    pub fn deinit(ast: *Ast) void {
        ast.arena.deinit();
    }
};

pub const Module = struct {
    imports: []const Import,
    definitions: []const Definition,
};

pub const Import = struct {
    path: []const Token,
    alias: ?Token,
};

pub const Visibility = enum { private, public, exported };

pub const Definition = struct {
    visibility: Visibility,
    kind: union(enum) {
        type_def: TypeDef,
        fn_def: FnDef,
        extern_def: ExternDef,
        interface_def: InterfaceDef,
        macro_def: MacroDef,
    },
};

pub const TypeDef = struct {
    name: Token,
    type_parameters: []const TypeParameter,
    interfaces: []const InterfaceMarker,
    base: *const TypeExpression,
};

/// A conformance marker ('type V<T> : Iterable<T, VectorCursor<T>>') or a
/// generic constraint ('It: Iterator<T>'): an interface name plus the type
/// arguments binding the interface's own type parameters (section 6.2).
pub const InterfaceMarker = struct {
    name: Token,
    type_arguments: []const *const TypeExpression,
};

pub const TypeParameter = struct {
    name: Token,
    constraint: ?InterfaceMarker,
};

pub const FnDef = struct {
    // 'fn Vector::empty(...)' associates the function with the named
    // type's namespace instead of the module's flat namespace
    qualifier: ?Token = null,
    name: Token,
    type_parameters: []const TypeParameter,
    function: Function,
};

pub const Function = struct {
    parameters: []const Parameter,
    return_type: ?*const TypeExpression,
    body: *const Statement,
};

pub const Parameter = struct {
    is_self: bool,
    name: Token,
    parameter_type: *const TypeExpression,
};

pub const ExternDef = struct {
    name: Token,
    parameters: []const Parameter,
    variadic: bool,
    return_type: ?*const TypeExpression,
};

pub const InterfaceDef = struct {
    name: Token,
    type_parameters: []const TypeParameter,
    functions: []const InterfaceFn,
};

// an interface function declares its receiver indirection ('self: &',
// 'self: &var', 'self: *', 'self: *var') as a nameless first parameter;
// 'parameters' holds the ordinary ones after it (section 6.2)
pub const InterfaceFn = struct {
    name: Token,
    receiver: TypeModifier,
    receiver_token: Token,
    parameters: []const Parameter,
    return_type: ?*const TypeExpression,
};

pub const MacroDef = struct {
    name: Token,
    parameters: []const MacroParameter,
    // the declared result type, required: nothing is inferred from the
    // body; may be a comptime type ('#Type', '&[#Type]') (section 7.3)
    return_type: *const TypeExpression,
    // null marks a declaration-only macro ('macro type_of(value);'):
    // implemented by the compiler, like an interface's functions
    body: ?*const Statement,
};

// a macro parameter's type is optional on declaration-only macros
pub const MacroParameter = struct {
    name: Token,
    parameter_type: ?*const TypeExpression,
};

pub const TypeModifier = enum {
    pointer,
    pointer_var,
    reference,
    reference_var,

    // the source spelling, for diagnostics
    pub fn lexeme(modifier: TypeModifier) []const u8 {
        return switch (modifier) {
            .pointer => "*",
            .pointer_var => "*var",
            .reference => "&",
            .reference_var => "&var",
        };
    }
};

pub const TypeExpression = union(enum) {
    modified: struct { modifier: TypeModifier, child: *const TypeExpression },
    named: NamedType,
    struct_type: []const StructMember,
    enum_type: []const EnumMember,
    array: struct { element: *const TypeExpression, length: ?Token },
    function: struct {
        parameter_types: []const *const TypeExpression,
        return_type: ?*const TypeExpression,
    },
    comptime_type: *const Expression,
    // '#Type' written as a type: the comptime descriptor itself, valid
    // only in a macro signature (section 4.4)
    type_description: Token,
};

pub const NamedType = struct {
    path: []const Token,
    type_arguments: []const *const TypeExpression,
    // '::Variant' in an 'is' target: the enum is implied by the subject
    implied: bool = false,
};

pub const StructMember = struct {
    name: Token,
    member_type: *const TypeExpression,
};

pub const EnumMember = struct {
    name: Token,
    payload: ?*const TypeExpression,
};

pub const Statement = union(enum) {
    var_def: VarDef,
    block: []const *const Statement,
    break_stmt: struct { keyword: Token, value: ?*const Expression },
    continue_stmt: struct { keyword: Token },
    // 'yield value' produces the value of the innermost value-position
    // 'if' or 'match' (section 5.3)
    yield_stmt: struct { keyword: Token, value: *const Expression },
    return_stmt: struct { keyword: Token, value: ?*const Expression },
    assign: struct {
        target: *const Expression,
        operator: Token,
        value: *const Expression,
    },
    expression: *const Expression,
};

pub const VarDef = struct {
    mutable: bool,
    name: Token,
    declared_type: ?*const TypeExpression,
    value: *const Expression,
};

pub const Expression = union(enum) {
    integer_literal: Token,
    float_literal: Token,
    string_literal: Token,
    character_literal: Token,
    bool_literal: struct { token: Token, value: bool },
    path: []const Token,
    // '::Variant': an enum variant whose enum type is implied from context
    implied_variant: Token,
    binary: struct {
        operator: Token,
        left: *const Expression,
        right: *const Expression,
    },
    // 'mutable' marks the '&var' form of the borrow operator (section 5.2)
    unary: struct { operator: Token, operand: *const Expression, mutable: bool = false },
    cast: struct {
        operator: Token,
        operand: *const Expression,
        target: *const TypeExpression,
        // 'x is ::Some |v|': the capture binds the payload inline, valid
        // only as a direct '&&' conjunct of an if or while condition
        // (section 4.2); always null for 'as' and 'to'
        capture: ?Capture = null,
    },
    call: struct {
        callee: *const Expression,
        type_arguments: []const *const TypeExpression,
        arguments: []const *const Expression,
    },
    member: struct { object: *const Expression, name: Token },
    index: struct { object: *const Expression, subscript: *const Expression },
    // 'arr[start..end]' borrows a slice viewing the element range
    // start..end-1 in place (section 4.2); a null start means 0
    subslice: struct {
        object: *const Expression,
        operator: Token,
        start: ?*const Expression,
        end: *const Expression,
    },
    // a null path is an anonymous structural literal ('{ .x = 1 }'); a
    // named literal may be module-qualified ('liba::Pair { ... }') and may
    // bind a generic type's parameters explicitly ('Vector<T> { ... }')
    struct_init: struct { path: ?[]const Token, type_arguments: []const *const TypeExpression = &.{}, members: []const MemberInit },
    array_literal: []const *const Expression,
    // an inline layout written in place under '#' ('#struct { id: u32 }',
    // '#enum { A, B: u8 }'): its '#Type', compile time only (section 4.4)
    type_literal: *const TypeExpression,
    array_fill: struct { value: *const Expression, count: *const Expression },
    // '[start..end]' integer range generator; a null start means 0
    array_range: struct { operator: Token, start: ?*const Expression, end: *const Expression },
    if_expr: IfExpression,
    while_expr: WhileExpression,
    for_expr: ForExpression,
    match_expr: MatchExpression,
    lambda: Lambda,
    comptime_expr: *const Expression,
    grouped: *const Expression,
};

pub const MemberInit = struct {
    name: Token,
    value: *const Expression,
};

// '|x|' copies, '|&x|' / '|&var x|' borrow, '|move x|' (modifier
// 'pointer') takes ownership; captures carry no annotation (section 3.1)
pub const Capture = struct {
    modifier: ?TypeModifier,
    name: Token,
};

pub const IfExpression = struct {
    condition: *const Expression,
    then_branch: *const Statement,
    else_branch: ?*const Statement,
};

pub const WhileExpression = struct {
    condition: *const Expression,
    body: *const Statement,
    else_branch: ?*const Statement,
};

pub const ForExpression = struct {
    subjects: []const *const Expression,
    captures: []const Capture,
    body: *const Statement,
    else_branch: ?*const Statement,
};

pub const MatchExpression = struct {
    subject: *const Expression,
    arms: []const MatchArm,
    else_branch: ?*const Statement,
};

pub const MatchArm = struct {
    pattern: ?*const Expression,
    capture: ?Capture,
    body: *const Statement,
};

pub const Lambda = struct {
    captures: []const Capture,
    function: Function,
};
