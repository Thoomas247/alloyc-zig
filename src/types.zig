//! Type representation for the type checking stage, mirroring section 3 of
//! LANGUAGE_SPEC.md. Types are immutable values allocated in the checker's
//! arena and compared structurally.

const std = @import("std");
const ast = @import("ast.zig");

pub const Primitive = enum {
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    bool,

    pub fn width(primitive: Primitive) u8 {
        return switch (primitive) {
            .u8, .i8, .bool => 1,
            .u16, .i16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
        };
    }

    pub fn isFloat(primitive: Primitive) bool {
        return primitive == .f32 or primitive == .f64;
    }

    pub fn isSigned(primitive: Primitive) bool {
        return switch (primitive) {
            .i8, .i16, .i32, .i64 => true,
            else => false,
        };
    }

    pub fn isUnsigned(primitive: Primitive) bool {
        return switch (primitive) {
            .u8, .u16, .u32, .u64 => true,
            else => false,
        };
    }

    pub fn isNumeric(primitive: Primitive) bool {
        return primitive != .bool;
    }
};

/// Nominal runtime identity of a struct value (section 5.2): the declaring
/// definition plus its view, so same-named types from different libraries
/// never confuse dispatch, downcasts, or match arms.
pub const TypeIdentity = struct {
    definition: *const ast.Definition,
    view_index: usize,
};

/// A serialization shape for 'as' reinterpretation (section 3.5): the
/// checker flattens a type's C-compatible layout (section 3.9) into byte
/// offsets so the interpreter can reinterpret values without re-deriving
/// type structure. Pointer-bearing types have no shape.
pub const Shape = union(enum) {
    primitive: Primitive,
    record: Record,
    array: Array,
    tagged: Tagged,

    pub const Record = struct {
        size: u64,
        // the declared name carried into reconstructed values; empty for
        // structural layouts
        name: []const u8,
        // the nominal identity reconstructed values receive, when declared
        identity: ?TypeIdentity = null,
        fields: []const Field,
    };

    pub const Field = struct {
        name: []const u8,
        offset: u64,
        shape: *const Shape,
    };

    pub const Array = struct {
        size: u64,
        stride: u64,
        count: u64,
        element: *const Shape,
    };

    pub const Tagged = struct {
        size: u64,
        tag_size: u64,
        payload_offset: u64,
        variants: []const Variant,
    };

    pub const Variant = struct {
        name: []const u8,
        payload: ?*const Shape,
    };

    pub fn byteSize(shape: *const Shape) u64 {
        return switch (shape.*) {
            .primitive => |primitive| primitive.width(),
            .record => |record| record.size,
            .array => |array| array.size,
            .tagged => |tagged| tagged.size,
        };
    }
};

/// The two shapes of one 'as' reinterpretation, recorded per cast site.
pub const CastShapes = struct {
    source: *const Shape,
    target: *const Shape,
};

pub const Type = union(enum) {
    // produced by a reported error; compatible with everything so one
    // mistake does not cascade into follow-up diagnostics
    unknown,
    // the absence of a value: a function without a return type
    void_type,
    // literals without a contextual type (section 3.3 rules 2 and 3)
    untyped_integer,
    untyped_float,
    primitive: Primitive,
    pointer: Indirection,
    reference: Indirection,
    heap_array: Indirection,
    slice: Indirection,
    fixed_array: FixedArray,
    function: Function,
    declared: Declared,
    structural: []const Field,
    type_parameter: TypeParameter,
    // an interface used as a type; only valid behind an indirection, where
    // it forms an interface object (section 5.2)
    interface: Interface,
    // an inline 'enum { ... }' in a type position; compared structurally
    // against other enum types (section 3.3)
    inline_enum: InlineEnum,
    // a synthesised enum ('type T = #...', section 3.4): a resolved variant
    // list with no syntax behind it
    structural_enum: []const EnumVariant,

    pub const Indirection = struct {
        mutable: bool,
        child: *const Type,
    };

    pub const FixedArray = struct {
        element: *const Type,
        length: u64,
    };

    pub const Function = struct {
        parameter_types: []const *const Type,
        return_type: *const Type,
    };

    /// An instance of a named 'type X<...> = ...' definition.
    pub const Declared = struct {
        definition: *const ast.Definition,
        view_index: usize,
        name: []const u8,
        arguments: []const *const Type,
    };

    pub const Field = struct {
        name: []const u8,
        field_type: *const Type,
    };

    /// An unbound generic type parameter inside a generic definition's body.
    pub const TypeParameter = struct {
        name: []const u8,
        constraint: ?Interface,
    };

    /// One resolved type argument of a generic call, recorded per call site
    /// so later stages can resolve type parameters without re-inference.
    pub const Binding = struct {
        name: []const u8,
        bound: *const Type,
    };

    pub const Interface = struct {
        definition: *const ast.Definition,
        view_index: usize,
        name: []const u8,
    };

    pub const InlineEnum = struct {
        members: []const ast.EnumMember,
        view_index: usize,
    };

    pub const EnumVariant = struct {
        name: []const u8,
        payload: ?*const Type,
    };

    pub fn eql(left: *const Type, right: *const Type) bool {
        if (left == right) return true;
        if (std.meta.activeTag(left.*) != std.meta.activeTag(right.*)) return false;
        return switch (left.*) {
            .unknown, .void_type, .untyped_integer, .untyped_float => true,
            .primitive => |primitive| primitive == right.primitive,
            .pointer => |indirection| indirection.mutable == right.pointer.mutable and indirection.child.eql(right.pointer.child),
            .reference => |indirection| indirection.mutable == right.reference.mutable and indirection.child.eql(right.reference.child),
            .heap_array => |indirection| indirection.mutable == right.heap_array.mutable and indirection.child.eql(right.heap_array.child),
            .slice => |indirection| indirection.mutable == right.slice.mutable and indirection.child.eql(right.slice.child),
            .fixed_array => |array| array.length == right.fixed_array.length and array.element.eql(right.fixed_array.element),
            .function => |function| eqlFunction(function, right.function),
            .declared => |declared| eqlDeclared(declared, right.declared),
            .structural => |fields| eqlFields(fields, right.structural),
            .type_parameter => |parameter| std.mem.eql(u8, parameter.name, right.type_parameter.name),
            .interface => |interface| interface.definition == right.interface.definition,
            // identity here is the same syntactic occurrence; structural
            // compatibility between distinct occurrences lives in coercion
            .inline_enum => |inline_enum| inline_enum.members.ptr == right.inline_enum.members.ptr,
            .structural_enum => |variants| variants.ptr == right.structural_enum.ptr,
        };
    }

    fn eqlFunction(left: Function, right: Function) bool {
        if (left.parameter_types.len != right.parameter_types.len) return false;
        for (left.parameter_types, right.parameter_types) |left_parameter, right_parameter| {
            if (!left_parameter.eql(right_parameter)) return false;
        }
        return left.return_type.eql(right.return_type);
    }

    fn eqlDeclared(left: Declared, right: Declared) bool {
        if (left.definition != right.definition) return false;
        if (left.arguments.len != right.arguments.len) return false;
        for (left.arguments, right.arguments) |left_argument, right_argument| {
            if (!left_argument.eql(right_argument)) return false;
        }
        return true;
    }

    fn eqlFields(left: []const Field, right: []const Field) bool {
        if (left.len != right.len) return false;
        for (left, right) |left_field, right_field| {
            if (!std.mem.eql(u8, left_field.name, right_field.name)) return false;
            if (!left_field.field_type.eql(right_field.field_type)) return false;
        }
        return true;
    }

    pub fn isNumeric(self: *const Type) bool {
        return switch (self.*) {
            .unknown, .untyped_integer, .untyped_float => true,
            .primitive => |primitive| primitive.isNumeric(),
            else => false,
        };
    }

    pub fn isInteger(self: *const Type) bool {
        return switch (self.*) {
            .unknown, .untyped_integer => true,
            .primitive => |primitive| primitive.isNumeric() and !primitive.isFloat(),
            else => false,
        };
    }

    pub fn isBool(self: *const Type) bool {
        return switch (self.*) {
            .unknown => true,
            .primitive => |primitive| primitive == .bool,
            else => false,
        };
    }

    /// Renders the type for diagnostics ('*var Packet', '&[u8]', 'Vec<u32>').
    pub fn render(self: *const Type, allocator: std.mem.Allocator) std.mem.Allocator.Error![]const u8 {
        var text: std.Io.Writer.Allocating = .init(allocator);
        defer text.deinit();
        self.write(&text.writer) catch return error.OutOfMemory;
        return allocator.dupe(u8, text.writer.buffered());
    }

    fn write(self: *const Type, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self.*) {
            .unknown => try writer.writeAll("<error>"),
            .void_type => try writer.writeAll("void"),
            .untyped_integer => try writer.writeAll("<integer literal>"),
            .untyped_float => try writer.writeAll("<float literal>"),
            .primitive => |primitive| try writer.writeAll(@tagName(primitive)),
            .pointer => |indirection| {
                try writer.writeAll(if (indirection.mutable) "*var " else "*");
                try indirection.child.write(writer);
            },
            .reference => |indirection| {
                try writer.writeAll(if (indirection.mutable) "&var " else "&");
                try indirection.child.write(writer);
            },
            .heap_array => |indirection| {
                try writer.writeAll(if (indirection.mutable) "*var [" else "*[");
                try indirection.child.write(writer);
                try writer.writeAll("]");
            },
            .slice => |indirection| {
                try writer.writeAll(if (indirection.mutable) "&var [" else "&[");
                try indirection.child.write(writer);
                try writer.writeAll("]");
            },
            .fixed_array => |array| {
                try writer.writeAll("[");
                try array.element.write(writer);
                try writer.print(" : {d}", .{array.length});
                try writer.writeAll("]");
            },
            .function => |function| {
                try writer.writeAll("(");
                for (function.parameter_types, 0..) |parameter_type, index| {
                    if (index != 0) try writer.writeAll(", ");
                    try parameter_type.write(writer);
                }
                try writer.writeAll(") -> ");
                try function.return_type.write(writer);
            },
            .declared => |declared| {
                try writer.writeAll(declared.name);
                if (declared.arguments.len != 0) {
                    try writer.writeAll("<");
                    for (declared.arguments, 0..) |argument, index| {
                        if (index != 0) try writer.writeAll(", ");
                        try argument.write(writer);
                    }
                    try writer.writeAll(">");
                }
            },
            .structural => |fields| {
                try writer.writeAll("struct { ");
                for (fields, 0..) |field, index| {
                    if (index != 0) try writer.writeAll(", ");
                    try writer.print("{s}: ", .{field.name});
                    try field.field_type.write(writer);
                }
                try writer.writeAll(" }");
            },
            .type_parameter => |parameter| try writer.writeAll(parameter.name),
            .interface => |interface| try writer.writeAll(interface.name),
            .inline_enum => try writer.writeAll("enum { ... }"),
            .structural_enum => |variants| {
                try writer.writeAll("enum { ");
                for (variants, 0..) |variant, index| {
                    if (index != 0) try writer.writeAll(", ");
                    try writer.writeAll(variant.name);
                    if (variant.payload) |payload| {
                        try writer.writeAll(": ");
                        try payload.write(writer);
                    }
                }
                try writer.writeAll(" }");
            },
        }
    }
};
