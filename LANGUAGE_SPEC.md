# Alloy Language — Formal Language Specification

## Table of Contents

1. [Lexical Grammar](#1-lexical-grammar)
2. [Syntactic Grammar](#2-syntactic-grammar)
3. [Type System & Static Semantics](#3-type-system--static-semantics)
4. [Execution Semantics](#4-execution-semantics)
5. [Standard Library & Primitives](#5-standard-library--primitives)
6. [Compile-Time Evaluation & Macros](#6-compile-time-evaluation--macros)

---

## 1. Lexical Grammar

### 1.1 Character Set

Source files are encoded in **UTF-8**. Identifiers and literal strings fully support Unicode character mappings.

### 1.2 Whitespace & Comments

```
whitespace     ::= ' ' | '\t' | '\n' | '\r' | '\v' | '\f'
line_comment   ::= '//' <any char except '\n'>*
block_comment  ::= '/*' ( block_comment | <any sequence not containing '/*' or '*/'> )* '*/'

```

Whitespace and comments are not significant to the grammar and are ignored between tokens. Newlines are ordinary whitespace; statements and declarations are terminated by the `;` separator (§2.1). A line comment ends at the newline.

Block comments nest arbitrarily. The compiler tracks the nesting depth, and a block comment is only considered terminated when all opened comment blocks are closed. An unterminated block comment is a compile-time error.

### 1.3 Identifiers

```
identifier     ::= ( letter | '_' ) ( letter | digit | '_' )*
letter         ::= [a-zA-Z] | <any non-ASCII UTF-8 sequence>
digit          ::= [0-9]

```

Any identifier that matches a reserved keyword (§1.4) is treated as that keyword and may not be used as a user-defined name.

Any non-ASCII UTF-8 byte sequence is currently identifier-legal in both start and continuation positions; restricting identifiers to the UAX #31 character categories is deferred to a later milestone.

### 1.4 Keywords

Reserved — cannot be used as identifiers:

```
import   as       extern   type     enum     struct
const    var      fn       if       else     while
for      match    break    yield    return   new     move
self     pub      exp      true     false    interface macro
is       to

```

### 1.5 Operators & Punctuation

Complete list of symbolic tokens:

| Category                | Symbols                                     |
| ----------------------- | ------------------------------------------- |
| Arithmetic              | `+` `-` `*` `/` `%`                         |
| Compound assign         | `+=` `-=` `*=` `/=` `%=` `<<=` `>>=` `&=` `\|=` `^=` |
| Comparison              | `==` `!=` `<` `<=` `>` `>=`                 |
| Logical                 | `&&` `\|\|` `!`                             |
| Bitwise                 | `&` `\|` `^` `~` `<<` `>>`                  |
| Assignment              | `=`                                         |
| Arrow                   | `->`                                        |
| Path separator          | `::`                                        |
| Member access           | `.`                                         |
| Range                   | `..`                                        |
| Spread / variadic       | `...`                                       |
| Type annotation         | `:`                                         |
| Separator               | `,` `;`                                     |
| Delimiters              | `(` `)` `{` `}` `[` `]`                     |
| Comptime / Macro prefix | `#`                                         |

When multiple symbolic tokens share a common prefix, the longest matching token is always chosen.

### 1.6 Literals

```
integer_literal  ::= decimal_literal | hex_literal | binary_literal | octal_literal
decimal_literal  ::= [0-9]+
hex_literal      ::= '0x' [0-9a-fA-F]+
binary_literal   ::= '0b' [01]+
octal_literal    ::= '0o' [0-7]+

float_literal    ::= [0-9]+ '.' [0-9]*
string_literal   ::= '"' ( escape_seq | <any char except '"' or '\n'> )* '"'
char_literal     ::= '\'' ( escape_seq | <any char except '\'' or '\n'> )+ '\''
escape_seq       ::= '\\' [nrt0\\'"xXuU] | '\\x' digit{2} | '\\u{' digit+ '}'

```

Integer literals support standard **decimal**, **hexadecimal** (`0x`), **binary** (`0b`), and **octal** (`0o`) radix representations.

String and character literals are single-line: a raw newline inside a literal is a compile-time error (the literal is reported as unterminated). A line break in string data is written with the `\n` escape. An escape introducer outside the recognized set is a compile-time error.

#### Escape Sequences

The compiler recognizes standard escape sequences within string and character literals:

- `\n` (line feed), `\r` (carriage return), `\t` (tab), `\0` (null byte)
- `\\` (backslash), `\"` (double quote), `\'` (single quote)
- `\xHH` (arbitrary 1-byte hex value)
- `\u{HHHH}` (arbitrary multi-byte Unicode scalar value)

#### String Literal Typing

A string literal has type `&[u8]`: a slice viewing the literal's bytes in
static program memory, valid for the whole program run. A string literal never
allocates; an owned heap copy is made explicitly (`new "text"`, type `*[u8]`).

#### Character Literal Typing

The specific underlying primitive type of a character literal dynamically matches its size in bytes:

- Single quotes can bind a single character or a sequence of sequential characters up to 8 bytes long.
- A standard 1-byte ASCII character literal (e.g., `'a'`) evaluates to a `u8`.
- A multi-character packed constant sequence (e.g., `'abcdefgh'`) spans 8 bytes and evaluates to a `u64`.
- Variable-width Unicode characters evaluate to the smallest unsigned primitive integer width capable of holding their entire byte representation (typically `u32` for standard individual scalar sequences).

---

## 2. Syntactic Grammar

### 2.1 Full EBNF

```ebnf
(* Top level *)
module          = { import_decl } { definition } EOF ;

import_decl     = "import" ident { "::" ident } [ "as" ident ] terminator ;

definition      = [ "pub" | "exp" ] ( type_def | fn_def | extern_def | interface_def | macro_def ) ;
type_def        = "type" ident [ "<" type_param { "," type_param } ">" ] [ ":" ident { "," ident } ] "=" base_type terminator ;
interface_def   = "interface" ident "{" { interface_fn } "}" ;
interface_fn    = "fn" ident "(" [ param { "," param } ] ")" [ "->" type ] terminator ;
fn_def          = "fn" [ ident "::" ] ident [ "<" type_param { "," type_param } ">" ] function ;
macro_def       = "macro" ident "(" [ param { "," param } ] ")" stmt_block ;
extern_def      = "extern" ident "(" extern_params ")" [ "->" type ] terminator ;
extern_params   = /* empty */
                | param { "," param } [ "," "..." ]
                | "..." ;

type_param      = ident [ ":" ident ] ;

(* Functions *)
function        = "(" [ param { "," param } ] ")" [ "->" type ] stmt_block ;
param           = [ "self" ] ident ":" type ;

(* Types *)
type            = type_modifier ( base_type | type ) ;
type_modifier   = /* none */ | "*" | "*" "var" | "&" | "&" "var" ;

base_type       = named_type
                | struct_type
                | enum_type
                | array_type
                | fn_type
                | comptime_expr ;

named_type      = ident { "::" ident } [ "<" type { "," type } ">" ] ;
struct_type     = "struct" "{" [ struct_member { "," struct_member } [ "," ] ] "}" ;
struct_member   = ident ":" type ;
enum_type       = "enum" "{" [ enum_member { "," enum_member } [ "," ] ] "}" ;
enum_member     = ident [ ":" type ] ;
array_type      = "[" type [ ":" integer_literal ] "]" ;
fn_type         = "(" [ type { "," type } ] ")" [ "->" type ] ;

(* Statements *)
statement       = var_def
                | stmt_block
                | if_expr
                | for_expr
                | while_expr
                | match_expr
                | break_stmt
                | yield_stmt
                | return_stmt
                | expr_stmt ;

var_def         = ( "var" | "const" ) ident [ ":" type ] "=" expression terminator ;
stmt_block      = "{" { statement } "}" ;
break_stmt      = "break" [ expression ] ( terminator | <block-like operand> ) ;
yield_stmt      = "yield" expression ( terminator | <block-like operand> ) ;
return_stmt     = "return" [ expression ] terminator ;
expr_stmt       = expression ( assign_op expression terminator | terminator ) ;

terminator      = ";" | <implicit before "}" / "else" / EOF> ;

assign_op       = "=" | "+=" | "-=" | "*=" | "/=" | "%="
                | "<<=" | ">>=" | "&=" | "|=" | "^=" ;

(* Expressions *)
expression      = cast_expr [ binary_op expression ] ;

cast_expr       = unary_expr { "is" ( implied_variant | named_type ) | "as" type | "to" type } ;

unary_expr      = unary_op unary_expr | postfix_expr ;
unary_op        = "-" | "~" | "!" | "&" | "new" | "move" ;

postfix_expr    = primary_expr { postfix_suffix } ;
postfix_suffix  = "(" [ expr { "," expr } ] ")"                            (* call *)
                | "<" type { "," type } ">" "(" [ expr { "," expr } ] ")"  (* generic call *)
                | "." ident                                                  (* member access *)
                | "[" expression "]"                                         (* array index *)
                | "[" [ expression ] ".." expression "]" ;                   (* subslice *)

primary_expr    = literal
                | identifier_expr
                | implied_variant
                | named_struct_init
                | anon_struct_init
                | "(" expression ")"
                | array_fill
                | array_range
                | array_literal
                | if_expr
                | for_expr
                | while_expr
                | match_expr
                | lambda_expr
                | comptime_expr ;

identifier_expr   = ident { "::" ident } ;
implied_variant   = "::" ident ;
literal           = integer_literal | float_literal | string_literal | char_literal | "true" | "false" ;

array_literal     = "[" expression { "," expression } "]" ;
array_fill        = "[" expression ":" expression "]" ;
array_range       = "[" [ expression ] ".." expression "]" ;

named_struct_init = ident { "::" ident } "{" [ member_init { "," member_init } ] "}" ;
anon_struct_init  = "{" [ member_init { "," member_init } ] "}" ;
member_init       = "." ident "=" expression ;

lambda_expr       = [ "|" [ capture { "," capture } ] "|" ] function ;
capture           = type_modifier ident
                  | ident ":" ( type | type_modifier ) ;

if_expr     = "if" "(" expression ")" [ "|" capture "|" ]
              statement [ "else" statement ] ;

for_expr    = "for" "(" expression { "," expression } ")"
              [ "|" capture { "," capture } "|" ]
              statement [ "else" statement ] ;

while_expr  = "while" "(" expression ")" statement [ "else" statement ] ;

match_expr  = "match" "(" expression ")" "{"
              { ( expression | "else" ) [ "|" capture "|" ] statement }
              "}" [ "else" statement ] ;

comptime_expr   = "#" postfix_expr ;

```

**Comptime prefix.** The `#` token marks any value-yielding expression for
compile-time evaluation (§6). It binds as a postfix expression: `#f(x)` is
`#(f(x))`, but `#a + b` is `(#a) + b` — parenthesise to mark a whole expression
(`#(a + b)`).

**Semicolon termination.** The `;` separator terminates statements and
declarations. Newlines are plain whitespace, so a statement may span any
number of lines and several statements may share one. A statement also
terminates implicitly before `}`, before `else`, and at end of file, which
permits forms like `{ yield 10 }` and `yield a else yield b` without a
trailing `;`. Redundant semicolons are permitted and parse as empty
statements.

**Array-fill count.** The count in `[value : count]` is any expression. When
the fill produces a stack-allocated fixed array (`[T : N]`), the count must be
compile-time evaluatable — verified by a later compilation stage, not by the
grammar. A runtime count is valid only for heap allocation (`new [value : count]`,
§4.2).

**Range generators.** `[start..end]` generates the array of consecutive
integers from `start` (inclusive) to `end` (exclusive): `[0..5]` is
`[0, 1, 2, 3, 4]`. Omitting the start (`[..end]`) starts at `0`. Both bounds
must be integers (§3.3 rules apply; untyped literal bounds default to `i32`),
the bounds must agree on one integer type, and a literal `end` smaller than a
literal `start` is a compile-time error. Like the array fill, literal bounds
produce a stack array `[T : end - start]`; runtime bounds require `new`
(yielding `*[T]`) — except as a `for` subject, where the range never
materializes and runtime bounds are always valid (§4.3).

**Subslicing.** `arr[start..end]` borrows a slice (`&[T]`) viewing the
subject's elements `start` (inclusive) to `end` (exclusive) **in place** — no
copy is made. Omitting the start (`arr[..end]`) starts at `0`. The subject may
be any array form (`[T : N]`, `&[T]`, `*[T]`); the view is mutable when the
subject location is. Bounds must satisfy `start <= end <= length`; a violation
is a runtime fault in checked builds. Like any reference, the view is
invalidated when the subject is dropped, moved, or reallocated.

**`break` and `yield` operand termination.** When a `break`'s or `yield`'s
operand is a block-like expression (`if` / `while` / `for` / `match`), that
operand is self-terminating: `break if (c) yield a else yield b`.

**Capture typing.** A capture (`|a|`) binds a **deep copy** of the captured
value by default. An explicit `: type` annotation overrides this, and the
annotation may also be a bare type modifier as shorthand — the captured value's
own type is then inferred, so `|a: &|` reads "whatever `a` is, I want a
reference to it":

- `|a|` — deep copy (default)
- `|a: T|` — deep copy, type stated explicitly
- `|a: &T|` / `|a: &|` — immutable reference to the value in place
- `|a: &var T|` / `|a: &var|` — mutable reference; the subject must be mutable
- `|a: *|` / `|a: *var|` — owning capture: takes ownership of the captured
  value, exactly like `move` (§4.2). The pointer transfers into `a`, and the
  source — the enum subject of an `is` test, or the captured variable of a
  lambda — is invalid (moved-from) after the construct. Valid only when the
  captured value's type is itself a pointer (`*T` / `*var T` / `*[T]`), and
  the subject must be mutable.

Lambda capture lists additionally accept the prefix-modifier form (`|&var x|`).
Interface-object captures are the exception to the copy default: they always
bind by reference, mirroring the subject's indirection (§3.2).

### 2.2 Operator Precedence & Associativity

Higher number = tighter binding. All binary operators are **left-associative**.

| Precedence     | Operators         |
| -------------- | ----------------- |
| 100 (tightest) | `*` `/` `%`       |
| 90             | `+` `-`           |
| 80             | `<<` `>>`         |
| 70             | `<` `<=` `>` `>=` |
| 60             | `==` `!=`         |
| 50             | `&` (bitwise AND) |
| 40             | `^`               |
| 30             | `\|` (bitwise OR) |
| 20             | `&&`              |
| 10 (loosest)   | `\|\|`            |

Unary prefix operators (`~`, `!`, `&`, `new`, `move`) bind tighter than all binary operators and are **right-associative**. Postfix operators (call `()`, generic call `<>()`, member `.`, index `[]`) bind tighter than all unary prefix operators. The cast operators (`is`, `as`, `to`, §3.5) bind looser than unary prefix operators and tighter than all binary operators.

---

## 3. Type System & Static Semantics

### 3.1 Primitive Types

| Name   | Width   | Signed | Float |
| ------ | ------- | ------ | ----- |
| `u8`   | 1 byte  | No     | No    |
| `u16`  | 2 bytes | No     | No    |
| `u32`  | 4 bytes | No     | No    |
| `u64`  | 8 bytes | No     | No    |
| `i8`   | 1 byte  | Yes    | No    |
| `i16`  | 2 bytes | Yes    | No    |
| `i32`  | 4 bytes | Yes    | No    |
| `i64`  | 8 bytes | Yes    | No    |
| `f32`  | 4 bytes | —      | Yes   |
| `f64`  | 8 bytes | —      | Yes   |
| `bool` | 1 byte  | No     | No    |

Boolean literals are explicitly reserved via the language keywords `true` and `false`. Character literal primitive types scale automatically to fit their byte layout width (§1.6).

### 3.2 Composite & Derived Types

| Syntax                         | Kind                    | Notes                                                                      |
| ------------------------------ | ----------------------- | -------------------------------------------------------------------------- |
| `struct { f: T, ... }`         | Struct                  | Declaration-order members                                                  |
| `enum { A, B: T, ... }`        | Enum (sum type)         | Variants with optional payload                                             |
| `[T : N]` (N > 0)              | Fixed array             | Size is a compile-time integer literal allocated on stack/inline           |
| `&[T]`                         | Slice                   | Unmanaged view (fat pointer containing a raw pointer + a `u64` length)     |
| `*[T]`                         | Dynamically Sized Array | Managed heap pointer to runtime-sized memory layout handle                 |
| `(T1, T2) -> R`                | Function type           | First-class function value                                                 |
| `type X = BaseType`            | Named alias             | Nominally distinct; transparently assignable to its underlying type        |
| `*T`                           | Immutable pointer       | Managed heap-allocated instance                                            |
| `*var T`                       | Mutable pointer         | Managed mutable heap-allocated instance                                    |
| `&T`                           | Immutable reference     | Unmanaged borrowed reference                                               |
| `&var T`                       | Mutable reference       | Unmanaged mutable borrowed reference                                       |
| `&I` / `*I` (`I` an interface) | Interface object        | Dynamic-dispatch fat pointer: a data pointer plus a vtable pointer (§5.2). |

An **interface used as a type** (only behind an indirection — `&I`, `&var I`, `*I`, `*var I`) is an _interface object_: a fat pointer carrying the address of a value together with the vtable of the concrete type's interface implementation. A value of concrete type `T` is implicitly convertible to an interface object of `I` if and only if `T` declares `I` among its interface markers (`type T : I = ...`). The reverse conversion (interface object down to a concrete type) is expressed through the same constructs used for enum discrimination:

- **Exhaustive (`match`).** A `match` whose subject is an interface object accepts concrete-type names as arm patterns, with a payload capture that binds a reference to the concrete value:

```alloy
match (shape) {            // shape: &Shape
    Circle |c| { /* c: &Circle */ }
    Square |s| { /* s: &Square */ }
    else    { /* unhandled concrete type */ }
}
```

The capture's indirection mirrors the subject's: a `&Shape` subject yields `&Concrete` captures, a `&var Shape` subject yields `&var Concrete` captures, and similarly for `*` / `*var`.

- **Non-exhaustive (`if (… is Type)`).** The new `is` keyword tests whether an interface object's concrete type matches a target type, and — when paired with a typed capture — also produces the downcasted reference:

```alloy
if (shape is Circle) |c| { /* c: &Circle, only runs when shape's concrete is Circle */ }
```

`shape is Type` is a boolean expression in its own right; the `|c|` capture is optional. The `is` operator accepts two kinds of left operand:

1. **Interface object** — the right operand is a concrete type implementing the same interface, and the capture binds the downcasted value (above).
2. **Enum value** — the right operand is a variant of that enum's type, and `is` evaluates whether the enum currently holds that variant. The capture then binds the variant's **payload**, and is only permitted on variants that carry one:

```alloy
type SomeEnum = enum {
    ValueA: T,
    ValueB,
}

var val: SomeEnum = SomeEnum::ValueA(t)

if (val is SomeEnum::ValueA) |a: &T| { }   // a borrows the payload in place
if (val is SomeEnum::ValueA) |a: &| { }    // same, payload type inferred
if (val is SomeEnum::ValueA) |a: T| { }    // a is a copy, type stated
if (val is SomeEnum::ValueA) |a| { }       // a is a copy (default)
```

Capture binding follows the capture-typing rules (§2.1): a deep copy by default, a borrow when annotated with a reference modifier (`&` / `&var`). An owning capture (`*` / `*var`) takes the payload **out** of the enum:

```alloy
type Holder = enum {
    Boxed: *T,
    Empty,
}

var h: Holder = Holder::Boxed(new T {})

if (h is Holder::Boxed) |p: *| {
    // p owns the payload allocation
}
// h is invalid (moved-from) after the if, whether or not the branch ran
```

When the branch is taken, ownership of the payload transfers into the capture, the remainder of the enum value is dropped, and the subject binding is cleared — exactly like `move` (§4.2). When the branch is not taken, the subject value is reclaimed by its normal scope-end drop. In both cases the subject binding is treated as **moved-from after the `if`**; using it again is a use-after-move error. Owning captures are only valid when the payload type is itself a pointer (`*T` / `*var T` / `*[T]`) — only pointers are movable (§4.2) — and `&var` / `*` / `*var` captures all require the subject to be mutable.

#### Implied enum variants (`::Variant`)

A variant may be written without its enum name by prefixing it with `::`, which tells the compiler to infer the enum type. This works everywhere a variant is named: construction (`::Some(x)`, `::None`), `match` arm patterns, and `is` targets. A bare variant name without the `::` prefix is never valid.

The inference resolves in two steps, and is only valid when **exactly one candidate** matches in the given context:

1. When a contextual type is available (a declared variable type, an expected payload or return type, a `match` or `is` subject) and that type is an enum carrying the variant, it is the candidate. As a call argument, the context is the parameter type of each overload candidate in turn (§3.6): a candidate whose parameter cannot resolve the variant is simply not viable, so `::X` participates in overload selection like any other argument.
2. Otherwise every visible enum definition is searched for a variant of that name. Exactly one carrying it makes that enum the candidate; none or several is a compile-time error (the ambiguity is resolved by spelling the enum name).

```alloy
var maybe: Option<u32> = ::Some(7);   // context: the declared type
match (state) {
    ::Idle { }                        // context: the match subject
    ::Busy |load| { }
}
if (state is ::Busy) |load| { }       // context: the 'is' subject
```

Generic enums infer their type parameters through the same unification as named construction (§3.7).

### 3.3 Type Compatibility & Coercion Rules

1. **Identity** — identical types are always compatible.
2. **Untyped integer literal** — compatible with any numeric primitive type (not `bool`), or any named alias whose underlying chain reaches one. Regardless of radix format (`0x`, `0b`, `0o`, decimal), it resolves to `i32` when no contextual type is available. An array literal whose elements are untyped likewise adopts a contextual element type (`var a: [u8 : 3] = [1, 2, 3]`).
3. **Untyped float literal** — compatible with any float primitive (`f32`, `f64`). Resolves to `f32` when no contextual type is available.
4. **Named alias transparency** — a named type is compatible with anything its underlying type is compatible with.
5. **Numeric widening** — a numeric primitive is implicitly compatible with a wider primitive of the **same sign class**: unsigned→unsigned, signed→signed, float→float (`f32`→`f64` is implicit; the reverse requires `to`). Cross-class conversions require an explicit conversion cast (`x to T`, §3.5).
6. **Chained Nominal-Structural struct compatibility** — Two named struct types are fundamentally distinct (**nominal compatibility**). However, type-chaining logic allows implicit casting from more specific (narrower) to more general types layout-wise. A target expecting an anonymous layout (e.g., `struct { a: u8, b: f32 }`) will accept _any_ value—named or anonymous—whose internal shape structurally provides a matching set of required fields. Implicit casting up the chain to a more "general" layout is legal at any recursive depth level. Conversely, converting from a general, structurally loose layout down to a narrower/more specific named type requires an explicit reinterpretation cast (`x as NarrowType`, §3.5).
7. **Structural enum compatibility** — an inline `enum { ... }` type is compatible with any enum type — named or inline — whose **ordered variant list matches exactly**: same variant names in the same order, with identical payload types. Two distinct *named* enums remain nominally distinct even when their shapes match; the structural rule applies only when at least one side is an inline enum type.

Inline `struct { ... }` and `enum { ... }` types are permitted **wherever a type is expected**: parameter and return types, variable annotations, struct fields, enum payloads, generic arguments, and array elements. Values of an inline enum type are constructed with the implied-variant syntax (`::Variant`, §3.2), since the type has no name to qualify with.

### 3.4 Compile-Time Special Types (`#Type`)

`#Type` is a dedicated system representation primitive available **exclusively during compile-time evaluation**. A `#Type` value is a first-class, mutable description of a type — a struct or enum layout, a primitive, or an interface — that compile-time code may inspect and reconstruct.

- `#Type` maps directly to abstract structures, built-in primitives, or structural layouts.
- It exposes programmable compile-time methods enabling reflection and mutation (see below).
- Any attempt to retain or use `#Type` inside a standard runtime declaration or variable state triggers an immediate compile-time error.

#### Obtaining a `#Type`

- **`#T`** — prefixing a type name with the comptime token `#` yields the `#Type` reflecting `T` (e.g. `#u32`, `#Packet`). `#T.member_names()` reflects on `T` directly.
- **`#type_of(expr)`** — a built-in macro returning the `#Type` of the value `expr`'s type.
- **`#struct_type()` / `#enum_type()`** — built-in macros returning a fresh, empty struct / enum `#Type`, for synthesising a type from scratch.
- **`#implementers_of(I)`** — a built-in macro returning an array of the `#Type`s of every type in the merged unit implementing the interface `I`, whole-world across libraries (§6.4).
- **`#void`** — the `#Type` denoting the absence of a payload. Passed as the member type to `add_member` on an enum `#Type`, it marks a payload-less variant; reflected enums encode their payload-less variants the same way in `member_types()`. `void` is not a value type: `#void` in any runtime position is a compile-time error.

#### `#Type` methods

All are evaluated at compile time and called with dot syntax on a `#Type` value.

| Method                 | Signature                    | Semantics                                                                       |
| ---------------------- | ---------------------------- | ------------------------------------------------------------------------------- |
| `is_struct`            | `() -> bool`                 | True iff the type is a struct.                                                  |
| `is_enum`              | `() -> bool`                 | True iff the type is an enum.                                                   |
| `is_primitive`         | `() -> bool`                 | True iff the type is a built-in primitive.                                      |
| `is_interface`         | `() -> bool`                 | True iff the type is an interface.                                              |
| `implements_interface` | `(other: #Type) -> bool`     | True iff this type implements `other` (an interface), by the same conformance rule as §5.2 — declared markers resolved by definition identity (same-named interfaces from different libraries never confuse it) plus lang-item conformance (`Number`, `Iterable`). Synthesised `#Type`s implement nothing. |
| `name`                 | `() -> &[u8]`                | The type's declared name.                                                       |
| `equals`               | `(other: #Type) -> bool`     | True iff the two `#Type`s denote the same type.                                 |
| `add_member`           | `(name: &[u8], type: #Type)` | Appends a member (struct field or enum variant) of the given name and type.     |
| `remove_member`        | `(name: &[u8])`              | Removes the member with the given name.                                         |
| `member_names`         | `() -> &[&[u8]]`             | Member names, in declaration order.                                             |
| `member_types`         | `() -> &[#Type]`             | Member types, parallel to `member_names()`.                                     |

A `#Type` is a **value**: `add_member` / `remove_member` mutate the `#Type` value in hand, not the original type it was reflected from. The final `#Type` value, assigned through `type T = <#Type-valued comptime expression>`, becomes the synthesised type `T`. A synthesised struct behaves as a structural layout (§3.3 rule 6); a synthesised enum behaves as an inline enum type (§3.3 rule 7), and its variants are constructed, matched, and tested through the alias name (`T::Variant`) like any declared enum.

### 3.5 Casting

Two explicit cast operators cover every cast the coercion rules (§3.3) do not perform implicitly. Both are binary keyword operators (`cast_expr`, §2.1) whose right operand is a type.

| Operator | Name                   | Semantics                                                                                                  |
| -------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `x as T` | Reinterpretation cast  | Reinterprets the bytes of `x` as type `T` without changing them. No runtime cost.                          |
| `x to T` | Conversion cast        | Converts the value of `x` to type `T`, producing a new value (e.g. numeric conversion between sign classes or widths). |

- `as` requires the source and target layouts to have the same byte width, computed by the §3.9 layout rules for every type. Applied to a reference (`&S`), it yields a reference (`&T`) viewing the same memory without copying; the width requirement then applies to the pointee layouts `S` and `T`, not to the references themselves.
- `to` is defined for `Number` types (§5.2) and follows standard numeric conversion semantics (truncation, sign conversion, float/integer rounding).
- `is` (§3.2) belongs to the same grammar family but is a runtime type test on interface objects, yielding `bool`.

```alloy
var raw: u32 = 0x3F800000
var f = raw as f32          // same bits, viewed as f32 (1.0)
var n: i64 = -5
var u = n to u32            // numeric conversion, value-changing
```

### 3.6 Function Overloading

Functions (free functions and extension functions alike) may **overload**: any
number of functions may share a name as long as their parameter type lists
differ. Declaring two functions with the same name and an identical parameter
type list is a redeclaration error, as is reusing a function's name for any
non-function definition (§5.4). `extern` declarations do not overload — a C
symbol name is unique.

**Overload resolution.** At a call site the candidate set is every visible
function with the called name. A candidate is *viable* when its arity matches
and every argument is compatible (§3.3) with the corresponding parameter type.

- Exactly one viable candidate: it is called.
- Several viable candidates: if exactly one matches every argument type
  **identically** (no coercion), it wins; otherwise the call is **ambiguous**
  and a compile-time error.
- No viable candidate: a compile-time error listing the candidates.

### 3.7 Generic Type Parameter Inference

A generic function's type parameters are bound at each call site:

1. **Explicit arguments** (`make<u64>(7)`) bind type parameters left-to-right.
2. Remaining parameters are **inferred by unification**: each declared
   parameter type is matched structurally against the corresponding argument
   type (the `self` receiver participates like any argument), and every
   occurrence of an unbound type parameter binds to the type found in its
   position. `vector.push(x)` with `fn push<T>(self v: &var Vector<T>, item: T)`
   binds `T` from the receiver and checks `x` against it.
3. When a contextual expected type is available, the declared return type
   unifies against it **before** the arguments, binding type parameters the
   same way generic variant construction does
   (`var v: Vector<u8> = Vector::empty();`); arguments then unify against
   (and may coerce to) those bindings.
4. A type parameter still unbound after unification is a compile-time error;
   conflicting bindings (`T` unified with both `u32` and `f32`) are too.

A bound type parameter must satisfy its constraint (`<T: Number>`, §5.2).
During overload resolution (§3.6) a candidate whose unification fails, leaves
a parameter unbound, or binds a type violating its constraint is simply not
viable.

Constructing a variant of a generic enum follows the same rules: in
`Option::Some(x)` the type parameters bind by unifying the variant's payload
type against the argument and against the contextual expected type, exactly
like a function call. A parameter left unbound by both sources is a
compile-time error (`cannot infer type parameter 'T'`). A generic variant
construction passed as a call argument takes its context from each overload
candidate's parameter type (§3.6), the same way implied variants do (§3.2).

### 3.8 Mutability

Bindings are immutable by default and mutability is explicit:

- **`const`** declares an immutable binding; **`var`** declares a mutable one.
- **Parameters are immutable bindings.** A function mutates caller state only
  through `&var` / `*var` indirections passed to it.
- **Assignment** (including compound assignment) requires a **mutable target**:
  a `var` local, or a location reached through a `&var` / `*var` indirection.
- **`&var x`** (and `&var` / `*var` captures) require `x` to be mutable.
- Mutability pierces with pointee transparency (§4.2): a field accessed
  through a `&var T` value is mutable even when the reference binding itself
  is `const`; through `&T` it is immutable even on a `var` binding. For direct
  (non-indirected) access, fields and elements inherit the binding's
  mutability.

### 3.9 Data Layout

Struct layout is **C-compatible**: fields are laid out in declaration order, each aligned to its natural alignment, with padding inserted as in C; the struct's size is rounded up to a multiple of its largest field alignment. This makes `extern` FFI structs work without annotations and gives `as` reinterpretation (§3.5) well-defined widths. Enums are laid out as a tag (the smallest unsigned integer that fits the variant count) followed by the payload area sized and aligned for the largest payload, as C would lay out the corresponding tagged union.

---

## 4. Execution Semantics

### 4.1 Evaluation Order

**Eager (strict) evaluation.** All sub-expressions are fully evaluated before their result is used. Function arguments are evaluated **left-to-right** before the call.

### 4.2 Memory Model & Pointer Assignment Syntax

Alloy maps memory mechanics transparently using direct, predictable assignment rules:

#### Pointee Transparency

A pointer (`*T`, a managed heap-owning object) and a reference (`&T`, an unmanaged raw pointer in the C sense) are **always treated as their pointee** when used. There are no explicit dereference (`*ptr`) or member arrow (`->`) symbols: field access (`.field`), array indices (`[index]`), operators, function arguments, and plain reads all operate directly on the pointed-at value.

This includes ordinary assignment — reading a pointer or reference yields a **copy of the pointee**, never a copy of the address:

```alloy
var p: *T = new T {}
var x = p               // x is a deep copy of the value p points at, type T
var r: &T = &p          // r references p's pointee
var y = r               // y is likewise a deep copy of the pointee
```

**Deep copies and pointer uniqueness.** Every assignment copies its right-hand side **deeply**: if the copied value owns heap — directly or through members, elements, or payloads — each owned allocation is duplicated into a fresh allocation owned by the copy. As a consequence, **two pointers can never point at the same object**: a `*T` is always the unique pointer to (and owner of) its allocation. The only way to hand an existing allocation to another binding is `move`, which transfers the pointer and clears the source, preserving uniqueness. References carry no ownership and **may alias freely** — any number of `&T` values can point at the same object.

Three operators step outside pointee transparency:

- **`move`** is the **only** operator that treats its operand as an address rather than a value: `var q: *T = move p` transfers `p`'s pointer into `q` (and clears `p`, see below). It is valid for any pointer-typed operand — `*T`, `*var T`, and `*[T]` alike — and always yields the operand's own pointer type.
- **`new <expression>`** evaluates any expression and deep-copies the resulting value into a fresh heap allocation, yielding a pointer: `new 5`, `new T {}`, `new [0 : n]`, `new some_local`. It is the pointer-producing allocator.
- **Unary `&`** yields a reference to any value — a stack local, a struct field, an array element, or a heap value behind a pointer (`&p` references the pointee of `p`). Applied to a heap array, `&` yields a **slice** (`&[T]`) viewing every element in place — the only non-owning view of a `*[T]`'s contents (`arr[start..end]` subslicing, §2.1, views a range the same way).

#### Explicit Assignment Restrictions

- **Assigning to a Reference (`&Type` / `&var Type`)**: The unary address-of operator `&` is **strictly required** on the right-hand side of the assignment (e.g., `var r: &i32 = &stack_var`).
- **Assigning to a Heap Pointer (`*Type` / `*var Type`) or Dynamically Sized Array (`*[T]`)**: The assignment expression **strictly requires** either the `new` allocation operator or the `move` ownership transfer keyword (e.g., `var p: *i32 = new 5`, `var p2: *i32 = move p`). A bare pointer on the right-hand side would copy the pointee (pointee transparency), never alias the pointer.
- **Plain `=` rebinds, compound operators reach through:** plain assignment to a pointer- or reference-typed place targets the place itself — it rebinds the pointer or reference under the two restrictions above (dropping what an owning place held, see free-on-reassign below). Compound assignment (`+=`, `<<=`, ...) is an operator and follows pointee transparency: it reads and writes the pointed-at value.

#### Slices (`&[T]`) versus Dynamically Sized Heap Arrays (`*[T]`)

- **Slices (`&[T]`)**: Represent an unmanaged view into a sequence of elements whose bounds are unknown at compile time. Slices are structured internally as a runtime fat pointer pairing an address pointer with a explicit `u64` size boundary.
- **Dynamically Sized Heap Arrays (`*[T]`)**: Represent a completely managed heap instance block instantiated via a `new` allocation expression:

```alloy
var arr: *[u32] = new [0 : 120]     // 120 elements of u32, initialised to 0
var n: u64 = 120
var dyn: *var [u32] = new [0 : n]   // count may be a runtime expression
```

The fill count in `new [value : count]` may be a compile-time integer literal (which also permits a fixed stack array `[T : N]`) **or a runtime expression** — a runtime count always allocates a dynamically sized heap array (`*[T]`) of `count` elements, each initialised to `value`.

- **Memory Layout & C-FFI Compatibility:** To retain total binary drop-in compatibility with legacy C ecosystems, a pointer to an Alloy dynamically sized array points directly to the memory address of the first active data element (`element[0]`).
- **Length Metadata Tracking:** The allocation's length value (returned via `arr.length()`) is stored automatically by the runtime in a dedicated metadata prefix block located **immediately before the array data pointer** (i.e., at a negative memory offset from the user-facing pointer address).

#### Ownership, `move`, and Structural Reclaim

Ownership is **structural and automatic**. A value _owns heap_ if it is a `*T` / `*var T` / `*[T]` pointer, a closure (which owns its captured environment), or a struct / array / enum that transitively contains an owning member, element, or active-variant payload. Plain references (`&T`, `&var T`) and slices (`&[T]`) are non-owning views and never own heap.

- **Scope-end drop.** When an owning local goes out of scope (at every `return` path and at the implicit fall-through), the runtime _drops_ it: it recursively frees the heap it owns. Dropping a pointer frees its allocation (the `*[T]` form releases the malloc base at `user_ptr - 8`); dropping a `*[T]` first drops every one of its elements (all elements of an array are initialised, so each is reclaimable); dropping a struct/array/enum drops each owning field/element/active payload; dropping a closure frees its environment. Recursive owning types (e.g. a node holding `*Self`) terminate at the first null pointer.
- **`move` transfers ownership.** `move` is the only operator that reads its operand as an address rather than a value (§4.2 Pointee Transparency). `var q = move p` copies the pointer into `q` and clears (zeroes) the source binding, so the source is no longer an owner. After `move p`, `p` is a null pointer and its scope-end drop is a no-op. `move` always yields a `*T` value — whole-struct transfer is therefore expressed by moving a `*Struct` pointer (or by borrowing through `&var`), not by copying the struct.
- **Returning owned values is explicit — no implicit move.** `return move v` transfers the local's allocation to the caller: the source is cleared and its scope-end drop is a no-op. A bare `return v` returns **by value** — like any read, it yields a deep copy of what `v` holds, and the local's own heap is reclaimed by its normal scope-end drop. The same by-value rule applies to `break v` and `yield v`. A value built in the `return` expression itself (a constructor, `new`, …) is owned by the caller directly.
- **Pointer parameters take ownership.** A parameter of type `*T` / `*var T` / `*[T]` declares "I am taking ownership of this allocation". The caller **must** either `move` an existing pointer in (`take(move p)`) or allocate inline (`take(new T {})`); a `*T` value is **never** borrowed. The callee becomes the owner, and the parameter is dropped at the function's scope end like any owning local (unless moved on or returned). To lend a value without transferring it, pass a reference (`&T` / `&var T`) instead.
- **By-value parameters follow assignment semantics.** Passing a non-pointer value by value deep-copies it into the parameter, exactly like assignment (§4.2 Pointee Transparency); references borrow without copying.
- **Free-on-reassign.** Assigning to an owning binding (`buf = new […]`, `obj.field = move p`) first drops whatever that binding currently owns, then stores the new value — so the previous allocation is reclaimed rather than leaked.
- **Integer overflow:** arithmetic that exceeds its type's range is a **runtime fault in checked builds** (like the null checks below) and **wraps two's-complement in release builds**. Compile-time evaluation and the interpreter always fault. Division by zero is a fault in every build mode.
- **Debug builds** insert a null check on every dereference of a `*T` binding. A use-after-move accesses the null slot and traps (`@llvm.trap`). **Release builds** skip the null checks, so a use-after-move dereferences null and the OS faults. The null-store on `move` itself is kept in every build mode: it is the moved-from mark the drop machinery reads, so scope-end drops and free-on-reassign stay single-free after a transfer.
- **Definite use-after-move is a compile-time error.** The compiler tracks moves of bare local variables flow-sensitively: after `move x` (or an owning lambda capture `|*x|`), reading `x`, writing through `x.field`, moving it again, or capturing it is rejected at compile time — until a plain `=` rebinds it. The analysis merges branches conservatively: a move survives a merge only when every falling-through path performs it, so moves under a condition, inside one branch, inside a loop body, or of a field (`move x.inner`) remain checked **runtime** faults rather than compile errors.

**Growth is manual.** A `*[T]` array has a fixed length once allocated (its length lives in the `user_ptr - 8` prefix); there is no in-place resize or `realloc` primitive. A growable collection (`Vector`/`String`) is built by hand: allocate a larger `*var [T]` buffer with a runtime-sized `new [value : count]`, copy the elements across, and reassign the owning field — free-on-reassign reclaims the old buffer automatically. The standard library's generic `Vector<T>` (`std/vector.alloy`) and owning `String` (`std/string.alloy`) are written exactly this way; their mutating operations (`push`, `append`, …) take a `&var self` receiver and are invoked as methods (`vector.push(x)`).

> **Manual-safety caveats.** Alloy does not run a borrow checker. References are unchecked raw pointers: a `&T` can outlive the value it points at (a dropped local, a moved-from or reassigned owner) and dangle. Double-frees are ruled out by construction — deep copying plus pointer uniqueness (§4.2) means no two owners ever share an allocation — at the cost of implicit allocation when an owning value is copied; use `move` to transfer or `&`/`&var` to borrow where a copy is not intended.

---

### 4.3 Control Flow Semantics

**`return [value]`**
Immediately exits the enclosing function, yielding `value` as its result.

**`break [value]`**
Exits the **innermost enclosing loop** — a `for` or a `while` — passing transparently through any `if` or `match` in between, so a conditional loop exit is written naturally:

```alloy
for (items) |item| {
    if (item.done()) { break; } // exits the 'for'
}
```

When a value is provided, the loop evaluates directly to the value. A `break` outside a loop is a compile-time error.

**`yield value`**
Produces the value of the **innermost enclosing value-position `if` or `match`**, passing transparently through any loop in between (the loop is exited on the way out, its owned locals dropping normally). A `yield` with no enclosing value-position `if` or `match` is a compile-time error — a statement-position `if` or `match` has nothing to receive the value and is transparent to both `break` and `yield`.

#### `if` as a Value-Yielding Construct

An `if` used as a value yields via `yield value` in its branches; every branch must yield (or the construct is a compile-time error), and an `else` branch is required:

```alloy
var grade = if (score > 90) { yield "high"; } else { yield "low"; };
```

#### Path Termination

The compiler performs a conservative flow analysis over every function body (§ compile-time, no runtime cost). A statement **terminates** when control cannot fall out of it normally: `return`, `break`, and `yield` terminate; a block terminates when any of its statements does; an `if` with an `else` terminates when both branches do; a `match` terminates when every arm does; `while (true)` with no `break` reaching it never completes (divergence). Conditions are never assumed and ordinary loops always count as skippable.

- **Definite return:** a function or lambda with a non-void return type must terminate on every path — control falling off the end is a compile-time error.
- **Definite yield:** every branch of a value-position `if` must terminate; every arm of a value-position `match` must terminate unless an external `else` supplies the fall-through value, in which case the external `else` must terminate; the `else` of a value-yielding loop must terminate.

#### Loop Semantics (`for` and `while`)

- Loops are completely interface-driven. Any structural data collection or type implementing the `Iterable` interface (a lang item, §5.1a) — such as a fixed array, a slice `&[T]`, or a dynamically sized heap array `*[T]`, all of which implement it implicitly (§5.1) — can be utilized inside a `for` loop statement.
- **Custom iterables (cursor protocol).** A user type becomes iterable by providing two extension functions; the cursor is a separate value, not part of the container:

  ```alloy
  // The container yields a fresh cursor by value.
  fn iterator(self c: &Container) -> SomeIter { ... }
  // The cursor advances and reports the next element, or None when exhausted.
  fn next(self it: &var SomeIter) -> Option<T> { ... }
  ```

  `for (c) |x| { ... }` lowers to: `it = c.iterator()` then repeatedly `match (it.next()) { Some |x| <body> None { break the loop } }`. The loop variable `x` is bound to the `Option`'s payload type `T`. Built-in arrays/slices/`*[T]` use the faster index-based lowering instead.

- **Counting-loop lowering.** Whenever the subject's iteration count and element access are directly available — every built-in array form (`[T : N]`, `&[T]`, `*[T]`) and every range generator (`[start..end]`, §2.1) — the compiler **must** lower the `for` to a plain counting loop (a C-style `for` over an index), never the cursor protocol. A range generator used as a `for` subject (`for ([..n]) |i| { ... }`) materializes no array at all: it lowers to a counter running from `start` to `end`, which also makes runtime bounds valid in this position without `new`.

- **Multi-subject loops.** A `for` may take several comma-separated subjects, with one capture per subject in order: `for (a, b) |x, y| { ... }`. The subjects iterate in lockstep — each pass binds the next element of every subject — and every subject must itself be iterable. All subjects must produce the same number of elements; a length mismatch is a runtime fault in checked builds.

- **Expression-Only `else` Clause:** The trailing `else` block on a `for` or `while` loop is **only permitted when the entire loop construct is evaluated as an expression** (e.g., when assigning its value to a variable). When an `else` block is supplied, a value expression is explicitly required along all execution paths: the loop body **must** yield a value via an explicit `break value` statement, and the `else` block must evaluate to a value matching that same type. Using an `else` arm on a loop that is executed purely as a statement is a compile-time error.

#### Match Expressions

```alloy
var x = match (subject) {
    Pattern1 |payload_capture| { yield 10 }
    Pattern2 { yield 20 }
} else {
    // External else block
    yield 30 // Required expression fallback
}

```

- **Subject Versatility:** The subject of a `match` statement can evaluate to an enum variant, a numeric primitive, a character literal, or a string literal (treated natively as an array of integral numbers). Enum arm patterns name variants either fully (`State::Idle`) or in the implied form (`::Idle`, §3.2); a bare variant name is invalid.
- **Pattern Captures:** The pattern capture clause (`|capture|`) is **exclusively valid** when matching enum variants containing attached data payloads. Utilizing a pattern capture when matching numbers, characters, or strings results in a compile-time error. Captures follow the capture-typing rules (§2.1): deep copy by default, optionally annotated (`|a: &|` borrows the payload in place; an owning capture `|a: *|` takes a pointer payload out, leaving the match subject moved-from after the `match`).
- **Exhaustiveness:** Every `match` must cover all possible subject values, in statement and expression position alike. An enum subject is exhaustive when every variant appears as an arm pattern, or when an internal `else` arm is present. All other subjects — numbers, characters, strings, and interface objects — have open or unbounded domains and therefore always require an internal `else` arm. A non-exhaustive `match` is a compile-time error. (The external `else` block below is not a coverage fallback: it handles an arm completing without `yield`, not an unmatched subject.)
- **Match Evaluation & Value Yielding:** Distinct match arms yield an evaluated value from the outer `match` expression block by terminating via a `yield value` statement. A `break` inside a match arm targets the enclosing loop (§4.3 `break`), never the match.
- **Expression-Only External Match `else` Block:** A `match` structure supports an optional **external `else` block** positioned after its closing bracket. This block is **only permitted when the match is evaluated as an expression**. It executes if and only if the selected match arm completes its execution path normally **without producing a value via a `yield` statement**. Because it is constrained to expression contexts, the external `else` block must also provide a final value matching the expression's expected return type. Appending an external `else` block to a `match` construct used purely as a statement is a compile-time error.

---

### 4.4 Lambda / Closure Semantics

```alloy
|&var x, y| (param: T) -> R { body }

```

- The optional capture list (`|...|`) names variables from the enclosing scope. Each capture may carry a type modifier (`&`, `&var`, `*`, `*var`) that controls how the outer variable is accessed within the lambda.
- Capture lists are **value-only**. Type names — including the enclosing function's generic type parameters — remain visible inside the lambda's parameter types, return type, and body without being captured.
- A `*` / `*var` capture is an **owning capture**: it takes ownership of the captured variable, moving its pointer into the closure environment. The outer binding is invalid (moved-from) after the lambda expression. Valid only for pointer-typed variables (§2.1 Capture typing).
- The parameter list and optional return type follow the same syntax as a regular function.
- The type of a lambda expression is the corresponding function type `(T) -> R`.
- A lambda with no captures may omit the capture delimiters: `(param: T) { ... }`.
- A named, non-generic function used in value position becomes a function value of its signature's type. A generic function cannot become a function value (its type parameters are unbound), an overloaded name needs a unique function, and an extern function cannot be used as a function value.
- Function values have no defined identity or structural equality: comparing them with `==` / `!=` is a compile-time error.

### 4.5 Extension Functions

Any function whose first parameter is prefixed with `self` is an extension function:

```alloy
fn add(self v: &Vec3, other: &Vec3) -> Vec3 { ... }

```

- Called via dot notation: `v.add(other)`.
- The receiver is treated as an implicit first argument for the purpose of overload resolution.
- **Dot-call precedence:** extension resolution wins. When no extension (or interface function, §5.2) provides the name, a call `v.f(...)` where `f` is a function-typed field of `v` calls through the field's value (§4.4).
- `self` must appear only on the **first** parameter; any other position is an error.
- A **temporary receiver** (a call result, a literal construction) may invoke an extension whose `self` is an immutable reference (`&T`): the temporary materializes for the duration of the call. A `&var` receiver still requires a mutable place.
- A **pointer `self`** (`*T` / `*var T`) takes ownership of the receiver, exactly like a pointer parameter (§4.2): an owning place must transfer explicitly (`(move p).consume()`), and only a fresh value (`(new T {}).consume()`, a call result) passes bare.
- When the `self` receiver's type is an **interface** (e.g. `fn area(self s: &Shape) -> f32`), the extension is the interface function's **default implementation** — see §5.2.

---

## 5. Standard Library & Primitives

### 5.1 Arrays & Built-in Methods

All array forms — fixed arrays `[T : N]`, slices `&[T]`, and dynamically sized heap arrays `*[T]` — **implicitly implement the `Iterable` interface** (§5.1a, §5.2). This is the only implicit interface implementation in the language; every other type declares its interfaces explicitly (`type T : I = ...`, §5.2).

The compiler provides the arrays' `.length() -> u64` implementation directly, because every array form already carries its size — each in a different place:

| Array form        | Where the length lives                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------- |
| `[T : N]` fixed   | Known statically at compile time; `.length()` folds to the constant `N`.                    |
| `&[T]` slice      | The `u64` length half of the slice's fat pointer.                                           |
| `*[T]` heap array | The metadata prefix stored immediately before the pointee (`user_ptr - 8`, §4.2).           |

Custom types implement `Iterable` like any other interface and supply `.length()` themselves.

Reinterpretation and conversion are performed by the `as` / `to` cast operators (§3.5), not by built-in methods.

### 5.1a Compiler-Recognized Declarations (Lang Items)

A small set of standard library declarations is **recognized by the compiler by canonical path** to power syntactic sugar. There is **no prelude**: lang items are ordinary Alloy source shipped with the standard library (§5.4), and nothing is in scope until imported. The compiler's own lowering references the canonical declaration directly — an import is only required to *name* the item in source code.

| Lang item   | Canonical path          | Declaration                                          | Compiler hook                                                                                                    |
| ----------- | ----------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `Option<T>` | `std::option::Option`   | `enum { Some: T, None }`                              | Cursor protocol: `for` over a custom iterable lowers to repeated `match` on `next()`'s `Option<T>` result (§4.3). |
| `Iterable`  | `std::iterable::Iterable` | Interface: `.length() -> u64` plus iteration support | Drives `for` loops; arrays implement it implicitly (§5.1).                                                        |
| `Number`    | `std::number::Number`   | Interface                                             | Satisfied by the primitive numeric types (§3.1); bounds generic constraints and the `to` conversion cast (§3.5).  |
| `arguments` | `std::process::arguments` | `fn arguments() -> &[&[u8]]`                        | The compiler supplies the command line (first element: the program's own path). Natively the entry wrapper captures argc/argv at startup; `alloyc run` serves the arguments after the program path. Unavailable at compile time (§6.2). |

```alloy
import std::option

var maybe: Option<u32> = Option::Some(42)
match (maybe) {
    Option::Some |v| { /* v : u32 */ }
    Option::None    { /* empty */ }
}
```

A user definition colliding with an imported lang-item name follows the normal redeclaration rules (§5.4); lang items carry no special naming privileges.

### 5.2 Standard & User-Defined Interfaces

The two standard interfaces are lang items (§5.1a) defined in ordinary standard library source:

| Name       | Satisfied by                                                                                                                     |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `Number`   | `u8` `u16` `u32` `u64` `i8` `i16` `i32` `i64` `f32` `f64`                                                                        |
| `Iterable` | Arrays, slices `&[T]`, and dynamic heap arrays `*[T]` (implicitly, §5.1); custom types providing `.length()` and iteration support. |

Used as a type-parameter constraint: `fn foo<T: Number>(...)`.

#### User-Defined Interfaces

Interfaces define traits or constraints as named contract blocks of function signatures:

```alloy
interface Serializable {
    fn serialize(format: u32) -> bool
}

```

A nominal type alias links itself explicitly to one or more user-defined interfaces using a C++-inspired mapping annotation during its declaration syntax:

```alloy
type Packet : Serializable, Iterable = struct {
    id: u32,
    payload: *[u8],
}

```

#### The Two Roles of an Interface

An interface may be used in **two distinct ways**:

1. **Dynamic dispatch.** An interface used as a type — always behind an indirection (`&I`, `&var I`, `*I`, `*var I`) — produces an _interface object_ (§3.2). A value of any concrete type that implements `I` is implicitly convertible to such an interface object. Calling an interface function through an interface object (`handle.do_something()` where `handle: &Shape`) is resolved at **runtime through the vtable** to the concrete type's implementation.

2. **Generic constraint.** An interface used as a type-parameter bound (`fn do<T: I>(...)`) restricts the generic to types that implement `I`. The call is resolved **statically** at each instantiation; no vtable is involved. Inside the generic body, a value of type `T` exposes the constraint's interface functions via dot notation; a `T: Number` value additionally supports the arithmetic and comparison operators (§3.1).

#### Default Implementations

An **extension function whose `self` receiver is an interface** is the **default implementation** of that interface function:

```alloy
interface Shape {
    fn area() -> f32
    fn name() -> &[u8]
}

// default implementation of Shape::name, shared by every implementer
fn name(self s: &Shape) -> &[u8] { return "shape" }
```

- A default implementation makes the corresponding interface function **optional** for implementing types.
- An extension function written for a **concrete type** _overrides_ the default for that type. When resolving a call on a concrete value, a type-specific extension is always preferred over an interface default.

#### Compilation & Verification Mechanics

When a type `T` is flagged with interface markers (`type T : I1, I2 = ...`), the compiler performs a static verification pass over the module scope. For every function declared inside each interface (`I1`, `I2`):

1. A satisfying **extension function** (§4.5) must be visible in the module — either an extension belonging to `T` itself, **or** a default implementation (an extension whose `self` receiver is the interface).
2. The satisfying extension must precisely match the method name, the parameter sequence, and the return type specified by the interface declaration. The parameters following `self` correspond positionally to the interface function's parameter list.
3. The receiver indirection of the satisfying extension's first parameter (`self: &T`, `self: *var T`, etc.) dictates what memory state or qualifier context is permitted when invoking that interface function polymorphically.
4. If neither a type-specific extension nor a default implementation is visible, verification fails with a compile-time error.

---

### 5.3 Extern FFI

External C functions must be explicitly described with fixed signatures that mandate concrete arrow return types:

```alloy
extern functionName(param: Type) -> ReturnType
extern variadicFunc(...) -> *var u8

```

- **Architecture Strategy:** The FFI layer is intentionally isolated. Raw `extern` declarations are designed strictly for low-level systems developers contextually porting legacy C libraries to Alloy ecosystems. Standard software applications are expected to consume safe, native Alloy modules that seamlessly wrap and encapsulate these unsafe FFI barriers.
- **Slice decay:** a slice (`&[T]`) crossing the extern boundary — as a parameter or a variadic argument — passes only its **data pointer**, matching the C convention; the length stays behind. String literal bytes are NUL-terminated in static memory, so a literal passed to C is a valid C string.
- **Variadic promotions:** arguments in a variadic tail follow the C default promotions (integers narrower than 32 bits widen to `i32` by their own signedness, `f32` widens to `f64`).

### 5.4 Module System

- **Strict File System Mirroring:** Qualified module pathways match physical disk positioning precisely. An instruction like `import a::b::c` commands the compiler to look explicitly for a source file located at `a/b/c.alloy`.
- **Standard library:** The standard library ships as ordinary Alloy source files alongside the compiler. It is **not** a prelude — nothing is in scope until imported. The modules: `std::option` (`Option<T>`, §5.1a), `std::number` and `std::iterable` (the constraint interfaces, §5.2), `std::vector` (the growable `Vector<T>`), `std::string` (the owned `String` builder), `std::io` (console and file input/output — the library's FFI barrier, §5.3: programs call these wrappers and never touch the C functions underneath), and `std::process` (`arguments()`, §5.1a).
- **`alloyc build` import resolution:** When compiling a single file to a native executable (`alloyc build file.alloy [-o out] [--release] [--emit-llvm]`), each `import a::b::c` is resolved to `a/b/c.alloy` searched under, in order: the current directory, the compiler-executable's directory, and `$ALLOY_STDLIB`. Every reachable module is **merged into one compilation unit**. The build is **checked** by default and `--release` selects the release semantics of §4.2; the backend emits LLVM IR and an external clang (located via `$ALLOY_CLANG`, then `PATH`) produces the executable.
- **Debug info:** checked builds embed DWARF debug metadata — file, function, and statement-level line/column locations — so native debuggers (LLDB, GDB) set source breakpoints, step by statement, and show Alloy names in call stacks. Release builds carry none. On Windows the debug link goes through lld (`/debug:dwarf`); if lld is unavailable the build falls back to linking without debug info and says so.
- **Program entry:** execution starts at a zero-parameter function named `main` in the entry module. An integer result becomes the process exit code (truncated to the platform's width); any other result type, or none, exits with 0.
- **Qualified vs. unqualified access:** an imported name may be written either unqualified (`Vector`) or module-qualified (`std::vector::Vector`). Every import also introduces an alias for qualified use — the explicit `as` name or the import path's last segment (`import pkg::mathx` allows `mathx::twice(...)`). Qualified access goes through the cross-module visibility check, so only `pub`/`exp` definitions are reachable that way. Unqualified access sees the requester's **own library** in full (the executable's own modules and `std::` count as one library), plus the `exp` definitions of each library the module imported **without an explicit `as`** — an unaliased library import *injects* its exports into that module's unqualified namespace, while an aliased import (`import pkg::liba as la`) is reachable through the alias only (`la::Pair`, `la::Pair { ... }`).
- **Qualified functions (constructors):** `fn Vector::empty<T>() -> Vector<T> { ... }` defines a plain free function living in the **type's namespace** instead of the module's flat namespace, called as `Vector::empty()` (or `alias::Vector::empty()` across an aliased import). The qualifier must name a type visible to the defining module - like extension functions, any accessible type qualifies, not only locally declared ones. Qualified functions of one type overload among themselves; the same name may freely exist as a free function or under other types. On an enum, a qualified name colliding with a variant is a compile-time error, so `Type::Name(...)` stays unambiguous and variant construction is unchanged. A qualified function must not declare a `self` receiver - it is a plain free function, not an extension (a `self` parameter is a compile-time error); no dot-call, no dispatch - the association is purely a namespace.
- **Name collisions:** within one library, a name colliding with an existing definition is a redeclaration error, except that functions overload (§3.6). Different libraries may reuse names internally. A name visible unqualified in one module from **two different libraries** (own declaration vs. an injected export, or two injected exports) is a **compile-time error at the import**, resolved by aliasing an import to take its exports out of the unqualified namespace. Nothing is resolved implicitly — no shadowing, no cross-library overload merging. Two imports whose aliases collide (implicit or explicit) are likewise an error.
- **Merge-then-codegen:** every reachable module — including every library module — always merges into ONE compilation unit before type checking and code generation. The whole-program stages depend on it (closed-world interface dispatch, monomorphization, §3.9 layouts); only the per-module front-end stages (tokenize, parse) run in parallel. Libraries therefore recompile with each consuming program; the `.alloylib` payload exists to make that cheap, not to skip it.

#### Import Namespaces

- `import a::b` — a **relative** import: the file `a/b.alloy` next to the importing code. Inside a library, relative imports resolve within that library's own namespace.
- `import std::x` — the **standard library**, shipped as Alloy source alongside the compiler.
- `import pkg::name[::module]` — a **package**: the compiler looks for `pkg/name.alloylib` in the project first, then (future) the trusted registry. `pkg::name` imports the package's entry module; `pkg::name::module` one of its members.

#### Libraries (`.alloylib`)

- `alloyc lib entry.alloy [-o name.alloylib]` fully checks the unit standalone (all §3/§4 rules, flow and move analysis included) and packs the entry module plus every module of its own into a container. `std::` and `pkg::` dependencies are not packed — they stay imports the consuming program resolves, so a library's package dependencies load transitively.
- The container embeds the **complete source** — the registry mandates open source, and the embedded source is authoritative: precompiled cache sections (future) are stamped with the producing compiler's version and silently ignored on mismatch, falling back to compiling the embedded source. A library therefore never breaks across compiler releases.
- **Export boundary:** `exp` marks a definition as exported from a library. Within one compilation unit `exp` behaves exactly like `pub`; across a library boundary, only `exp` definitions are visible to consumers — qualified (`mathx::twice(...)`) or unqualified via an unaliased import per the injection rules above — while `pub` covers a library's internal cross-module structure without leaking it.
- **Interface satisfaction stays closed-world:** whether a type satisfies an interface (§5.2) considers every extension in the merged unit, even library-internal ones — vtables span the whole program. Visibility governs who may *name* an extension in a direct call, not whether it backs dynamic dispatch.
- **Comptime re-runs per program** (§6): library comptime and macros are re-evaluated in the consuming program's merged unit, so compile-time reflection can see the final closed world (every interface implementer, every type), not just the library's own.

---

## 6. Compile-Time Evaluation & Macros

### 6.1 The Comptime Modifier (`#`)

Any **value-yielding expression** prefixed by the token `#` — an `#if`, `#while`, `#match`, an arbitrary function call (`#compute(x)`), an identifier, or a parenthesised expression — is intercepted by the compiler and executed at compile time via an internal interpreter. A `#`-marked construct must produce a value; a bare statement block (`{ … }`) is not a value and cannot be marked.

#### Value-Substitution Model

Comptime expressions operate on a pure value-substitution model. Once a compile-time expression completes execution, its entire syntax node tree is stripped from the final runtime code layout and replaced with its final calculated literal value, struct initialization block, or nominal type signature.

A value-yielding construct yields its value via `yield` (§4.3), so an `#if` selecting between two values is written:

```alloy
const a = #if (cond) yield 50 else yield 100

```

#### Implicit Comptime Inheritance

When a `#` modifier marks an outer expression, all nested child expressions, loop structures, and evaluation flows contained within that outer node scope implicitly inherit compile-time evaluation.

#### Visible Names

A comptime expression may reference literals, any function in the program (including ones defined later in the file), and enclosing `const` locals whose initializers are themselves compile-time evaluable (transitively: a `const` built from literals, other compile-time-known constants, and function calls over them). Runtime state — `var` bindings, function parameters, and `const` locals whose initializers depend on runtime state — is invisible to compile-time evaluation; referencing it is a compile-time error.

A fault during compile-time evaluation (integer overflow, division by zero, an exhausted evaluation budget) is a compile-time error.

### 6.2 The Compile-Time Pointer Barrier & Sandboxing

To eliminate cross-compilation target safety errors and prevent memory leakage from host architectures into generated binaries, compile-time blocks are bound to strict computational constraints:

- **The Pointer Barrier:** A compile-time evaluation block cannot yield an unmanaged reference (`&T`), a managed pointer (`*T`), or a closure (which owns captured references) that escapes into a runtime variable. Any data crossing the boundary from compile-time execution to a runtime variable state must be handled strictly as values. Breaking this constraint triggers a compile-time error. Slices (`&[T]`) and heap arrays (`*[T]`) are the exception: a slice or heap-array result is **materialized** — deep-copied into static program data — so a comptime call yielding a string works.
- **Strict Sandboxing Boundaries:** The compile-time interpreter is strictly restricted to the project's physical root directory workspace (mirroring physical filesystem rules §5.4).
- **Foreign Function Isolation:** Comptime blocks are strictly prohibited from invoking low-level `extern` C functions (§5.3). Compile-time evaluation can only run safe user-defined Alloy code or built-in system macros.

### 6.3 Macros

Macros are specialized compile-time functions used for code reflection, source file introspection, and automated type generation.

```alloy
macro readTypeFromJson(path: &[u8]) {
    // Introspection and type mutation using compile-time features
}

```

- **Signature and Inferred Types:** Macros are defined using the `macro` keyword. While macro input parameters are strictly typed, **macro return types are completely inferred by the compiler** based on the generated AST layout or the underlying type node replacement it yields.
- **Invocation Syntax:** To explicitly distinguish macros from standard functions, all macro calls must be preceded by the `#` character token.
- **Declaration Order:** Because a macro's result type is inferred from the value it produces, the compiler evaluates a macro the moment it checks the invocation site. Every definition a macro's body touches (functions, types, other macros) must therefore appear **earlier in program order** than the invocation. (`#` expressions calling only regular functions are exempt: their signatures are known statically, so evaluation waits until checking completes and forward references work.)
- **Value Position:** A macro invoked in value position (`const x = #m(1)`) takes the type of the value its body produced. Legal results are plain values: primitives, bools, strings and slices, fixed arrays, and named struct or enum values. A `#Type` result or a pointer in value position is a compile-time error (§3.4, §6.2).
- **Macro Bodies:** A macro body is not statically type-checked; it executes in the compile-time interpreter, where calls resolve by name and arity. Faults inside a macro body are compile-time errors at the invocation site.
- **Type Synthesis Examples:**

```alloy
type T = #readTypeFromJson("types/T.json")
type P = #if (DEVELOPMENT) yield struct { id: u32 } else yield #readTypeFromJson("types/P.json")

```

### 6.4 Built-in Macros

The compiler provides a small set of built-in macros. Like all macros they are invoked with a leading `#`.

| Macro             | Signature               | Semantics                                                |
| ----------------- | ----------------------- | -------------------------------------------------------- |
| `type_of`         | `(value) -> #Type`      | The `#Type` of the argument expression's type (§3.4).    |
| `struct_type`     | `() -> #Type`           | A fresh, empty struct `#Type`, for synthesising a type.  |
| `enum_type`       | `() -> #Type`           | A fresh, empty enum `#Type`.                             |
| `implementers_of` | `(interface) -> [#Type]` | Every type in the merged unit implementing the interface. |

A type may also be reflected directly by prefixing its name with `#` (`#u32`, `#Packet`) — see §3.4.

**`implementers_of` is whole-world** (§5.4): because every module — including every library module — merges before checking, the returned array covers the entire program regardless of declaration order or library visibility: a library-internal type implementing an exported interface is included. Elements arrive in module order, then declaration order, so results are deterministic. Generic types are excluded (they reflect only as instances, §3.4). The array is a compile-time value: consume it inside the `#` expression (`#(implementers_of(Shape).length())`, `#(implementers_of(Shape)[0].name())`) — a `#Type` itself cannot be retained into runtime (§6.2). Types synthesised by comptime (`type T : I = #...`) are enumerated like any other declaration, but their member descriptions are only complete once their own `#` initialiser has evaluated (declaration order, §6.3).

---

_End of specification._
