"""
parser.py — Recursive Descent Parser for Small-C
==================================================
Consumes a flat list of tokens and builds an Abstract Syntax Tree.
Each grammar rule is implemented as one method. Operator precedence
is enforced by the call chain depth — deeper methods bind tighter.

Pipeline position: Token List → [Parser] → AST → Semantic Analyzer
"""

from __future__ import annotations
from typing import List, Optional, Any

from errors import ParseError
from lexer import Token
from lexer import (
    T_INT, T_CHAR, T_VOID, T_IF, T_ELSE, T_WHILE, T_FOR, T_RETURN, T_BREAK,
    T_INTEGER_LITERAL, T_CHAR_LITERAL, T_STRING_LITERAL, T_IDENTIFIER,
    T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
    T_EQ, T_NEQ, T_LT, T_GT, T_LEQ, T_GEQ,
    T_ASSIGN, T_AND, T_OR, T_NOT, T_INC, T_DEC,
    T_LPAREN, T_RPAREN, T_LBRACE, T_RBRACE, T_LBRACKET, T_RBRACKET,
    T_SEMICOLON, T_COMMA, T_EOF,
)
from ast_nodes import (
    Program, FunctionDecl, ParamDecl, Block,
    VarDecl, ArrayDecl,
    IfStmt, WhileStmt, ForStmt, ReturnStmt, BreakStmt, ExprStmt,
    AssignExpr, BinaryExpr, UnaryExpr, PostfixExpr,
    CallExpr, ArrayAccessExpr, IdentifierExpr,
    IntLiteral, CharLiteral, StringLiteral,
)


# Set of token types that represent a type specifier
TYPE_TOKENS = {T_INT, T_CHAR, T_VOID}


class Parser:
    """Recursive descent parser for the Small-C grammar.

    Usage:
        parser = Parser(token_list, "factorial.c")
        ast = parser.parse()
    """

    def __init__(self, tokens: List[Token], filename: str):
        self.tokens = tokens
        self.pos = 0
        self.filename = filename

    # ── helper methods ────────────────────────────────────────────

    def peek(self) -> Token:
        """Return the current token without consuming it."""
        return self.tokens[self.pos]

    def peek_type(self) -> str:
        """Return the type string of the current token."""
        return self.tokens[self.pos].type

    def consume(self) -> Token:
        """Return the current token and advance the position."""
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, token_type: str) -> Token:
        """Consume the current token if its type matches, else raise ParseError."""
        tok = self.peek()
        if tok.type != token_type:
            raise ParseError(
                f"expected {token_type}, found {tok.type} ('{tok.value}')",
                self.filename, tok.line
            )
        return self.consume()

    def match(self, *types: str) -> Optional[Token]:
        """If the current token matches any of the given types, consume and return it.
        Otherwise return None without advancing."""
        if self.peek_type() in types:
            return self.consume()
        return None

    def _error(self, msg: str) -> None:
        """Raise a ParseError at the current token's line."""
        raise ParseError(msg, self.filename, self.peek().line)

    # ── top-level ─────────────────────────────────────────────────

    def parse(self) -> Program:
        """Public entry point — parses the entire source into a Program AST."""
        return self.parse_program()

    def parse_program(self) -> Program:
        """program → function_decl* EOF"""
        functions: List[FunctionDecl] = []
        while self.peek_type() != T_EOF:
            functions.append(self.parse_function_decl())
        return Program(functions=functions)

    def parse_function_decl(self) -> FunctionDecl:
        """function_decl → type IDENTIFIER '(' params ')' block"""
        type_tok = self.peek()
        if type_tok.type not in TYPE_TOKENS:
            self._error(f"expected type specifier (int/char/void), found '{type_tok.value}'")
        return_type = self.consume().value
        line = type_tok.line

        name_tok = self.expect(T_IDENTIFIER)
        name = name_tok.value

        self.expect(T_LPAREN)
        params = self.parse_params()
        self.expect(T_RPAREN)

        body = self.parse_block()

        return FunctionDecl(
            name=name, return_type=return_type,
            params=params, body=body, line=line
        )

    def parse_params(self) -> List[ParamDecl]:
        """params → ε | type IDENTIFIER (',' type IDENTIFIER)*"""
        params: List[ParamDecl] = []

        # Empty parameter list
        if self.peek_type() == T_RPAREN:
            return params

        # First parameter
        params.append(self._parse_one_param())

        # Remaining parameters
        while self.match(T_COMMA):
            params.append(self._parse_one_param())

        return params

    def _parse_one_param(self) -> ParamDecl:
        """Parse a single 'type IDENTIFIER' parameter declaration."""
        type_tok = self.peek()
        if type_tok.type not in (T_INT, T_CHAR):
            self._error(f"expected parameter type (int/char), found '{type_tok.value}'")
        param_type = self.consume().value
        name_tok = self.expect(T_IDENTIFIER)
        return ParamDecl(name=name_tok.value, type=param_type, line=type_tok.line)

    # ── blocks and statements ─────────────────────────────────────

    def parse_block(self) -> Block:
        """block → '{' statement* '}'"""
        lbrace = self.expect(T_LBRACE)
        stmts: List[Any] = []
        while self.peek_type() != T_RBRACE:
            stmts.append(self.parse_statement())
        self.expect(T_RBRACE)
        return Block(statements=stmts, line=lbrace.line)

    def parse_statement(self) -> Any:
        """Dispatch to the correct statement parser based on the current token."""
        tt = self.peek_type()

        if tt in (T_INT, T_CHAR):
            return self.parse_var_decl()
        if tt == T_IF:
            return self.parse_if_stmt()
        if tt == T_WHILE:
            return self.parse_while_stmt()
        if tt == T_FOR:
            return self.parse_for_stmt()
        if tt == T_RETURN:
            return self.parse_return_stmt()
        if tt == T_BREAK:
            return self.parse_break_stmt()

        return self.parse_expr_stmt()

    def parse_var_decl(self) -> Any:
        """var_decl → type IDENTIFIER ';'
                    | type IDENTIFIER '=' expr ';'
                    | type IDENTIFIER '[' INTEGER_LITERAL ']' ';'
        """
        type_tok = self.consume()  # INT or CHAR
        var_type = type_tok.value
        line = type_tok.line

        name_tok = self.expect(T_IDENTIFIER)
        name = name_tok.value

        # Array declaration: type name[size];
        if self.match(T_LBRACKET):
            size_tok = self.expect(T_INTEGER_LITERAL)
            size = int(size_tok.value)
            self.expect(T_RBRACKET)
            self.expect(T_SEMICOLON)
            return ArrayDecl(name=name, type=var_type, size=size, line=line)

        # Variable with initialiser: type name = expr;
        if self.match(T_ASSIGN):
            init_expr = self.parse_expr()
            self.expect(T_SEMICOLON)
            return VarDecl(name=name, type=var_type, init_expr=init_expr, line=line)

        # Plain declaration: type name;
        self.expect(T_SEMICOLON)
        return VarDecl(name=name, type=var_type, init_expr=None, line=line)

    def parse_if_stmt(self) -> IfStmt:
        """if_stmt → 'if' '(' expr ')' block
                     ('else' 'if' '(' expr ')' block)*
                     ('else' block)?
        """
        if_tok = self.expect(T_IF)
        line = if_tok.line

        self.expect(T_LPAREN)
        condition = self.parse_expr()
        self.expect(T_RPAREN)
        then_block = self.parse_block()

        elif_clauses = []
        else_block = None

        # Process else / else-if chain
        while self.match(T_ELSE):
            if self.peek_type() == T_IF:
                # else if
                self.consume()  # consume IF
                self.expect(T_LPAREN)
                elif_cond = self.parse_expr()
                self.expect(T_RPAREN)
                elif_block = self.parse_block()
                elif_clauses.append((elif_cond, elif_block))
            else:
                # plain else — must be the last branch
                else_block = self.parse_block()
                break

        return IfStmt(
            condition=condition, then_block=then_block,
            elif_clauses=elif_clauses, else_block=else_block,
            line=line
        )

    def parse_while_stmt(self) -> WhileStmt:
        """while_stmt → 'while' '(' expr ')' block"""
        tok = self.expect(T_WHILE)
        self.expect(T_LPAREN)
        condition = self.parse_expr()
        self.expect(T_RPAREN)
        body = self.parse_block()
        return WhileStmt(condition=condition, body=body, line=tok.line)

    def parse_for_stmt(self) -> ForStmt:
        """for_stmt → 'for' '(' init ';'? condition ';' update ')' block

        The init clause can be either a var_decl (which already eats its ';')
        or an expression statement (needs explicit ';' consumption).
        """
        tok = self.expect(T_FOR)
        self.expect(T_LPAREN)

        # --- init clause ---
        if self.peek_type() in (T_INT, T_CHAR):
            # var_decl already consumes the trailing semicolon
            init = self.parse_var_decl()
        elif self.peek_type() == T_SEMICOLON:
            # Empty init
            self.consume()
            init = None
        else:
            init = self.parse_expr()
            self.expect(T_SEMICOLON)

        # --- condition ---
        if self.peek_type() == T_SEMICOLON:
            condition = None
        else:
            condition = self.parse_expr()
        self.expect(T_SEMICOLON)

        # --- update ---
        if self.peek_type() == T_RPAREN:
            update = None
        else:
            update = self.parse_expr()
        self.expect(T_RPAREN)

        body = self.parse_block()

        return ForStmt(init=init, condition=condition, update=update,
                       body=body, line=tok.line)

    def parse_return_stmt(self) -> ReturnStmt:
        """return_stmt → 'return' ';' | 'return' expr ';'"""
        tok = self.expect(T_RETURN)
        if self.peek_type() == T_SEMICOLON:
            self.consume()
            return ReturnStmt(expr=None, line=tok.line)
        expr = self.parse_expr()
        self.expect(T_SEMICOLON)
        return ReturnStmt(expr=expr, line=tok.line)

    def parse_break_stmt(self) -> BreakStmt:
        """break_stmt → 'break' ';'"""
        tok = self.expect(T_BREAK)
        self.expect(T_SEMICOLON)
        return BreakStmt(line=tok.line)

    def parse_expr_stmt(self) -> ExprStmt:
        """expr_stmt → expr ';'"""
        expr = self.parse_expr()
        self.expect(T_SEMICOLON)
        return ExprStmt(expr=expr, line=expr.line)

    # ── expressions (ordered from lowest to highest precedence) ───

    def parse_expr(self) -> Any:
        """expr → assignment"""
        return self.parse_assignment()

    def parse_assignment(self) -> Any:
        """assignment → (IDENTIFIER | array_access) '=' assignment
                      | logical_or

        We parse the left side as a full logical_or expression, then
        check if it's a valid assignment target followed by '='.
        This avoids ambiguity without backtracking.
        """
        # Save position so we can try assignment first
        left = self.parse_logical_or()

        if self.peek_type() == T_ASSIGN:
            self.consume()  # eat '='
            # Validate the left-hand side is assignable
            if isinstance(left, (IdentifierExpr, ArrayAccessExpr)):
                value = self.parse_assignment()  # right-associative
                return AssignExpr(target=left, value=value, line=left.line)
            else:
                raise ParseError(
                    "invalid assignment target — left side must be a variable or array element",
                    self.filename, left.line
                )

        return left

    def parse_logical_or(self) -> Any:
        """logical_or → logical_and ('||' logical_and)*"""
        left = self.parse_logical_and()
        while self.peek_type() == T_OR:
            op_tok = self.consume()
            right = self.parse_logical_and()
            left = BinaryExpr(op="||", left=left, right=right, line=op_tok.line)
        return left

    def parse_logical_and(self) -> Any:
        """logical_and → equality ('&&' equality)*"""
        left = self.parse_equality()
        while self.peek_type() == T_AND:
            op_tok = self.consume()
            right = self.parse_equality()
            left = BinaryExpr(op="&&", left=left, right=right, line=op_tok.line)
        return left

    def parse_equality(self) -> Any:
        """equality → comparison (('==' | '!=') comparison)*"""
        left = self.parse_comparison()
        while self.peek_type() in (T_EQ, T_NEQ):
            op_tok = self.consume()
            right = self.parse_comparison()
            left = BinaryExpr(op=op_tok.value, left=left, right=right, line=op_tok.line)
        return left

    def parse_comparison(self) -> Any:
        """comparison → addition (('<' | '>' | '<=' | '>=') addition)*"""
        left = self.parse_addition()
        while self.peek_type() in (T_LT, T_GT, T_LEQ, T_GEQ):
            op_tok = self.consume()
            right = self.parse_addition()
            left = BinaryExpr(op=op_tok.value, left=left, right=right, line=op_tok.line)
        return left

    def parse_addition(self) -> Any:
        """addition → multiplication (('+' | '-') multiplication)*"""
        left = self.parse_multiplication()
        while self.peek_type() in (T_PLUS, T_MINUS):
            op_tok = self.consume()
            right = self.parse_multiplication()
            left = BinaryExpr(op=op_tok.value, left=left, right=right, line=op_tok.line)
        return left

    def parse_multiplication(self) -> Any:
        """multiplication → unary (('*' | '/' | '%') unary)*"""
        left = self.parse_unary()
        while self.peek_type() in (T_STAR, T_SLASH, T_PERCENT):
            op_tok = self.consume()
            right = self.parse_unary()
            left = BinaryExpr(op=op_tok.value, left=left, right=right, line=op_tok.line)
        return left

    def parse_unary(self) -> Any:
        """unary → '!' unary
                 | '-' unary
                 | '++' IDENTIFIER
                 | '--' IDENTIFIER
                 | postfix
        """
        # Logical NOT
        if self.peek_type() == T_NOT:
            op_tok = self.consume()
            operand = self.parse_unary()
            return UnaryExpr(op="!", operand=operand, line=op_tok.line)

        # Unary minus
        if self.peek_type() == T_MINUS:
            op_tok = self.consume()
            operand = self.parse_unary()
            return UnaryExpr(op="-", operand=operand, line=op_tok.line)

        # Prefix increment
        if self.peek_type() == T_INC:
            op_tok = self.consume()
            name_tok = self.expect(T_IDENTIFIER)
            ident = IdentifierExpr(name=name_tok.value, line=name_tok.line)
            return UnaryExpr(op="++", operand=ident, line=op_tok.line)

        # Prefix decrement
        if self.peek_type() == T_DEC:
            op_tok = self.consume()
            name_tok = self.expect(T_IDENTIFIER)
            ident = IdentifierExpr(name=name_tok.value, line=name_tok.line)
            return UnaryExpr(op="--", operand=ident, line=op_tok.line)

        return self.parse_postfix()

    def parse_postfix(self) -> Any:
        """postfix → primary
                   | primary '++'
                   | primary '--'
                   | IDENTIFIER '(' args ')'
                   | IDENTIFIER '[' expr ']'

        We parse primary first, then check for postfix operators.
        Function calls and array accesses start with IDENTIFIER, which
        primary() will return as IdentifierExpr — we then promote it.
        """
        node = self.parse_primary()

        # Postfix increment / decrement
        if self.peek_type() == T_INC:
            op_tok = self.consume()
            return PostfixExpr(op="++", operand=node, line=op_tok.line)
        if self.peek_type() == T_DEC:
            op_tok = self.consume()
            return PostfixExpr(op="--", operand=node, line=op_tok.line)

        # Function call:  identifier(args)
        if isinstance(node, IdentifierExpr) and self.peek_type() == T_LPAREN:
            self.consume()  # eat '('
            args = self._parse_args()
            self.expect(T_RPAREN)
            return CallExpr(name=node.name, args=args, line=node.line)

        # Array access:  identifier[index]
        if isinstance(node, IdentifierExpr) and self.peek_type() == T_LBRACKET:
            self.consume()  # eat '['
            index = self.parse_expr()
            self.expect(T_RBRACKET)
            return ArrayAccessExpr(name=node.name, index=index, line=node.line)

        return node

    def _parse_args(self) -> List[Any]:
        """Parse a comma-separated argument list (possibly empty)."""
        args: List[Any] = []
        if self.peek_type() == T_RPAREN:
            return args
        args.append(self.parse_expr())
        while self.match(T_COMMA):
            args.append(self.parse_expr())
        return args

    def parse_primary(self) -> Any:
        """primary → INTEGER_LITERAL | CHAR_LITERAL | STRING_LITERAL
                   | IDENTIFIER | '(' expr ')'
        """
        tok = self.peek()

        if tok.type == T_INTEGER_LITERAL:
            self.consume()
            return IntLiteral(value=int(tok.value), line=tok.line)

        if tok.type == T_CHAR_LITERAL:
            self.consume()
            return CharLiteral(value=tok.value, line=tok.line)

        if tok.type == T_STRING_LITERAL:
            self.consume()
            return StringLiteral(value=tok.value, line=tok.line)

        if tok.type == T_IDENTIFIER:
            self.consume()
            return IdentifierExpr(name=tok.value, line=tok.line)

        # Parenthesised expression
        if tok.type == T_LPAREN:
            self.consume()
            expr = self.parse_expr()
            self.expect(T_RPAREN)
            return expr

        self._error(f"unexpected token '{tok.value}' — expected expression")


# ──────────────────────────────────────────────────────────────────
# Public convenience function
# ──────────────────────────────────────────────────────────────────

def parse(tokens: List[Token], filename: str = "<stdin>") -> Program:
    """Parse a token list into a Program AST.

    This is the public entry point that the rest of the compiler
    imports and calls.
    """
    return Parser(tokens, filename).parse()
