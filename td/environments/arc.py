from typing import Tuple, Any, Dict, List
import numpy as np
from lark import Lark, Tree, Transformer
from td.environments.environment import Environment
from td.environments.goal_checker import BinaryIOUGoalChecker
from td.grammar import Compiler, Grammar
from functools import partial
import dsl

_grammar_spec = r"""
// Main program structure
s: operation | compose

// Composition of operations
compose: "(" "compose" " " s " " s ")"

// Basic operations
operation: transform | color_op | grid_op | patch_op | select_op

// Transform operations
transform: rotate | flip | move | scale
rotate: "(" "rot90" " " s ")" -> rot90
      | "(" "rot180" " " s ")" -> rot180
      | "(" "rot270" " " s ")" -> rot270
flip: "(" "hmirror" " " s ")" -> hmirror
    | "(" "vmirror" " " s ")" -> vmirror
    | "(" "dmirror" " " s ")" -> dmirror
move: "(" "move" " " s " " vector ")"
scale: "(" "upscale" " " s " " number ")" -> upscale
     | "(" "downscale" " " s " " number ")" -> downscale

// Color operations
color_op: "(" "replace" " " s " " number " " number ")" -> replace
       | "(" "fill" " " s " " number " " patch ")" -> fill
       | "(" "paint" " " s " " object ")" -> paint
       | "(" "recolor" " " s " " object ")" -> recolor

// Grid operations
grid_op: split | concat | crop
split: "(" "hsplit" " " s " " number ")" -> hsplit
     | "(" "vsplit" " " s " " number ")" -> vsplit
concat: "(" "hconcat" " " s " " s ")" -> hconcat
      | "(" "vconcat" " " s " " s ")" -> vconcat
crop: "(" "crop" " " s " " vector " " vector ")" -> crop

// Patch operations
patch_op: "(" "normalize" " " patch ")" -> normalize
        | "(" "center" " " patch ")" -> center
        | "(" "box" " " patch ")" -> box

// Selection operations
select_op: "(" "objects" " " s " " boolean " " boolean " " boolean ")" -> objects
        | "(" "partition" " " s ")" -> partition
        | "(" "ofcolor" " " s " " number ")" -> ofcolor

// Basic types
number: DIGIT -> to_int
boolean: "true" -> true | "false" -> false
vector: "(" number " " number ")"
patch: "[patch]" // Placeholder - actual patches will be constructed from operations
object: "[object]" // Placeholder - actual objects will be constructed from operations

DIGIT: /[0-9]/

%ignore /[\t\n\f\r]+/
"""

class ARCTransformer(Transformer):
    """Transforms parsed tree into executable operations"""
    
    def __init__(self):
        super().__init__()
        self.grid = None
        
    def set_grid(self, grid):
        self.grid = grid
        
    def s(self, items):
        return items[0]
        
    def compose(self, items):
        def composed_fn(grid):
            intermediate = items[0](grid)
            return items[1](intermediate)
        return composed_fn
    
    def to_int(self, items):
        return int(items[0])
    
    def true(self, _):
        return True
        
    def false(self, _):
        return False
        
    def vector(self, items):
        return (items[0], items[1])
        
    # Transform operations
    def rot90(self, items):
        return lambda grid: dsl.rot90(grid)
        
    def rot180(self, items):
        return lambda grid: dsl.rot180(grid)
        
    def rot270(self, items):
        return lambda grid: dsl.rot270(grid)
        
    def hmirror(self, items):
        return lambda grid: dsl.hmirror(grid)
        
    def vmirror(self, items):
        return lambda grid: dsl.vmirror(grid)
        
    def dmirror(self, items):
        return lambda grid: dsl.dmirror(grid)
        
    def move(self, items):
        operation, offset = items
        return lambda grid: dsl.move(grid, operation(grid), offset)
        
    def upscale(self, items):
        operation, factor = items
        return lambda grid: dsl.upscale(operation(grid), factor)
        
    def downscale(self, items):
        operation, factor = items
        return lambda grid: dsl.downscale(operation(grid), factor)
    
    # Color operations
    def replace(self, items):
        operation, old_color, new_color = items
        return lambda grid: dsl.replace(operation(grid), old_color, new_color)
        
    def fill(self, items):
        operation, color, patch = items
        return lambda grid: dsl.fill(operation(grid), color, patch(grid))
        
    def paint(self, items):
        operation, obj = items
        return lambda grid: dsl.paint(operation(grid), obj(grid))
        
    def recolor(self, items):
        value, obj = items
        return lambda grid: dsl.recolor(value, obj(grid))
    
    # Grid operations
    def hsplit(self, items):
        operation, n = items
        return lambda grid: dsl.hsplit(operation(grid), n)
        
    def vsplit(self, items):
        operation, n = items
        return lambda grid: dsl.vsplit(operation(grid), n)
        
    def hconcat(self, items):
        return lambda grid: dsl.hconcat(items[0](grid), items[1](grid))
        
    def vconcat(self, items):
        return lambda grid: dsl.vconcat(items[0](grid), items[1](grid))
        
    def crop(self, items):
        operation, start, dims = items
        return lambda grid: dsl.crop(operation(grid), start, dims)
        
    # Selection operations
    def objects(self, items):
        operation, univalued, diagonal, without_bg = items
        return lambda grid: dsl.objects(operation(grid), univalued, diagonal, without_bg)
        
    def partition(self, items):
        return lambda grid: dsl.partition(items[0](grid))
        
    def ofcolor(self, items):
        operation, color = items
        return lambda grid: dsl.ofcolor(operation(grid), color)

class ARCCompiler(Compiler):
    def __init__(self):
        super().__init__()
        self.transformer = ARCTransformer()
        
    def compile(self, expression: Tree) -> np.ndarray:
        """Compiles the expression tree into a grid transformation function"""
        # Transform the expression tree into a callable function
        transform_fn = self.transformer.transform(expression)
        
        # Initialize an empty grid if needed (or use provided input grid)
        if self.transformer.grid is None:
            self.transformer.grid = dsl.canvas(0, (30, 30))
            
        # Apply the transformation
        result = transform_fn(self.transformer.grid)
        
        # Convert the result to numpy array
        if isinstance(result, tuple):  # If it's a grid
            return np.array(result)
        # If it's a patch/object/indices, convert to a binary grid
        grid = np.zeros((30, 30), dtype=np.int32)
        if hasattr(result, '__iter__'):
            for item in result:
                if isinstance(item, tuple) and len(item) == 2:
                    i, j = item
                    if 0 <= i < 30 and 0 <= j < 30:
                        grid[i, j] = 1
        return grid

class ARC(Environment):
    def __init__(self):
        super().__init__()
        self._grammar = Grammar(
            _grammar_spec,
            start="s",
            primitives=["fill", "replace", "move", "objects", "partition"]
        )
        self._compiler = ARCCompiler()
        self._goal_checker = BinaryIOUGoalChecker(threshold=0.99)

    @property
    def grammar(self) -> Grammar:
        return self._grammar

    @property
    def compiler(self) -> Compiler:
        return self._compiler

    @property
    def compiled_shape(self) -> Tuple[int, ...]:
        return (30, 30, 1)  # Standard ARC grid size

    @classmethod
    def name(cls) -> str:
        return "arc"

    def goal_reached(self, compiledA, compiledB) -> bool:
        return self._goal_checker.goal_reached(compiledA, compiledB)

    def set_input_grid(self, grid):
        """Sets the input grid for the compiler to transform"""
        self._compiler.transformer.set_grid(grid)