from typing import Any, Callable, Optional

RED = True
BLACK = False

class Node:
    def __init__(self, item: Any, color=RED):
        self.item = item
        self.color = color
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None

class ChatGPTRBT:
    def __init__(self, comparator: Callable[[int, int], int]):
        self.root: Optional[Node] = None
        self.comparator = comparator
        self._size = 0

    def is_red(self, node: Optional[Node]) -> bool:
        return node is not None and node.color == RED

    def rotate_left(self, h: Node) -> Node:
        x = h.right
        h.right = x.left
        x.left = h
        x.color = h.color
        h.color = RED
        return x

    def rotate_right(self, h: Node) -> Node:
        x = h.left
        h.left = x.right
        x.right = h
        x.color = h.color
        h.color = RED
        return x

    def flip_colors(self, h: Node):
        h.color = RED
        if h.left is not None:
            h.left.color = BLACK
        if h.right is not None:
            h.right.color = BLACK

    def add(self, item: Any) -> None:
        def insert(h: Optional[Node], item: Any) -> Node:
            if h is None:
                return Node(item)

            cmp = self.comparator(item, h.item)
            if cmp < 0:
                h.left = insert(h.left, item)
            elif cmp > 0:
                h.right = insert(h.right, item)
            else:
                raise ValueError("Duplicate item")

            if self.is_red(h.right) and not self.is_red(h.left):
                h = self.rotate_left(h)
            if self.is_red(h.left) and self.is_red(h.left.left):
                h = self.rotate_right(h)
            if self.is_red(h.left) and self.is_red(h.right):
                self.flip_colors(h)

            return h

        self.root = insert(self.root, item)
        self.root.color = BLACK
        self._size += 1

    def contains(self, item: Any) -> bool:
        node = self.root
        while node:
            cmp = self.comparator(item, node.item)
            if cmp < 0:
                node = node.left
            elif cmp > 0:
                node = node.right
            else:
                return True
        return False

    def move_red_left(self, h: Node) -> Node:
        self.flip_colors(h)
        if h.right and self.is_red(h.right.left):
            h.right = self.rotate_right(h.right)
            h = self.rotate_left(h)
            self.flip_colors(h)
        return h

    def move_red_right(self, h: Node) -> Node:
        self.flip_colors(h)
        if h.left and self.is_red(h.left.left):
            h = self.rotate_right(h)
            self.flip_colors(h)
        return h

    def fix_up(self, h: Node) -> Node:
        if self.is_red(h.right):
            h = self.rotate_left(h)
        if self.is_red(h.left) and self.is_red(h.left.left):
            h = self.rotate_right(h)
        if self.is_red(h.left) and self.is_red(h.right):
            self.flip_colors(h)
        return h

    def min_node(self, h: Node) -> Node:
        while h.left:
            h = h.left
        return h

    def delete_min(self, h: Node) -> Optional[Node]:
        if h.left is None:
            return None
        if not self.is_red(h.left) and not self.is_red(h.left.left):
            h = self.move_red_left(h)
        h.left = self.delete_min(h.left)
        return self.fix_up(h)

    def remove(self, item: Any) -> Any:
        removed = [None]

        def delete(h: Optional[Node], item: Any) -> Optional[Node]:
            if h is None:
                return None
            if self.comparator(item, h.item) < 0:
                if h.left:
                    if not self.is_red(h.left) and not self.is_red(h.left.left):
                        h = self.move_red_left(h)
                    h.left = delete(h.left, item)
            else:
                if self.is_red(h.left):
                    h = self.rotate_right(h)
                if self.comparator(item, h.item) == 0:
                    removed[0] = h.item
                    if h.right is None:
                        return None
                    min_node = self.min_node(h.right)
                    h.item = min_node.item
                    h.right = self.delete_min(h.right)
                else:
                    if h.right:
                        if not self.is_red(h.right) and not self.is_red(h.right.left):
                            h = self.move_red_right(h)
                        h.right = delete(h.right, item)
            return self.fix_up(h)

        self.root = delete(self.root, item)
        if removed[0] is None:
            raise ValueError("Item not found")
        if self.root:
            self.root.color = BLACK
        self._size -= 1
        return removed[0]

    def remove_first(self) -> Any:
        if self.root is None:
            raise ValueError("Tree is empty")
        node = self.root
        while node.left:
            node = node.left
        return self.remove(node.item)

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self.root = None
        self._size = 0