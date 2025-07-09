from typing import Any, Callable
from enum import Enum

class Color(Enum):
    RED = 1
    BLACK = 2

class Node:
    def __init__(self, item: Any, color: Color = Color.RED):
        self.item = item
        self.color = color
        self.left = None
        self.right = None
        self.parent = None

class GrokRBT:
    def __init__(self, comparator: Callable[[Any, Any], int]):
        self.comparator = comparator
        self.root = None
        self._size = 0

    def add(self, item: Any) -> None:
        if self.contains(item):
            raise ValueError("Item already exists in the tree")

        new_node = Node(item)
        if not self.root:
            self.root = new_node
            self.root.color = Color.BLACK
            self._size = 1
            return

        current = self.root
        while True:
            comp = self.comparator(item, current.item)
            if comp < 0:
                if not current.left:
                    current.left = new_node
                    new_node.parent = current
                    break
                current = current.left
            else:
                if not current.right:
                    current.right = new_node
                    new_node.parent = current
                    break
                current = current.right

        self._size += 1
        self._fix_after_insertion(new_node)

    def _fix_after_insertion(self, node: Node) -> None:
        while node != self.root and node.parent.color == Color.RED:
            parent = node.parent
            grandparent = parent.parent

            if parent == grandparent.left:
                uncle = grandparent.right
                if uncle and uncle.color == Color.RED:
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    node = grandparent
                else:
                    if node == parent.right:
                        node = parent
                        self._rotate_left(node)
                        parent = node.parent
                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    self._rotate_right(grandparent)
            else:
                uncle = grandparent.left
                if uncle and uncle.color == Color.RED:
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    node = grandparent
                else:
                    if node == parent.left:
                        node = parent
                        self._rotate_right(node)
                        parent = node.parent
                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    self._rotate_left(grandparent)

        self.root.color = Color.BLACK

    def _rotate_left(self, node: Node) -> None:
        right = node.right
        node.right = right.left
        if right.left:
            right.left.parent = node
        right.parent = node.parent
        if not node.parent:
            self.root = right
        elif node == node.parent.left:
            node.parent.left = right
        else:
            node.parent.right = right
        right.left = node
        node.parent = right

    def _rotate_right(self, node: Node) -> None:
        left = node.left
        node.left = left.right
        if left.right:
            left.right.parent = node
        left.parent = node.parent
        if not node.parent:
            self.root = left
        elif node == node.parent.right:
            node.parent.right = left
        else:
            node.parent.left = left
        left.right = node
        node.parent = left

    def contains(self, item: Any) -> bool:
        current = self.root
        while current:
            comp = self.comparator(item, current.item)
            if comp == 0:
                return True
            elif comp < 0:
                current = current.left
            else:
                current = current.right
        return False

    def remove(self, item: Any) -> Any:
        node = self._find_node(item)
        if not node:
            raise ValueError("Item not found in the tree")

        result = node.item
        self._delete_node(node)
        self._size -= 1
        return result

    def _find_node(self, item: Any) -> Node:
        current = self.root
        while current:
            comp = self.comparator(item, current.item)
            if comp == 0:
                return current
            elif comp < 0:
                current = current.left
            else:
                current = current.right
        return None

    def _delete_node(self, node: Node) -> None:
        if node.left and node.right:
            successor = self._find_min(node.right)
            node.item = successor.item
            node = successor

        replacement = node.left if node.left else node.right
        if replacement:
            replacement.parent = node.parent
            if not node.parent:
                self.root = replacement
            elif node == node.parent.left:
                node.parent.left = replacement
            else:
                node.parent.right = replacement
            if node.color == Color.BLACK:
                self._fix_after_deletion(replacement)
        elif node == self.root:
            self.root = None
        else:
            if node.color == Color.BLACK:
                self._fix_after_deletion(node)
            if node.parent:
                if node == node.parent.left:
                    node.parent.left = None
                else:
                    node.parent.right = None

    def _fix_after_deletion(self, node: Node) -> None:
        while node != self.root and (not node or node.color == Color.BLACK):
            parent = node.parent if node else node.parent
            if node == parent.left or (not node and parent.left):
                sibling = parent.right
                if sibling.color == Color.RED:
                    sibling.color = Color.BLACK
                    parent.color = Color.RED
                    self._rotate_left(parent)
                    sibling = parent.right
                if (not sibling.left or sibling.left.color == Color.BLACK) and \
                        (not sibling.right or sibling.right.color == Color.BLACK):
                    sibling.color = Color.RED
                    node = parent
                else:
                    if not sibling.right or sibling.right.color == Color.BLACK:
                        if sibling.left:
                            sibling.left.color = Color.BLACK
                        sibling.color = Color.RED
                        self._rotate_right(sibling)
                        sibling = parent.right
                    sibling.color = parent.color
                    parent.color = Color.BLACK
                    if sibling.right:
                        sibling.right.color = Color.BLACK
                    self._rotate_left(parent)
                    node = self.root
            else:
                sibling = parent.left
                if sibling.color == Color.RED:
                    sibling.color = Color.BLACK
                    parent.color = Color.RED
                    self._rotate_right(parent)
                    sibling = parent.left
                if (not sibling.right or sibling.right.color == Color.BLACK) and \
                        (not sibling.left or sibling.left.color == Color.BLACK):
                    sibling.color = Color.RED
                    node = parent
                else:
                    if not sibling.left or sibling.left.color == Color.BLACK:
                        if sibling.right:
                            sibling.right.color = Color.BLACK
                        sibling.color = Color.RED
                        self._rotate_left(sibling)
                        sibling = parent.left
                    sibling.color = parent.color
                    parent.color = Color.BLACK
                    if sibling.left:
                        sibling.left.color = Color.BLACK
                    self._rotate_right(parent)
                    node = self.root
        if node:
            node.color = Color.BLACK

    def _find_min(self, node: Node) -> Node:
        current = node
        while current.left:
            current = current.left
        return current

    def remove_first(self) -> Any:
        if not self.root:
            raise ValueError("Tree is empty")

        node = self._find_min(self.root)
        result = node.item
        self._delete_node(node)
        self._size -= 1
        return result

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self.root = None
        self._size = 0