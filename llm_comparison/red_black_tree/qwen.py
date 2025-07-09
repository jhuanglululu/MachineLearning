from __future__ import annotations
from typing import Any, Callable, Optional

class RBTNode:
    def __init__(self, value: Any):
        self.value = value
        self.left: Optional[RBTNode] = None
        self.right: Optional[RBTNode] = None
        self.parent: Optional[RBTNode] = None
        self.color: str = 'red'  # Can be 'red' or 'black'

class QwenRBT:
    def __init__(self, comparator: Callable[[Any, Any], int]):
        self._comparator = comparator
        self._root: Optional[RBTNode] = None
        self._count = 0

    def _find_node(self, value: Any) -> Optional[RBTNode]:
        node = self._root
        while node:
            cmp_result = self._comparator(value, node.value)
            if cmp_result == 0:
                return node
            elif cmp_result < 0:
                node = node.left
            else:
                node = node.right
        return None

    def contains(self, item: Any) -> bool:
        return self._find_node(item) is not None

    def add(self, item: Any) -> None:
        if self.contains(item):
            raise ValueError("Item already exists in the tree.")

        node = RBTNode(item)
        self._count += 1

        if not self._root:
            self._root = node
            self._root.color = 'black'
            return

        current = self._root
        parent = None
        while current:
            parent = current
            cmp_result = self._comparator(item, current.value)
            if cmp_result < 0:
                if current.left:
                    current = current.left
                else:
                    current.left = node
                    node.parent = current
                    break
            else:
                if current.right:
                    current = current.right
                else:
                    current.right = node
                    node.parent = current
                    break

        self._insert_fixup(node)

    def _rotate_left(self, node: RBTNode) -> None:
        right = node.right
        node.right = right.left
        if right.left:
            right.left.parent = node
        right.parent = node.parent
        if not node.parent:
            self._root = right
        elif node == node.parent.left:
            node.parent.left = right
        else:
            node.parent.right = right
        right.left = node
        node.parent = right

    def _rotate_right(self, node: RBTNode) -> None:
        left = node.left
        node.left = left.right
        if left.right:
            left.right.parent = node
        left.parent = node.parent
        if not node.parent:
            self._root = left
        elif node == node.parent.right:
            node.parent.right = left
        else:
            node.parent.left = left
        left.right = node
        node.parent = left

    def _uncle(self, node: RBTNode) -> Optional[RBTNode]:
        if not node.parent or not node.parent.parent:
            return None
        if node.parent == node.parent.parent.left:
            return node.parent.parent.right
        else:
            return node.parent.parent.left

    def _sibling(self, node: RBTNode) -> Optional[RBTNode]:
        if not node.parent:
            return None
        if node == node.parent.left:
            return node.parent.right
        else:
            return node.parent.left

    def _insert_fixup(self, node: RBTNode) -> None:
        while node != self._root and node.parent and node.parent.color == 'red':
            uncle = self._uncle(node)
            if uncle and uncle.color == 'red':
                node.parent.color = 'black'
                uncle.color = 'black'
                node.parent.parent.color = 'red'
                node = node.parent.parent
            else:
                parent = node.parent
                grandparent = parent.parent
                if parent == grandparent.left and node == parent.right:
                    self._rotate_left(parent)
                    node = parent
                elif parent == grandparent.right and node == parent.left:
                    self._rotate_right(parent)
                    node = parent

                parent = node.parent
                grandparent = parent.parent
                if parent and grandparent:
                    parent.color = 'black'
                    grandparent.color = 'red'
                    if parent == grandparent.left:
                        self._rotate_right(grandparent)
                    else:
                        self._rotate_left(grandparent)
        self._root.color = 'black'

    def _replace_node(self, old_node: RBTNode, new_node: Optional[RBTNode]) -> None:
        if not old_node.parent:
            self._root = new_node
        elif old_node == old_node.parent.left:
            old_node.parent.left = new_node
        else:
            old_node.parent.right = new_node
        if new_node:
            new_node.parent = old_node.parent

    def _minimum(self, node: RBTNode) -> RBTNode:
        while node.left:
            node = node.left
        return node

    def _maximum(self, node: RBTNode) -> RBTNode:
        while node.right:
            node = node.right
        return node

    def _delete_one_child(self, node: RBTNode) -> None:
        child = node.right if node.left is None else node.left
        if not node.parent and not child:
            self._root = None
            return
        self._replace_node(node, child)
        if node.color == 'black':
            if child and child.color == 'red':
                child.color = 'black'
            else:
                self._delete_case2(child)

    def _delete_case2(self, node: Optional[RBTNode]) -> None:
        if not node or not node.parent:
            return
        sibling = self._sibling(node)
        if sibling and sibling.color == 'red':
            node.parent.color = 'red'
            sibling.color = 'black'
            if node == node.parent.left:
                self._rotate_left(node.parent)
            else:
                self._rotate_right(node.parent)
        self._delete_case3(node)

    def _delete_case3(self, node: Optional[RBTNode]) -> None:
        sibling = self._sibling(node)
        if (
                node and node.parent and
                node.parent.color == 'black' and
                sibling and sibling.color == 'black' and
                ((not sibling.left or sibling.left.color == 'black') and
                 (not sibling.right or sibling.right.color == 'black'))
        ):
            sibling.color = 'red'
            self._delete_case1(node.parent)
        else:
            self._delete_case4(node)

    def _delete_case1(self, node: RBTNode) -> None:
        if node.parent:
            self._delete_case2(node)

    def _delete_case4(self, node: Optional[RBTNode]) -> None:
        if not node or not node.parent or not node.parent.parent:
            return
        sibling = self._sibling(node)
        if node == node.parent.left and sibling and sibling.color == 'black' and \
                (not sibling.left or sibling.left.color == 'black') and \
                (sibling.right and sibling.right.color == 'red'):
            sibling.color = 'red'
            if sibling.right:
                sibling.right.color = 'black'
                self._rotate_left(sibling)
        elif node == node.parent.right and sibling and sibling.color == 'black' and \
                (not sibling.right or sibling.right.color == 'black') and \
                (sibling.left and sibling.left.color == 'red'):
            sibling.color = 'red'
            if sibling.left:
                sibling.left.color = 'black'
                self._rotate_right(sibling)

        self._delete_case5(node)

    def _delete_case5(self, node: Optional[RBTNode]) -> None:
        if not node or not node.parent:
            return
        sibling = self._sibling(node)
        if node == node.parent.left and sibling:
            if sibling.color == 'black' and \
                    (not sibling.right or sibling.right.color == 'black') and \
                    (sibling.left and sibling.left.color == 'red'):
                sibling.color = 'red'
                sibling.left.color = 'black'
                self._rotate_right(sibling)
        elif node == node.parent.right and sibling:
            if sibling.color == 'black' and \
                    (not sibling.left or sibling.left.color == 'black') and \
                    (sibling.right and sibling.right.color == 'red'):
                sibling.color = 'red'
                sibling.right.color = 'black'
                self._rotate_left(sibling)

        if sibling and node.parent:
            sibling.color = node.parent.color
            node.parent.color = 'black'
            if node == node.parent.left:
                if sibling.right:
                    sibling.right.color = 'black'
                self._rotate_left(node.parent)
            else:
                if sibling.left:
                    sibling.left.color = 'black'
                self._rotate_right(node.parent)

    def remove(self, item: Any) -> Any:
        node = self._find_node(item)
        if not node:
            raise ValueError("Item not found in the tree.")
        val = node.value
        self._delete_node(node)
        self._count -= 1
        return val

    def _delete_node(self, node: RBTNode) -> None:
        if node.left and node.right:
            successor = self._minimum(node.right)
            node.value = successor.value
            self._delete_node(successor)
            return
        self._delete_one_child(node)

    def remove_first(self) -> Any:
        if not self._root:
            raise ValueError("Tree is empty.")
        first = self._minimum(self._root)
        val = first.value
        self._delete_node(first)
        self._count -= 1
        return val

    def size(self) -> int:
        return self._count

    def clear(self) -> None:
        self._root = None
        self._count = 0