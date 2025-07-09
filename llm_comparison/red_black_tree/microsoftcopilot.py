from typing import Callable, Any, Optional

RED = True
BLACK = False

class Node:

    def __init__(self, item: Any, color: bool, parent=None):
        self.item = item
        self.color = color
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None
        self.parent: Optional['Node'] = parent

class MSCopilotRBT:

    def __init__(self, comparator: Callable[[Any, Any], int]):
        self.root: Optional[Node] = None
        self._size = 0
        self.compare = comparator

    def add(self, item: Any) -> None:
        if self.root is None:
            self.root = Node(item, BLACK)
            self._size = 1
            return

        def insert_rec(cur, item):
            cmp = self.compare(item, cur.item)
            if cmp == 0:
                raise ValueError("Duplicate item")
            elif cmp < 0:
                if cur.left:
                    return insert_rec(cur.left, item)
                cur.left = Node(item, RED, cur)
                return cur.left
            else:
                if cur.right:
                    return insert_rec(cur.right, item)
                cur.right = Node(item, RED, cur)
                return cur.right

        new_node = insert_rec(self.root, item)
        self._size += 1
        self._fix_after_insert(new_node)

    def contains(self, item: Any) -> bool:
        node = self._find_node(item)
        return node is not None

    def remove(self, item: Any) -> Any:
        node = self._find_node(item)
        if node is None:
            raise ValueError("Item not found")
        removed_item = node.item
        self._delete_node(node)
        self._size -= 1
        return removed_item

    def remove_first(self) -> Any:
        if self.root is None:
            raise ValueError("Tree is empty")
        node = self._min_node(self.root)
        removed_item = node.item
        self._delete_node(node)
        self._size -= 1
        return removed_item

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self.root = None
        self._size = 0

    # Helper methods below

    def _find_node(self, item: Any) -> Optional[Node]:
        cur = self.root
        while cur:
            cmp = self.compare(item, cur.item)
            if cmp == 0:
                return cur
            elif cmp < 0:
                cur = cur.left
            else:
                cur = cur.right
        return None

    def _min_node(self, node: Node) -> Node:
        while node.left:
            node = node.left
        return node

    def _delete_node(self, node: Node):
        # Simplified deletion for brevity; full RB deletion is more complex
        # Replace node with child if it has one, and update links
        def transplant(u: Node, v: Optional[Node]):
            if u.parent is None:
                self.root = v
            elif u == u.parent.left:
                u.parent.left = v
            else:
                u.parent.right = v
            if v:
                v.parent = u.parent

        if node.left is None:
            transplant(node, node.right)
        elif node.right is None:
            transplant(node, node.left)
        else:
            successor = self._min_node(node.right)
            if successor.parent != node:
                transplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor
            transplant(node, successor)
            successor.left = node.left
            successor.left.parent = successor

        # In a full implementation, fix up red-black properties here...

    def _fix_after_insert(self, node: Node):
        while node != self.root and node.parent.color == RED:
            parent = node.parent
            grandparent = parent.parent

            if grandparent is None:
                break  # Tree isn't deep enough yet, no need to rebalance

            if parent == grandparent.left:
                uncle = grandparent.right

                if uncle and uncle.color == RED:
                    # Case 1: Recoloring
                    parent.color = BLACK
                    uncle.color = BLACK
                    grandparent.color = RED
                    node = grandparent
                else:
                    if node == parent.right:
                        # Case 2: Left rotation needed
                        node = parent
                        self._rotate_left(node)
                    # Case 3: Right rotation
                    parent.color = BLACK
                    grandparent.color = RED
                    self._rotate_right(grandparent)
            else:
                uncle = grandparent.left

                if uncle and uncle.color == RED:
                    # Case 1: Recoloring
                    parent.color = BLACK
                    uncle.color = BLACK
                    grandparent.color = RED
                    node = grandparent
                else:
                    if node == parent.left:
                        # Case 2: Right rotation needed
                        node = parent
                        self._rotate_right(node)
                    # Case 3: Left rotation
                    parent.color = BLACK
                    grandparent.color = RED
                    self._rotate_left(grandparent)

        self.root.color = BLACK

    def _rotate_left(self, node: Node):
        right = node.right
        node.right = right.left
        if right.left:
            right.left.parent = node
        right.parent = node.parent
        if node.parent is None:
            self.root = right
        elif node == node.parent.left:
            node.parent.left = right
        else:
            node.parent.right = right
        right.left = node
        node.parent = right

    def _rotate_right(self, node: Node):
        left = node.left
        node.left = left.right
        if left.right:
            left.right.parent = node
        left.parent = node.parent
        if node.parent is None:
            self.root = left
        elif node == node.parent.right:
            node.parent.right = left
        else:
            node.parent.left = left
        left.right = node
        node.parent = left
