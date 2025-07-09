from typing import Any, Callable, Optional

class RBTNode:
    """Node class for Red-Black Tree"""
    def __init__(self, data: Any, color: str = "RED"):
        self.data = data
        self.color = color  # "RED" or "BLACK"
        self.left: Optional['RBTNode'] = None
        self.right: Optional['RBTNode'] = None
        self.parent: Optional['RBTNode'] = None

class ClaudeRBT:
    """Red-Black Tree implementation"""

    def __init__(self, comparator: Callable[[int, int], int]):
        """
        Initialize Red-Black Tree with a comparator function.
        comparator(a, b) should return:
        - negative value if a < b
        - 0 if a == b
        - positive value if a > b
        """
        self.comparator = comparator
        self.root: Optional[RBTNode] = None
        self._size = 0

    def _compare(self, a: Any, b: Any) -> int:
        """Helper method to compare two items using the comparator"""
        return self.comparator(a, b)

    def _rotate_left(self, node: RBTNode) -> None:
        """Perform left rotation on the given node"""
        right_child = node.right
        node.right = right_child.left

        if right_child.left:
            right_child.left.parent = node

        right_child.parent = node.parent

        if not node.parent:
            self.root = right_child
        elif node == node.parent.left:
            node.parent.left = right_child
        else:
            node.parent.right = right_child

        right_child.left = node
        node.parent = right_child

    def _rotate_right(self, node: RBTNode) -> None:
        """Perform right rotation on the given node"""
        left_child = node.left
        node.left = left_child.right

        if left_child.right:
            left_child.right.parent = node

        left_child.parent = node.parent

        if not node.parent:
            self.root = left_child
        elif node == node.parent.right:
            node.parent.right = left_child
        else:
            node.parent.left = left_child

        left_child.right = node
        node.parent = left_child

    def _fix_insert(self, node: RBTNode) -> None:
        """Fix Red-Black Tree properties after insertion"""
        while node.parent and node.parent.color == "RED":
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right

                if uncle and uncle.color == "RED":
                    # Case 1: Uncle is red
                    node.parent.color = "BLACK"
                    uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Node is right child
                        node = node.parent
                        self._rotate_left(node)

                    # Case 3: Node is left child
                    node.parent.color = "BLACK"
                    node.parent.parent.color = "RED"
                    self._rotate_right(node.parent.parent)
            else:
                uncle = node.parent.parent.left

                if uncle and uncle.color == "RED":
                    # Case 1: Uncle is red
                    node.parent.color = "BLACK"
                    uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        # Case 2: Node is left child
                        node = node.parent
                        self._rotate_right(node)

                    # Case 3: Node is right child
                    node.parent.color = "BLACK"
                    node.parent.parent.color = "RED"
                    self._rotate_left(node.parent.parent)

        self.root.color = "BLACK"

    def _transplant(self, u: RBTNode, v: Optional[RBTNode]) -> None:
        """Replace subtree rooted at u with subtree rooted at v"""
        if not u.parent:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v

        if v:
            v.parent = u.parent

    def _minimum(self, node: RBTNode) -> RBTNode:
        """Find minimum node in subtree"""
        while node.left:
            node = node.left
        return node

    def _fix_delete(self, node: Optional[RBTNode]) -> None:
        """Fix Red-Black Tree properties after deletion"""
        while node != self.root and (not node or node.color == "BLACK"):
            if not node:
                # If node is None, we can't continue the fixing process
                break

            if node == node.parent.left:
                sibling = node.parent.right

                if sibling and sibling.color == "RED":
                    # Case 1: Sibling is red
                    sibling.color = "BLACK"
                    node.parent.color = "RED"
                    self._rotate_left(node.parent)
                    sibling = node.parent.right

                if sibling and ((not sibling.left or sibling.left.color == "BLACK") and
                                (not sibling.right or sibling.right.color == "BLACK")):
                    # Case 2: Both children of sibling are black
                    sibling.color = "RED"
                    node = node.parent
                else:
                    if sibling and (not sibling.right or sibling.right.color == "BLACK"):
                        # Case 3: Sibling's right child is black
                        if sibling.left:
                            sibling.left.color = "BLACK"
                        sibling.color = "RED"
                        self._rotate_right(sibling)
                        sibling = node.parent.right

                    # Case 4: Sibling's right child is red
                    if sibling:
                        sibling.color = node.parent.color
                        node.parent.color = "BLACK"
                        if sibling.right:
                            sibling.right.color = "BLACK"
                        self._rotate_left(node.parent)
                    node = self.root
            else:
                sibling = node.parent.left

                if sibling and sibling.color == "RED":
                    # Case 1: Sibling is red
                    sibling.color = "BLACK"
                    node.parent.color = "RED"
                    self._rotate_right(node.parent)
                    sibling = node.parent.left

                if sibling and ((not sibling.right or sibling.right.color == "BLACK") and
                                (not sibling.left or sibling.left.color == "BLACK")):
                    # Case 2: Both children of sibling are black
                    sibling.color = "RED"
                    node = node.parent
                else:
                    if sibling and (not sibling.left or sibling.left.color == "BLACK"):
                        # Case 3: Sibling's left child is black
                        if sibling.right:
                            sibling.right.color = "BLACK"
                        sibling.color = "RED"
                        self._rotate_left(sibling)
                        sibling = node.parent.left

                    # Case 4: Sibling's left child is red
                    if sibling:
                        sibling.color = node.parent.color
                        node.parent.color = "BLACK"
                        if sibling.left:
                            sibling.left.color = "BLACK"
                        self._rotate_right(node.parent)
                    node = self.root

        if node:
            node.color = "BLACK"

    def _find_node(self, item: Any) -> Optional[RBTNode]:
        """Find node with the given item"""
        current = self.root
        while current:
            cmp = self._compare(item, current.data)
            if cmp == 0:
                return current
            elif cmp < 0:
                current = current.left
            else:
                current = current.right
        return None

    def add(self, item: Any) -> None:
        """Add item to the tree. Raises ValueError if item already exists."""
        if not self.root:
            self.root = RBTNode(item, "BLACK")
            self._size += 1
            return

        current = self.root
        parent = None

        while current:
            parent = current
            cmp = self._compare(item, current.data)
            if cmp == 0:
                raise ValueError("Item already exists in the tree")
            elif cmp < 0:
                current = current.left
            else:
                current = current.right

        new_node = RBTNode(item, "RED")
        new_node.parent = parent

        if self._compare(item, parent.data) < 0:
            parent.left = new_node
        else:
            parent.right = new_node

        self._size += 1
        self._fix_insert(new_node)

    def remove(self, item: Any) -> Any:
        """Remove and return item from tree. Raises ValueError if item doesn't exist."""
        node = self._find_node(item)
        if not node:
            raise ValueError("Item not found in the tree")

        data = node.data
        original_color = node.color

        if not node.left:
            fix_node = node.right
            self._transplant(node, node.right)
        elif not node.right:
            fix_node = node.left
            self._transplant(node, node.left)
        else:
            successor = self._minimum(node.right)
            original_color = successor.color
            fix_node = successor.right

            if successor.parent == node:
                if fix_node:
                    fix_node.parent = successor
            else:
                self._transplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor

            self._transplant(node, successor)
            successor.left = node.left
            successor.left.parent = successor
            successor.color = node.color

        self._size -= 1

        if original_color == "BLACK":
            self._fix_delete(fix_node)

        return data

    def remove_first(self) -> Any:
        """Remove and return the first (minimum) item. Raises ValueError if tree is empty."""
        if not self.root:
            raise ValueError("Tree is empty")

        min_node = self._minimum(self.root)
        return self.remove(min_node.data)

    def contains(self, item: Any) -> bool:
        """Check if item exists in the tree."""
        return self._find_node(item) is not None

    def size(self) -> int:
        """Return the number of items in the tree."""
        return self._size

    def clear(self) -> None:
        """Clear all items from the tree."""
        self.root = None
        self._size = 0


# Example usage and testing
if __name__ == "__main__":
    # Create a comparator for integers
    def int_comparator(a: int, b: int) -> int:
        return a - b

    # Create and test the Red-Black Tree
    rbt = RBT(int_comparator)

    # Test adding elements
    items = [20, 10, 30, 5, 15, 25, 35]
    for item in items:
        rbt.add(item)
        print(f"Added {item}, size: {rbt.size()}")

    # Test contains
    print(f"Contains 15: {rbt.contains(15)}")
    print(f"Contains 40: {rbt.contains(40)}")

    # Test removing elements
    print(f"Removed: {rbt.remove(15)}")
    print(f"Size after removal: {rbt.size()}")

    # Test remove_first
    print(f"First item removed: {rbt.remove_first()}")
    print(f"Size after removing first: {rbt.size()}")

    # Test clear
    rbt.clear()
    print(f"Size after clear: {rbt.size()}")

    # Test error cases
    try:
        rbt.remove_first()  # Should raise ValueError
    except ValueError as e:
        print(f"Expected error: {e}")

    rbt.add(10)
    try:
        rbt.add(10)  # Should raise ValueError
    except ValueError as e:
        print(f"Expected error: {e}")

    try:
        rbt.remove(20)  # Should raise ValueError
    except ValueError as e:
        print(f"Expected error: {e}")