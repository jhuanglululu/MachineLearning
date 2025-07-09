from collections.abc import Callable

class GeminiRBT:
    """
    A self-balancing binary search tree based on the Red-Black Tree properties.
    """

    class _Node:
        """
        Internal class representing a node in the Red-Black Tree.
        """
        RED = True
        BLACK = False

        def __init__(self, item, color=BLACK, parent=None, left=None, right=None):
            self.item = item
            self.color = color
            self.parent = parent
            self.left = left
            self.right = right

    _RED = _Node.RED
    _BLACK = _Node.BLACK

    def __init__(self, comparator: Callable[[int, int], int]):
        """
        Initializes an empty Red Black Tree.

        Args:
            comparator: A callable that takes two items and returns:
                        - a negative integer if the first item is less than the second
                        - zero if the first item is equal to the second
                        - a positive integer if the first item is greater than the second
        """
        self._root = None
        self._size = 0
        self._comparator = comparator

    def _compare(self, item1: any, item2: any) -> int:
        """Helper to use the provided comparator."""
        return self._comparator(item1, item2)

    def add(self, item: any) -> None:
        """
        Adds the item into the tree.

        Raises:
            ValueError: If the same (based on the comparator) item already exists.
        """
        new_node = self._Node(item, color=self._RED)
        if self._root is None:
            self._root = new_node
            self._root.color = self._BLACK  # Root is always black
        else:
            current = self._root
            parent = None
            while current:
                parent = current
                cmp = self._compare(item, current.item)
                if cmp < 0:
                    current = current.left
                elif cmp > 0:
                    current = current.right
                else:
                    raise ValueError(f"Item '{item}' already exists in the tree.")

            new_node.parent = parent
            if self._compare(item, parent.item) < 0:
                parent.left = new_node
            else:
                parent.right = new_node

            self._fix_after_insertion(new_node)
        self._size += 1

    def _fix_after_insertion(self, node: _Node) -> None:
        """
        Restores Red-Black Tree properties after an insertion.
        """
        while node != self._root and node.parent.color == self._RED:
            if node.parent == node.parent.parent.left:  # Parent is a left child
                uncle = node.parent.parent.right
                if uncle and uncle.color == self._RED:  # Case 1: Uncle is red
                    node.parent.color = self._BLACK
                    uncle.color = self._BLACK
                    node.parent.parent.color = self._RED
                    node = node.parent.parent
                else:  # Case 2 & 3: Uncle is black
                    if node == node.parent.right:  # Case 2: Node is a right child
                        node = node.parent
                        self._rotate_left(node)
                    # Case 3: Node is a left child
                    node.parent.color = self._BLACK
                    node.parent.parent.color = self._RED
                    self._rotate_right(node.parent.parent)
            else:  # Parent is a right child
                uncle = node.parent.parent.left
                if uncle and uncle.color == self._RED:  # Case 1: Uncle is red
                    node.parent.color = self._BLACK
                    uncle.color = self._BLACK
                    node.parent.parent.color = self._RED
                    node = node.parent.parent
                else:  # Case 2 & 3: Uncle is black
                    if node == node.parent.left:  # Case 2: Node is a left child
                        node = node.parent
                        self._rotate_right(node)
                    # Case 3: Node is a right child
                    node.parent.color = self._BLACK
                    node.parent.parent.color = self._RED
                    self._rotate_left(node.parent.parent)
        self._root.color = self._BLACK  # Ensure root is always black

    def remove(self, item: any) -> any:
        """
        Removes and returns the same (based on the comparator) item.

        Raises:
            ValueError: If no such item exists.
        """
        node_to_remove = self._find_node(item)
        if not node_to_remove:
            raise ValueError(f"Item '{item}' not found in the tree.")

        original_item = node_to_remove.item  # Store the item before potential replacement

        if node_to_remove.left and node_to_remove.right:
            # Node has two children, find successor
            successor = self._find_min(node_to_remove.right)
            node_to_remove.item = successor.item  # Replace item with successor's item
            node_to_remove = successor  # Now, the successor is the node to truly remove

        # At this point, node_to_remove has at most one child
        child = node_to_remove.left if node_to_remove.left else node_to_remove.right

        if child:
            child.parent = node_to_remove.parent
            if node_to_remove.parent is None:
                self._root = child
            elif node_to_remove == node_to_remove.parent.left:
                node_to_remove.parent.left = child
            else:
                node_to_remove.parent.right = child

            if node_to_remove.color == self._BLACK:
                self._fix_after_deletion(child)
        else:  # node_to_remove is a leaf
            if node_to_remove == self._root:
                self._root = None
            else:
                if node_to_remove.color == self._BLACK:
                    self._fix_after_deletion(node_to_remove) # Pass the removed node to fix
                if node_to_remove.parent:
                    if node_to_remove == node_to_remove.parent.left:
                        node_to_remove.parent.left = None
                    else:
                        node_to_remove.parent.right = None
        self._size -= 1
        return original_item

    def _fix_after_deletion(self, node: _Node) -> None:
        """
        Restores Red-Black Tree properties after a deletion.
        """
        while node != self._root and node.color == self._BLACK:
            if node == node.parent.left:  # Node is a left child
                sibling = node.parent.right
                if sibling and sibling.color == self._RED:  # Case 1: Sibling is red
                    sibling.color = self._BLACK
                    node.parent.color = self._RED
                    self._rotate_left(node.parent)
                    sibling = node.parent.right  # Update sibling after rotation

                # Cases 2, 3, 4: Sibling is black
                if (not sibling.left or sibling.left.color == self._BLACK) and \
                        (not sibling.right or sibling.right.color == self._BLACK):
                    # Case 2: Sibling and both its children are black
                    sibling.color = self._RED
                    node = node.parent
                else:
                    if not sibling.right or sibling.right.color == self._BLACK:
                        # Case 3: Sibling is black, left child is red, right child is black
                        if sibling.left:
                            sibling.left.color = self._BLACK
                        sibling.color = self._RED
                        self._rotate_right(sibling)
                        sibling = node.parent.right  # Update sibling after rotation
                    # Case 4: Sibling is black, right child is red
                    sibling.color = node.parent.color
                    node.parent.color = self._BLACK
                    if sibling.right:
                        sibling.right.color = self._BLACK
                    self._rotate_left(node.parent)
                    node = self._root  # End of loop
            else:  # Node is a right child (symmetric cases)
                sibling = node.parent.left
                if sibling and sibling.color == self._RED:  # Case 1: Sibling is red
                    sibling.color = self._BLACK
                    node.parent.color = self._RED
                    self._rotate_right(node.parent)
                    sibling = node.parent.left  # Update sibling after rotation

                # Cases 2, 3, 4: Sibling is black
                if (not sibling.left or sibling.left.color == self._BLACK) and \
                        (not sibling.right or sibling.right.color == self._BLACK):
                    # Case 2: Sibling and both its children are black
                    sibling.color = self._RED
                    node = node.parent
                else:
                    if not sibling.left or sibling.left.color == self._BLACK:
                        # Case 3: Sibling is black, right child is red, left child is black
                        if sibling.right:
                            sibling.right.color = self._BLACK
                        sibling.color = self._RED
                        self._rotate_left(sibling)
                        sibling = node.parent.left  # Update sibling after rotation
                    # Case 4: Sibling is black, left child is red
                    sibling.color = node.parent.color
                    node.parent.color = self._BLACK
                    if sibling.left:
                        sibling.left.color = self._BLACK
                    self._rotate_right(node.parent)
                    node = self._root  # End of loop
        if node:
            node.color = self._BLACK


    def remove_first(self) -> any:
        """
        Removes and returns the first (based on the comparator) item.

        Raises:
            ValueError: If the tree is empty.
        """
        if self._root is None:
            raise ValueError("Tree is empty.")
        first_node = self._find_min(self._root)
        return self.remove(first_node.item)

    def contains(self, item: any) -> bool:
        """
        Returns if the same (based on the comparator) item exists.
        """
        return self._find_node(item) is not None

    def size(self) -> int:
        """
        Returns the size of the tree.
        """
        return self._size

    def clear(self) -> None:
        """
        Clears the tree.
        """
        self._root = None
        self._size = 0

    def _find_node(self, item: any) -> _Node | None:
        """
        Helper method to find a node with the given item.
        """
        current = self._root
        while current:
            cmp = self._compare(item, current.item)
            if cmp < 0:
                current = current.left
            elif cmp > 0:
                current = current.right
            else:
                return current
        return None

    def _find_min(self, node: _Node) -> _Node:
        """
        Helper method to find the minimum node in a subtree.
        """
        current = node
        while current.left:
            current = current.left
        return current

    def _rotate_left(self, node: _Node) -> None:
        """
        Performs a left rotation around the given node.
        """
        right_child = node.right
        node.right = right_child.left
        if right_child.left:
            right_child.left.parent = node
        right_child.parent = node.parent
        if node.parent is None:
            self._root = right_child
        elif node == node.parent.left:
            node.parent.left = right_child
        else:
            node.parent.right = right_child
        right_child.left = node
        node.parent = right_child

    def _rotate_right(self, node: _Node) -> None:
        """
        Performs a right rotation around the given node.
        """
        left_child = node.left
        node.left = left_child.right
        if left_child.right:
            left_child.right.parent = node
        left_child.parent = node.parent
        if node.parent is None:
            self._root = left_child
        elif node == node.parent.right:
            node.parent.right = left_child
        else:
            node.parent.left = left_child
        left_child.right = node
        node.parent = left_child

    # Optional: For testing and visualization (pre-order traversal)
    def _pre_order_traversal(self, node: _Node, result: list) -> None:
        if node:
            result.append(f"{node.item}({'R' if node.color else 'B'})")
            self._pre_order_traversal(node.left, result)
            self._pre_order_traversal(node.right, result)

    def to_list_pre_order(self) -> list:
        """
        Returns a list of items in pre-order traversal for visualization/testing.
        (Includes color 'R' for Red, 'B' for Black)
        """
        result = []
        self._pre_order_traversal(self._root, result)
        return result

    # Optional: For testing and debugging (in-order traversal)
    def _in_order_traversal(self, node: _Node, result: list) -> None:
        if node:
            self._in_order_traversal(node.left, result)
            result.append(node.item)
            self._in_order_traversal(node.right, result)

    def to_list_in_order(self) -> list:
        """
        Returns a sorted list of items in in-order traversal.
        """
        result = []
        self._in_order