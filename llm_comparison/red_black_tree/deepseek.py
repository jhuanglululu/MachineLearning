class DeepseekRBT:
    class _Node:
        __slots__ = ('key', 'left', 'right', 'parent', 'color')
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None
            self.parent = None
            self.color = 'RED'

    def __init__(self, comparator):
        self._nil = DeepseekRBT._Node(None)
        self._nil.color = 'BLACK'
        self._root = self._nil
        self._size = 0
        self._comparator = comparator

    def add(self, item):
        new_node = DeepseekRBT._Node(item)
        new_node.left = self._nil
        new_node.right = self._nil
        new_node.parent = self._nil

        current = self._root
        parent = self._nil
        while current != self._nil:
            parent = current
            cmp_val = self._comparator(item, current.key)
            if cmp_val < 0:
                current = current.left
            elif cmp_val > 0:
                current = current.right
            else:
                raise ValueError("Item already exists")

        new_node.parent = parent
        if parent == self._nil:
            self._root = new_node
        else:
            cmp_val = self._comparator(item, parent.key)
            if cmp_val < 0:
                parent.left = new_node
            else:
                parent.right = new_node

        self._size += 1
        self._add_fixup(new_node)

    def _add_fixup(self, node):
        while node.parent.color == 'RED':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._left_rotate(node)
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self._left_rotate(node.parent.parent)
        self._root.color = 'BLACK'

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self._nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == self._nil:
            self._root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, y):
        x = y.left
        y.left = x.right
        if x.right != self._nil:
            x.right.parent = y
        x.parent = y.parent
        if y.parent == self._nil:
            self._root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x

    def remove(self, item):
        node = self._search(item)
        if node == self._nil:
            raise ValueError("Item not found")
        return self._remove_node(node)

    def remove_first(self):
        if self._root == self._nil:
            raise ValueError("Tree is empty")
        node = self._minimum(self._root)
        return self._remove_node(node)

    def _remove_node(self, node):
        original_key = node.key
        y = node
        y_original_color = y.color
        if node.left == self._nil:
            x = node.right
            self._transplant(node, node.right)
        elif node.right == self._nil:
            x = node.left
            self._transplant(node, node.left)
        else:
            y = self._minimum(node.right)
            y_original_color = y.color
            x = y.right
            if y.parent == node:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = node.right
                y.right.parent = y
            self._transplant(node, y)
            y.left = node.left
            y.left.parent = y
            y.color = node.color
        if y_original_color == 'BLACK':
            self._remove_fixup(x)
        self._size -= 1
        return original_key

    def _remove_fixup(self, x):
        while x != self._root and x.color == 'BLACK':
            if x == x.parent.left:
                w = x.parent.right
                if w.color == 'RED':
                    w.color = 'BLACK'
                    x.parent.color = 'RED'
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if w.left.color == 'BLACK' and w.right.color == 'BLACK':
                    w.color = 'RED'
                    x = x.parent
                else:
                    if w.right.color == 'BLACK':
                        w.left.color = 'BLACK'
                        w.color = 'RED'
                        self._right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = 'BLACK'
                    w.right.color = 'BLACK'
                    self._left_rotate(x.parent)
                    x = self._root
            else:
                w = x.parent.left
                if w.color == 'RED':
                    w.color = 'BLACK'
                    x.parent.color = 'RED'
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if w.right.color == 'BLACK' and w.left.color == 'BLACK':
                    w.color = 'RED'
                    x = x.parent
                else:
                    if w.left.color == 'BLACK':
                        w.right.color = 'BLACK'
                        w.color = 'RED'
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = 'BLACK'
                    w.left.color = 'BLACK'
                    self._right_rotate(x.parent)
                    x = self._root
        x.color = 'BLACK'

    def _search(self, item):
        current = self._root
        while current != self._nil:
            cmp_val = self._comparator(item, current.key)
            if cmp_val < 0:
                current = current.left
            elif cmp_val > 0:
                current = current.right
            else:
                return current
        return self._nil

    def contains(self, item):
        return self._search(item) != self._nil

    def _minimum(self, node):
        current = node
        while current.left != self._nil:
            current = current.left
        return current

    def _transplant(self, u, v):
        if u.parent == self._nil:
            self._root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def size(self):
        return self._size

    def clear(self):
        self._root = self._nil
        self._size = 0