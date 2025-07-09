class ClaudeCharacterTrie:
    def __init__(self):
        """Initialize an empty trie."""
        self.root = {}
        self.end_marker = '__END__'

    def add(self, word: str) -> None:
        """Add a word to the trie.

        Args:
            word: The word to add to the trie
        """
        if not word:
            return

        current = self.root
        for char in word:
            if char not in current:
                current[char] = {}
            current = current[char]

        # Mark the end of a word
        current[self.end_marker] = True

    def contains(self, word: str) -> bool:
        """Check if the exact word exists in the trie.

        Args:
            word: The word to search for

        Returns:
            True if the word exists in the trie, False otherwise
        """
        if not word:
            return False

        current = self.root
        for char in word:
            if char not in current:
                return False
            current = current[char]

        # Check if this is actually the end of a word
        return self.end_marker in current

    def prefix(self, word: str) -> list[str]:
        """Find all words in the trie that start with the given prefix.

        Args:
            word: The prefix to search for

        Returns:
            A list of all words in the trie that start with the prefix,
            including the prefix itself if it's a complete word
        """
        if not word:
            # If empty prefix, return all words in trie
            return self._get_all_words(self.root, "")

        current = self.root
        # Navigate to the end of the prefix
        for char in word:
            if char not in current:
                return []  # Prefix not found
            current = current[char]

        # Collect all words starting from this node
        return self._get_all_words(current, word)

    def _get_all_words(self, node: dict, prefix: str) -> list[str]:
        """Helper method to collect all words from a given node.

        Args:
            node: The current node in the trie
            prefix: The current prefix built so far

        Returns:
            A list of all words that can be formed from this node
        """
        words = []

        # If this node marks the end of a word, add it to results
        if self.end_marker in node:
            words.append(prefix)

        # Recursively collect words from child nodes
        for char, child_node in node.items():
            if char != self.end_marker:
                words.extend(self._get_all_words(child_node, prefix + char))

        return words


# Example usage and testing
if __name__ == "__main__":
    # Create a trie and add some words
    trie = CharacterTrie()

    # Add words
    words_to_add = ["cat", "car", "card", "care", "careful", "cars", "carry", "dog", "dodge", "door"]
    for word in words_to_add:
        trie.add(word)

    # Test contains method
    print("Testing contains method:")
    print(f"Contains 'car': {trie.contains('car')}")  # True
    print(f"Contains 'ca': {trie.contains('ca')}")    # False
    print(f"Contains 'care': {trie.contains('care')}")  # True
    print(f"Contains 'careful': {trie.contains('careful')}")  # True
    print(f"Contains 'cat': {trie.contains('cat')}")  # True
    print(f"Contains 'bat': {trie.contains('bat')}")  # False

    print("\nTesting prefix method:")
    print(f"Prefix 'car': {trie.prefix('car')}")  # ['car', 'card', 'care', 'careful', 'cars', 'carry']
    print(f"Prefix 'ca': {trie.prefix('ca')}")    # ['cat', 'car', 'card', 'care', 'careful', 'cars', 'carry']
    print(f"Prefix 'do': {trie.prefix('do')}")    # ['dog', 'dodge', 'door']
    print(f"Prefix 'xyz': {trie.prefix('xyz')}")  # []
    print(f"Prefix 'careful': {trie.prefix('careful')}")  # ['careful']