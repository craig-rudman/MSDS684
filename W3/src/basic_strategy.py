class BasicStrategy:
    """Standard Blackjack basic strategy for hit/stand decisions only.

    Simplified to the two actions available in Gymnasium's Blackjack-v1:
    0 = stick, 1 = hit. Does not include double-down or split.
    """

    def __init__(self):
        # Hard totals: (player_sum, dealer_card) -> action
        # Dealer card 1 = Ace
        self._hard = {}
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                if player_sum >= 17:
                    self._hard[(player_sum, dealer_card)] = 0  # stand
                elif player_sum <= 11:
                    self._hard[(player_sum, dealer_card)] = 1  # hit
                elif player_sum == 12:
                    if dealer_card in (4, 5, 6):
                        self._hard[(player_sum, dealer_card)] = 0  # stand
                    else:
                        self._hard[(player_sum, dealer_card)] = 1  # hit
                else:  # 13-16
                    if dealer_card in (2, 3, 4, 5, 6):
                        self._hard[(player_sum, dealer_card)] = 0  # stand
                    else:
                        self._hard[(player_sum, dealer_card)] = 1  # hit

        # Soft totals: (player_sum, dealer_card) -> action
        self._soft = {}
        for player_sum in range(12, 22):
            for dealer_card in range(1, 11):
                if player_sum >= 19:
                    self._soft[(player_sum, dealer_card)] = 0  # stand
                elif player_sum == 18:
                    if dealer_card in (9, 10, 1):
                        self._soft[(player_sum, dealer_card)] = 1  # hit
                    else:
                        self._soft[(player_sum, dealer_card)] = 0  # stand
                else:  # 12-17
                    self._soft[(player_sum, dealer_card)] = 1  # hit

    def get_action(self, state: tuple) -> int:
        """Return the basic strategy action for a given state.

        Args:
            state: (player_sum, dealer_card, usable_ace) tuple.

        Returns:
            0 (stick) or 1 (hit).
        """
        player_sum, dealer_card, usable_ace = state
        if usable_ace:
            return self._soft.get((player_sum, dealer_card), 1)
        return self._hard.get((player_sum, dealer_card), 1)

    def select_action(self, state: tuple) -> int:
        """Compatible with EpisodeRunner - delegates to get_action."""
        return self.get_action(state)

    def update(self, episode: list) -> None:
        """No-op. BasicStrategy does not learn."""
        pass

