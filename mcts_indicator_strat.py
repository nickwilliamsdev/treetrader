import pandas as pd
import numpy as np
import math
import random
from abc import ABC, abstractmethod
import itertools

# --- 1. Abstract Base Class for Trading Strategy ---

class TradingStrategy(ABC):
    """
    Abstract Base Class for a trading strategy.
    Subclass this to define your specific trading strategy.
    """
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the trading strategy with historical market data.

        Args:
            data (pd.DataFrame): Historical market data (e.g., OHLCV).
                                 Expected to have a 'Close' column.
        """
        self.data = data

    @abstractmethod
    def get_default_hyperparameters(self) -> dict:
        """
        Returns a dictionary of default hyperparameters for the strategy.
        These will be used to initialize the root node of the MCTS tree.
        """
        pass

    @abstractmethod
    def get_hyperparameter_ranges(self) -> dict:
        """
        Returns a dictionary defining the search space for each hyperparameter.
        Each key is a hyperparameter name, and its value is a list of
        possible discrete values or a tuple (min, max, step) for integer ranges.
        Example: {'short_window': [5, 10, 15], 'long_window': (20, 50, 5)}
        """
        pass

    @abstractmethod
    def evaluate(self, hyperparameters: dict) -> float:
        """
        Evaluates the trading strategy's performance given a set of hyperparameters.
        This method should run a backtest on the historical data and return a
        single scalar reward (e.g., Sharpe Ratio, total return, etc.).

        Args:
            hyperparameters (dict): A dictionary of hyperparameters to evaluate.

        Returns:
            float: A scalar value representing the strategy's performance (reward).
                   Higher values are better.
        """
        pass

# --- 2. Example Concrete Trading Strategy (Moving Average Crossover) ---

class ExampleTradingStrategy(TradingStrategy):
    """
    An example trading strategy using a simple Moving Average Crossover.
    It buys when the short moving average crosses above the long moving average,
    and sells when the short moving average crosses below the long moving average.
    The reward is the Sharpe Ratio of the simulated returns.
    """
    def get_default_hyperparameters(self) -> dict:
        """
        Default hyperparameters for the moving average crossover strategy.
        """
        return {
            'short_window': 10,
            'long_window': 30,
            'initial_capital': 100000,
            'transaction_cost_bps': 1 # 0.01% per trade
        }

    def get_hyperparameter_ranges(self) -> dict:
        """
        Defines the search space for the hyperparameters.
        'short_window' and 'long_window' are integer ranges with a step.
        'transaction_cost_bps' is a fixed value for this example, but could be a range.
        """
        return {
            'short_window': (5, 20, 5),  # (min, max, step)
            'long_window': (20, 60, 10), # (min, max, step)
            'initial_capital': [100000], # Fixed for this example
            'transaction_cost_bps': [1]  # Fixed for this example
        }

    def evaluate(self, hyperparameters: dict) -> float:
        """
        Evaluates the moving average crossover strategy.
        Calculates daily returns and then the Sharpe Ratio.

        Args:
            hyperparameters (dict): Dictionary with 'short_window', 'long_window',
                                    'initial_capital', 'transaction_cost_bps'.

        Returns:
            float: Sharpe Ratio of the strategy's returns. Returns -inf if std dev is zero.
        """
        short_window = hyperparameters['short_window']
        long_window = hyperparameters['long_window']
        initial_capital = hyperparameters['initial_capital']
        transaction_cost_bps = hyperparameters['transaction_cost_bps'] / 10000.0 # Convert bps to decimal

        if short_window >= long_window:
            # Invalid hyperparameter combination
            return -np.inf

        df = self.data.copy()
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()

        # Generate trading signals
        # 1 for buy (long position), -1 for sell (short position), 0 for hold
        df['Signal'] = 0
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
        df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = -1

        # Calculate position changes (trades)
        df['Position'] = df['Signal'].shift(1) # Position held from previous day's signal
        df['Position'].fillna(0, inplace=True) # Start with no position
        df['Trade'] = df['Position'].diff().abs() # 1 if a trade occurred (buy or sell)

        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()

        # Calculate strategy returns
        # Strategy return = position * daily_return - transaction_cost_if_trade
        df['Strategy_Return'] = df['Position'] * df['Daily_Return']
        df['Transaction_Cost'] = df['Trade'] * transaction_cost_bps
        df['Strategy_Return'] -= df['Transaction_Cost']

        # Remove NaN values introduced by rolling means and shifts
        df.dropna(inplace=True)

        if df.empty:
            return -np.inf # No valid data points for evaluation

        # Calculate cumulative returns for Sharpe Ratio
        cumulative_returns = (1 + df['Strategy_Return']).cumprod()

        # Calculate Sharpe Ratio
        # Assuming risk-free rate is 0 for simplicity
        # Annualized Sharpe Ratio = (Mean Daily Return - Risk-Free Rate) / Std Dev of Daily Returns * sqrt(Trading Days)
        daily_returns = df['Strategy_Return']
        if daily_returns.std() == 0:
            return -np.inf # Avoid division by zero if no volatility

        annualized_sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) # 252 trading days in a year

        return annualized_sharpe_ratio

# --- 3. Hyperparameter Node for MCTS Tree ---

class HyperparameterNode:
    """
    Represents a node in the Monte Carlo Tree Search for hyperparameter tuning.
    Each node corresponds to a specific set of hyperparameters.
    """
    def __init__(self, hyperparameters: dict, parent=None):
        """
        Initializes a new node in the MCTS tree.

        Args:
            hyperparameters (dict): The set of hyperparameters represented by this node.
            parent (HyperparameterNode, optional): The parent node. Defaults to None (for root).
        """
        self.hyperparameters = hyperparameters
        self.parent = parent
        self.children = []
        self.visits = 0 # Number of times this node has been visited
        self.total_reward = 0.0 # Sum of rewards obtained from simulations passing through this node

    def uct_value(self, exploration_constant: float) -> float:
        """
        Calculates the UCT (Upper Confidence Bound 1) value for this node.
        This value balances exploitation (average reward) and exploration (unvisited nodes).

        Formula: Q/N + C * sqrt(ln(N_parent) / N)

        Args:
            exploration_constant (float): The exploration constant (C) for UCT.

        Returns:
            float: The UCT value. Returns infinity for unvisited nodes to prioritize them.
        """
        if self.visits == 0:
            return float('inf') # Prioritize unvisited nodes

        # Exploitation term: average reward
        exploitation_term = self.total_reward / self.visits

        # Exploration term: based on parent visits and node visits
        exploration_term = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation_term + exploration_term

    def add_child(self, child_hyperparameters: dict):
        """
        Adds a new child node to this node.

        Args:
            child_hyperparameters (dict): The hyperparameters for the new child node.
        """
        child_node = HyperparameterNode(child_hyperparameters, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal(self) -> bool:
        """
        For hyperparameter tuning, a node is 'terminal' if it has been evaluated.
        However, in MCTS, a node is typically terminal if it represents a game-ending state.
        Here, we use it to indicate if a simulation has been run from this node.
        """
        return self.visits > 0 # A node is considered "evaluated" if it has been visited at least once.

# --- 4. MCTS Hyperparameter Tuner ---

class MCTSHyperparameterTuner:
    """
    Implements the Monte Carlo Tree Search algorithm for hyperparameter tuning.
    """
    def __init__(self, strategy: TradingStrategy, iterations: int, exploration_constant: float = 1.0):
        """
        Initializes the MCTS tuner.

        Args:
            strategy (TradingStrategy): An instance of the trading strategy to tune.
            iterations (int): The number of MCTS iterations to run.
            exploration_constant (float): The exploration constant (C) for UCT.
        """
        self.strategy = strategy
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.root = HyperparameterNode(self.strategy.get_default_hyperparameters())
        self.hyperparameter_ranges = self._process_hyperparameter_ranges(self.strategy.get_hyperparameter_ranges())
        self.all_possible_hyperparameters = self._generate_all_possible_hyperparameters()
        self.evaluated_hyperparameters = set() # To keep track of evaluated hyperparameter sets

    def _process_hyperparameter_ranges(self, raw_ranges: dict) -> dict:
        """
        Processes raw hyperparameter ranges into lists of discrete values.
        Handles (min, max, step) tuples for integer ranges.
        """
        processed_ranges = {}
        for param, values in raw_ranges.items():
            if isinstance(values, tuple) and len(values) == 3:
                # Handle integer ranges (min, max, step)
                min_val, max_val, step = values
                processed_ranges[param] = list(range(min_val, max_val + step, step))
            else:
                # Assume it's already a list of discrete values
                processed_ranges[param] = list(values)
        return processed_ranges

    def _generate_all_possible_hyperparameters(self) -> list[dict]:
        """
        Generates all possible combinations of hyperparameters from the defined ranges.
        This is used for expansion to ensure all valid combinations can be explored.
        """
        keys = self.hyperparameter_ranges.keys()
        values = self.hyperparameter_ranges.values()
        
        # Create a list of dictionaries for all combinations
        all_combinations = []
        for combination in itertools.product(*values):
            all_combinations.append(dict(zip(keys, combination)))
        return all_combinations

    def run_mcts(self):
        """
        Runs the MCTS hyperparameter tuning process for the specified number of iterations.
        """
        print(f"Starting MCTS hyperparameter tuning for {self.iterations} iterations...")
        for i in range(self.iterations):
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.iterations}")

            # 1. Selection: Traverse the tree to find the best node to expand/simulate
            node = self._select(self.root)

            # 2. Expansion: Add new child nodes if the selected node is not fully expanded
            if not node.is_terminal(): # Only expand if not yet simulated
                 self._expand(node)
                 # After expansion, we might have new children to explore.
                 # Select one of the newly expanded children for simulation.
                 # If no new children were added (e.g., all combinations explored from parent),
                 # we simulate the current node.
                 if node.children:
                     node = random.choice(node.children) # Pick a new child to simulate

            # 3. Simulation: Run a backtest with the chosen hyperparameters
            reward = self._simulate(node.hyperparameters)

            # 4. Backpropagation: Update node statistics from the simulated reward
            self._backpropagate(node, reward)

        print("\nMCTS tuning complete.")
        print("Best hyperparameters found:")
        best_node = self.get_best_hyperparameters()
        print(best_node.hyperparameters)
        print(f"With average reward: {best_node.total_reward / best_node.visits:.4f}")


    def _select(self, node: HyperparameterNode) -> HyperparameterNode:
        """
        Selects the best child node to traverse based on UCT values.
        Continues until a leaf node (unvisited or not fully expanded) is reached.
        """
        while node.children and node.visits > 0: # If node has children and has been visited
            best_child = None
            best_uct = -float('inf')

            for child in node.children:
                uct = child.uct_value(self.exploration_constant)
                if uct > best_uct:
                    best_uct = uct
                    best_child = child
            
            if best_child is None: # Should not happen if node.children is not empty
                break
            node = best_child
        return node

    def _expand(self, node: HyperparameterNode):
        """
        Expands the current node by adding new child nodes representing
        neighboring hyperparameter combinations that haven't been explored yet.
        """
        current_hps = node.hyperparameters
        
        # Find all possible next steps (modifications to current hyperparameters)
        # For simplicity, we'll try to change one hyperparameter at a time
        # to its next available value in the defined range.
        
        unexplored_children_hps = []
        
        # Iterate over each hyperparameter
        for param_name, current_value in current_hps.items():
            possible_values = self.hyperparameter_ranges.get(param_name)
            if possible_values is None:
                continue # Skip if no range defined (e.g., fixed parameter)

            current_value_idx = -1
            try:
                current_value_idx = possible_values.index(current_value)
            except ValueError:
                # Current value not in possible_values, this might happen if default_hps
                # are outside the defined ranges. Handle gracefully.
                pass

            # Try incrementing the value
            if current_value_idx != -1 and current_value_idx < len(possible_values) - 1:
                next_value = possible_values[current_value_idx + 1]
                new_hps = current_hps.copy()
                new_hps[param_name] = next_value
                
                # Check if this combination has already been added as a child or evaluated
                if not any(c.hyperparameters == new_hps for c in node.children) and \
                   frozenset(new_hps.items()) not in self.evaluated_hyperparameters:
                    unexplored_children_hps.append(new_hps)
            
            # Try decrementing the value
            if current_value_idx > 0:
                prev_value = possible_values[current_value_idx - 1]
                new_hps = current_hps.copy()
                new_hps[param_name] = prev_value
                
                # Check if this combination has already been added as a child or evaluated
                if not any(c.hyperparameters == new_hps for c in node.children) and \
                   frozenset(new_hps.items()) not in self.evaluated_hyperparameters:
                    unexplored_children_hps.append(new_hps)
        
        # If no direct neighbors, consider a random unexplored combination from the full space
        if not unexplored_children_hps:
            all_hps_set = {frozenset(d.items()) for d in self.all_possible_hyperparameters}
            explored_hps_set = {frozenset(c.hyperparameters.items()) for c in node.children} | self.evaluated_hyperparameters
            
            remaining_hps = list(all_hps_set - explored_hps_set)
            
            if remaining_hps:
                # Convert frozenset back to dict
                random_unexplored_hp_set = dict(list(random.choice(remaining_hps)))
                unexplored_children_hps.append(random_unexplored_hp_set)
        
        # Add a new child if there are unexplored options
        if unexplored_children_hps:
            # Pick one to expand, typically a random one if multiple
            new_child_hps = random.choice(unexplored_children_hps)
            node.add_child(new_child_hps)


    def _simulate(self, hyperparameters: dict) -> float:
        """
        Simulates the trading strategy with the given hyperparameters and returns the reward.
        This is the 'playout' step in MCTS.
        """
        # Add to evaluated set to avoid re-evaluating the same set
        self.evaluated_hyperparameters.add(frozenset(hyperparameters.items()))
        reward = self.strategy.evaluate(hyperparameters)
        return reward

    def _backpropagate(self, node: HyperparameterNode, reward: float):
        """
        Updates the visit counts and total rewards for all nodes in the path
        from the simulated node up to the root.
        """
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.total_reward += reward
            current_node = current_node.parent

    def get_best_hyperparameters(self) -> HyperparameterNode:
        """
        After running MCTS, returns the node with the highest average reward.
        This represents the best set of hyperparameters found.
        """
        best_node = self.root
        queue = [self.root]

        while queue:
            current_node = queue.pop(0)
            if current_node.visits > 0: # Only consider nodes that have been simulated
                if (current_node.total_reward / current_node.visits) > \
                   (best_node.total_reward / best_node.visits if best_node.visits > 0 else -float('inf')):
                    best_node = current_node
            queue.extend(current_node.children)
        return best_node

# --- Example Usage ---

if __name__ == "__main__":
    # 1. Generate some dummy historical data (replace with your actual data)
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
    data = pd.DataFrame({'Close': close_prices}, index=dates)

    print("Sample Data Head:")
    print(data.head())
    print("\n")

    # 2. Instantiate your trading strategy
    my_strategy = ExampleTradingStrategy(data)

    # 3. Instantiate the MCTS Hyperparameter Tuner
    # You might need to adjust iterations and exploration_constant based on your problem
    tuner = MCTSHyperparameterTuner(strategy=my_strategy, iterations=500, exploration_constant=math.sqrt(2))

    # 4. Run the MCTS tuning
    tuner.run_mcts()

    # 5. Get the best hyperparameters found
    best_hps_node = tuner.get_best_hyperparameters()
    print("\n--- Final Best Hyperparameters ---")
    print(f"Hyperparameters: {best_hps_node.hyperparameters}")
    print(f"Average Reward (Sharpe Ratio): {best_hps_node.total_reward / best_hps_node.visits:.4f}")
    print(f"Total Visits: {best_hps_node.visits}")

    # You can also inspect the root node's children to see explored paths
    # print("\nRoot Node Children (first few):")
    # for i, child in enumerate(tuner.root.children):
    #     if i >= 5: break
    #     print(f"  Child {i+1}: HPs={child.hyperparameters}, Avg Reward={child.total_reward / child.visits if child.visits > 0 else 'N/A'}")

```