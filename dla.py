import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import time
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# --- 1. Neural Network Model ---
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], num_classes=10):
        super(SimpleNN, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- 2. Apply layer mask to control backprop ---
def apply_layer_mask(model, layers_to_update):
    """
    Apply mask to update only the last 'layers_to_update' layers.
    """
    all_layers = [module for module in model.model.modules()
                  if isinstance(module, nn.Linear)]

    total_linear_layers = len(all_layers)
    layers_to_freeze = max(0, total_linear_layers - layers_to_update)

    for i, layer in enumerate(all_layers):
        requires_grad = (i >= layers_to_freeze)
        if hasattr(layer, 'weight'):
            layer.weight.requires_grad = requires_grad
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.requires_grad = requires_grad

# --- 3. Local training with heterogeneity simulation ---
def train_locally_heterogeneous(model, data_loader, layers_to_update, device,
                               client_speed=1.0, epochs=1, lr=0.01):
    """
    Train locally with simulated client heterogeneity
    """
    model = copy.deepcopy(model).to(device)
    apply_layer_mask(model, layers_to_update)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    start_time = time.time()
    model.train()

    total_loss = 0
    for epoch in range(epochs):
        for batch_idx, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Simulate different client speeds
            if client_speed < 1.0:  # Slower clients
                time.sleep(0.001 * (1.0 - client_speed))

    elapsed_time = (time.time() - start_time) / client_speed
    avg_loss = total_loss / (len(data_loader) * epochs)

    return model.state_dict(), elapsed_time, avg_loss

# --- 4. Improved Layer depth adaptation with stability ---
def adjust_layers_improved(elapsed_time, current_depth, client_id,
                         T_low=3.0, T_high=10.0, max_layers=4):
    """
    Improved layer adjustment with lenient thresholds and stability mechanism
    """
    # More lenient, client-specific thresholds
    client_T_low = T_low * (0.7 + client_id * 0.2)   # Less aggressive variation
    client_T_high = T_high * (0.8 + client_id * 0.3)  # More reasonable ranges

    if elapsed_time > client_T_high and current_depth > 1:
        return current_depth - 1
    elif elapsed_time < client_T_low and current_depth < max_layers:
        return current_depth + 1
    else:
        return current_depth

# --- 4b. Stability tracking for DLA ---
class DLAStabilityTracker:
    """
    Track adaptation history to prevent oscillation
    """
    def __init__(self, num_clients):
        self.adaptation_history = {cid: [] for cid in range(num_clients)}
        self.consecutive_suggestions = {cid: {'increase': 0, 'decrease': 0, 'stay': 0}
                                      for cid in range(num_clients)}

    def should_adapt(self, client_id, current_depth, suggested_depth, stability_threshold=2):
        """
        Determine if client should actually change depth based on stability
        """
        if suggested_depth == current_depth:
            # Reset counters if no change suggested
            self.consecutive_suggestions[client_id] = {'increase': 0, 'decrease': 0, 'stay': 0}
            return current_depth

        # Track suggestion type
        if suggested_depth > current_depth:
            self.consecutive_suggestions[client_id]['increase'] += 1
            self.consecutive_suggestions[client_id]['decrease'] = 0
            self.consecutive_suggestions[client_id]['stay'] = 0

            # Only increase if we've had multiple consecutive suggestions
            if self.consecutive_suggestions[client_id]['increase'] >= stability_threshold:
                self.consecutive_suggestions[client_id]['increase'] = 0
                return suggested_depth

        elif suggested_depth < current_depth:
            self.consecutive_suggestions[client_id]['decrease'] += 1
            self.consecutive_suggestions[client_id]['increase'] = 0
            self.consecutive_suggestions[client_id]['stay'] = 0

            # Only decrease if we've had multiple consecutive suggestions
            if self.consecutive_suggestions[client_id]['decrease'] >= stability_threshold:
                self.consecutive_suggestions[client_id]['decrease'] = 0
                return suggested_depth

        # Default: don't change
        return current_depth

# --- 5. Model evaluation ---
def evaluate_model(model, test_loader, device):
    """
    Evaluate model accuracy on test set
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# --- 6. Weighted Average of Model Weights ---
def average_weights(weight_list, client_sizes=None):
    """
    Weighted average of model weights based on client dataset sizes
    """
    if client_sizes is None:
        client_sizes = [1] * len(weight_list)

    total_size = sum(client_sizes)
    avg_weights = copy.deepcopy(weight_list[0])

    # Initialize with zeros
    for key in avg_weights:
        avg_weights[key] = torch.zeros_like(avg_weights[key])

    # Weighted sum
    for i, weights in enumerate(weight_list):
        weight = client_sizes[i] / total_size
        for key in avg_weights:
            avg_weights[key] += weights[key] * weight

    return avg_weights

# --- 7. Create federated MNIST dataset ---
def create_federated_mnist(num_clients=5, batch_size=32, iid=True):
    """
    Create federated MNIST dataset with real data
    """
    print("Downloading MNIST dataset...")

    # Download MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Split training data among clients
    total_samples = len(train_dataset)
    samples_per_client = total_samples // num_clients

    clients = []
    client_speeds = []

    print("Creating federated data splits...")

    for client_id in range(num_clients):
        # Define client characteristics
        if client_id == 0:  # Slow client with more data
            speed_factor = 0.5
            client_samples = int(samples_per_client * 1.3)
        elif client_id == 1:  # Fast client with less data
            speed_factor = 2.0
            client_samples = int(samples_per_client * 0.7)
        elif client_id == 2:  # Normal client
            speed_factor = 1.0
            client_samples = samples_per_client
        else:  # Other clients with random characteristics
            speed_factor = np.random.uniform(0.6, 1.8)
            client_samples = int(samples_per_client * np.random.uniform(0.8, 1.2))

        if iid:
            # IID: random samples from all classes
            indices = np.random.choice(range(total_samples),
                                     min(client_samples, total_samples), replace=False)
        else:
            # Non-IID: give each client data from limited classes
            target_classes = np.random.choice(range(10),
                                            size=np.random.randint(2, 6), replace=False)
            indices = []
            targets = np.array(train_dataset.targets)

            for target_class in target_classes:
                class_indices = np.where(targets == target_class)[0]
                class_samples = min(len(class_indices), client_samples // len(target_classes))
                selected = np.random.choice(class_indices, class_samples, replace=False)
                indices.extend(selected)

            # Ensure we don't exceed client_samples
            indices = indices[:client_samples]

        client_dataset = Subset(train_dataset, indices)
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

        clients.append((client_loader, len(client_dataset)))
        client_speeds.append(speed_factor)

        print(f"Client {client_id}: {len(client_dataset)} samples, speed {speed_factor:.1f}x")

    return clients, test_loader, client_speeds

# --- 8. Federated Training Loop ---
def federated_training_loop(
    clients,
    client_speeds,
    rounds,
    device,
    strategy="fedavg",
    max_layers=4,
    fixed_depth=2,
    test_loader=None,
    lr=0.01,
    local_epochs=1
):
    """
    Federated training with heterogeneous clients
    """
    global_model = SimpleNN().to(device)

    # Metrics tracking
    results = {
        'test_accuracies': [],
        'training_times': defaultdict(list),
        'client_depths': defaultdict(list),
        'round_times': [],
        'losses': defaultdict(list)
    }

    # Initialize client depths for DLA - start at depth 2 consistently
    current_depths = {cid: 2 for cid in range(len(clients))}  # Start all at depth 2

    # Initialize DLA stability tracker
    dla_tracker = DLAStabilityTracker(len(clients)) if strategy == "dla" else None

    print(f"\nStarting Federated Training with {strategy.upper()} strategy")
    print(f"Clients: {len(clients)}, Rounds: {rounds}, Max Layers: {max_layers}")
    print(f"Client speeds: {[f'{s:.1f}x' for s in client_speeds]}")
    print("-" * 70)

    for r in range(rounds):
        round_start_time = time.time()
        print(f"\n--- Round {r+1}/{rounds} ---")

        local_weights = []
        local_sizes = []

        for cid, (train_loader, dataset_size) in enumerate(clients):
            # Determine layer depth based on strategy
            if strategy == "fedavg":
                layer_depth = max_layers
            elif strategy == "fedpmt":
                layer_depth = fixed_depth
            elif strategy == "feddrop":
                layer_depth = random.randint(1, max_layers)
            elif strategy == "dla":
                layer_depth = current_depths[cid]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Record depth
            results['client_depths'][cid].append(layer_depth)

            # Local training with client-specific speed
            local_state_dict, elapsed_time, avg_loss = train_locally_heterogeneous(
                global_model, train_loader, layer_depth, device,
                client_speed=client_speeds[cid], epochs=local_epochs, lr=lr
            )

            # Record metrics
            results['training_times'][cid].append(elapsed_time)
            results['losses'][cid].append(avg_loss)

            print(f"Client {cid}: depth={layer_depth}, "
                  f"time={elapsed_time:.2f}s, speed={client_speeds[cid]:.1f}x, loss={avg_loss:.4f}")

            # Update depth for DLA strategy with stability check
            if strategy == "dla":
                suggested_depth = adjust_layers_improved(elapsed_time, layer_depth, cid,
                                                       T_low=3.0, T_high=10.0, max_layers=max_layers)

                # Use stability tracker to prevent oscillation
                new_depth = dla_tracker.should_adapt(cid, layer_depth, suggested_depth, stability_threshold=2)
                current_depths[cid] = new_depth

                if new_depth != layer_depth:
                    print(f"  → Client {cid} depth adapted: {layer_depth} → {new_depth}")
                elif suggested_depth != layer_depth:
                    print(f"  → Client {cid} depth change suggested ({layer_depth}→{suggested_depth}) but delayed for stability")

            # Collect weights and sizes for aggregation
            local_weights.append(local_state_dict)
            local_sizes.append(dataset_size)

        # Aggregate weights
        global_weights = average_weights(local_weights, local_sizes)
        global_model.load_state_dict(global_weights)

        # Evaluate global model
        if test_loader:
            test_acc = evaluate_model(global_model, test_loader, device)
            results['test_accuracies'].append(test_acc)
            print(f"Global Test Accuracy: {test_acc:.2f}%")

        round_time = time.time() - round_start_time
        results['round_times'].append(round_time)
        print(f"Round completed in {round_time:.2f}s")

    return global_model, results

# --- 9. Improved Plotting Functions ---
def plot_comparison_results_improved(all_results, strategies):
    """
    Plot comparison with improved visibility and styling
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Define better colors and styles for visibility
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    linestyles = ['-', '--', '-.', ':']
    linewidths = [3, 2.5, 2.5, 2.5]  # Make FedAvg thicker

    # Test Accuracy with improved styling
    axes[0, 0].set_title('Test Accuracy Over Rounds (MNIST)', fontsize=14, fontweight='bold')
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies']:
            axes[0, 0].plot(range(1, len(results['test_accuracies'])+1),
                           results['test_accuracies'],
                           marker=markers[i],
                           label=strategy.upper(),
                           color=colors[i],
                           linestyle=linestyles[i],
                           linewidth=linewidths[i],
                           markersize=8,
                           markerfacecolor='white',
                           markeredgewidth=2,
                           markeredgecolor=colors[i])

    axes[0, 0].set_xlabel('Round', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].legend(fontsize=11, framealpha=0.9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_facecolor('#fafafa')

    # Training Times with better contrast
    axes[0, 1].set_title('Average Training Time per Round', fontsize=14, fontweight='bold')
    avg_times = []
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        round_avg_times = []
        for round_num in range(len(results['round_times'])):
            round_times = [results['training_times'][cid][round_num]
                          for cid in results['training_times'].keys()]
            round_avg_times.append(np.mean(round_times))
        avg_time = np.mean(round_avg_times)
        avg_times.append(avg_time)

        # Use different colors and add value labels
        bar = axes[0, 1].bar(strategy.upper(), avg_time,
                            color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        # Add value labels on bars
        axes[0, 1].text(i, avg_time + max(avg_times)*0.01, f'{avg_time:.2f}s',
                       ha='center', va='bottom', fontweight='bold')

    axes[0, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 1].set_facecolor('#fafafa')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Layer Depths with improved styling
    axes[1, 0].set_title('Layer Depth Evolution (DLA Strategy)', fontsize=14, fontweight='bold')
    dla_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    for strategy, results in zip(strategies, all_results):
        if strategy == "dla" and results['client_depths']:
            for cid in range(len(results['client_depths'])):
                depths = results['client_depths'][cid]
                axes[1, 0].plot(range(1, len(depths)+1), depths,
                               marker='s', label=f'Client {cid}',
                               color=dla_colors[cid % len(dla_colors)],
                               linewidth=3, markersize=8,
                               markerfacecolor='white',
                               markeredgewidth=2)

    axes[1, 0].set_xlabel('Round', fontsize=12)
    axes[1, 0].set_ylabel('Layer Depth', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_facecolor('#fafafa')

    # Final accuracy with improved styling
    axes[1, 1].set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    final_accs = []
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies']:
            final_acc = results['test_accuracies'][-1]
            final_accs.append(final_acc)
            bar = axes[1, 1].bar(strategy.upper(), final_acc,
                                color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
            # Add value labels
            axes[1, 1].text(i, final_acc + max(final_accs)*0.01, f'{final_acc:.1f}%',
                           ha='center', va='bottom', fontweight='bold')

    axes[1, 1].set_ylabel('Final Accuracy (%)', fontsize=12)
    axes[1, 1].set_facecolor('#fafafa')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# --- 10. Separate accuracy plot for maximum visibility ---
def plot_accuracy_only(all_results, strategies):
    """
    Dedicated accuracy plot with maximum visibility
    """
    plt.figure(figsize=(12, 8))

    # High contrast colors and styles
    colors = ['#000080', '#FF8C00', '#228B22', '#DC143C']  # Navy, DarkOrange, ForestGreen, Crimson
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies']:
            plt.plot(range(1, len(results['test_accuracies'])+1),
                    results['test_accuracies'],
                    marker=markers[i],
                    label=f'{strategy.upper()}',
                    color=colors[i],
                    linestyle=linestyles[i],
                    linewidth=4,  # Thick lines
                    markersize=10,
                    markerfacecolor='white',
                    markeredgewidth=3,
                    markeredgecolor=colors[i])

    plt.title('MNIST Test Accuracy: Federated Learning Strategies', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, framealpha=0.95, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.gca().set_facecolor('#f8f9fa')

    # Add value annotations for final points
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies']:
            final_acc = results['test_accuracies'][-1]
            plt.annotate(f'{final_acc:.1f}%',
                        xy=(len(results['test_accuracies']), final_acc),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7),
                        color='white')

    plt.tight_layout()
    plt.show()

# --- 11. Main Execution Function ---
def run_mnist_federated_experiment(iid=True, num_rounds=15):
    """
    Run comprehensive federated learning experiment with MNIST
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create federated MNIST data
    data_type = "IID" if iid else "Non-IID"
    print(f"Creating federated MNIST dataset ({data_type})...")
    clients, test_loader, client_speeds = create_federated_mnist(
        num_clients=5,
        batch_size=64,
        iid=iid
    )

    strategies = ["fedavg", "fedpmt", "feddrop", "dla"]
    all_results = []

    print(f"\n{'='*70}")
    print(f"FEDERATED LEARNING EXPERIMENT ON MNIST ({data_type})")
    print(f"{'='*70}")

    # Run experiments for each strategy
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Running {strategy.upper()} Strategy on MNIST")
        print(f"{'='*70}")

        set_seed(42)  # Reset seed for fair comparison

        global_model, results = federated_training_loop(
            clients=clients,
            client_speeds=client_speeds,
            rounds=num_rounds,
            device=device,
            strategy=strategy,
            max_layers=4,
            fixed_depth=2,
            test_loader=test_loader,
            lr=0.01,
            local_epochs=3  # More epochs for real data
        )

        all_results.append(results)

        # Print summary
        if results['test_accuracies']:
            final_acc = results['test_accuracies'][-1]
            best_acc = max(results['test_accuracies'])
            print(f"\n{strategy.upper()} Final Results:")
            print(f"  Final Test Accuracy: {final_acc:.2f}%")
            print(f"  Best Test Accuracy: {best_acc:.2f}%")
            print(f"  Average Round Time: {np.mean(results['round_times']):.2f}s")

    # Plot comparison with improved visibility
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE - Generating Comparison Plots")
    print(f"{'='*70}")

    plot_comparison_results_improved(all_results, strategies)
    plot_accuracy_only(all_results, strategies)

    # Print final comparison
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for i, strategy in enumerate(strategies):
        if all_results[i]['test_accuracies']:
            final_acc = all_results[i]['test_accuracies'][-1]
            best_acc = max(all_results[i]['test_accuracies'])
            avg_time = np.mean(all_results[i]['round_times'])
            print(f"{strategy.upper():8} | Final: {final_acc:5.1f}% | Best: {best_acc:5.1f}% | Time: {avg_time:5.2f}s")

    return all_results

# --- 12. Run the experiment ---
if __name__ == "__main__":
    # Run with IID data distribution
    print("Running MNIST Federated Learning Experiment...")
    results_iid = run_mnist_federated_experiment(iid=True, num_rounds=15)

    # Uncomment to also run with Non-IID data
    # print("\n" + "="*70)
    # print("Running Non-IID experiment...")
    # results_noniid = run_mnist_federated_experiment(iid=False, num_rounds=15)



