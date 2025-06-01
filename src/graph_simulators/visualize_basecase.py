import os
import torch
import matplotlib.pyplot as plt

# --- Local imports (adjust to your codebase) ---
from src.graph_simulators.config import parse_args
from src.graph_simulators.dataloaders import SequenceGraphSettingsDataLoaders
from src.graph_models.context_models.scale_graph_networks import ScaleAwareLogRatioConditionalGraphNetwork
from src.graph_simulators.utils import set_random_seed
from your_script_above import unscale_predictions  # Make sure you import or redefine unscale_predictions

def visualize_basecase():
    # 1) Parse arguments
    args = parse_args()
    args.mode = 'evaluate'
    device = torch.device('cpu') if args.cpu_only else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seed(args.random_seed)

    # 2) Create data loaders
    graph_data_dir = os.path.join(args.base_data_dir, args.dataset, f"{args.data_keyword}_graphs")
    data_loaders = SequenceGraphSettingsDataLoaders(
        graph_data_dir=graph_data_dir,
        initial_step=args.initial_step,
        final_step=args.final_step,
        max_prediction_horizon=args.horizon,
        include_settings=args.include_settings,
        identical_settings=args.identical_settings,
        use_edge_attr=args.use_edge_attr,
        subsample_size=args.subsample_size,
        include_position_index=args.include_position_index,
        include_scaling_factors=args.include_scaling_factors,
        scaling_factors_file=args.scaling_factors_file,
        batch_size=args.batch_size,
        n_train=args.ntrain,
        n_val=args.nval,
        n_test=args.ntest,
    )
    
    train_loader = data_loaders.get_train_loader()
    
    # 3) Retrieve the first batch from the train set
    #    Depending on args.include_settings, the structure of the batch differs.
    batch = next(iter(train_loader))
    if args.include_settings:
        batch_initial_graph, batch_target_graph, seq_lengths, settings_tensor = batch
    else:
        batch_initial_graph, batch_target_graph, seq_lengths = batch

    # For simplicity, let's just consider the first sample in the batch.
    #   batch_*_graph are PyG Batch objects if you're using PyTorch Geometric
    #   or plain custom objects if you're using your own data structure.
    #   They can contain multiple graphs concatenated. We'll just pick the
    #   first example's node indices for a direct illustration.

    # 4) Setup model and load checkpoint
    #    (We assume your checkpoint includes the model_state_dict, etc.)
    model = ScaleAwareLogRatioConditionalGraphNetwork(
        node_in_dim=args.node_in_dim,        # Modify these to match your model signature
        edge_in_dim=args.edge_in_dim,
        cond_in_dim=args.cond_in_dim if args.include_settings else 0,
        scale_dim=args.scale_dim,
        node_out_dim=args.node_out_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        log_ratio_dim=args.scale_dim
    )
    model.to(device)
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.eval()

    # 5) Forward pass to predict step1 from step0
    #    We'll select the entire batch (just for the first example you might isolate node indices),
    #    but here let's directly apply the model to the whole batch_initial_graph:
    batch_initial_graph = batch_initial_graph.to(device)
    with torch.no_grad():
        if args.include_settings:
            pred_x_scaled, pred_log_ratios = model(
                x=batch_initial_graph.x,
                edge_index=batch_initial_graph.edge_index,
                edge_attr=batch_initial_graph.edge_attr,
                conditions=settings_tensor.to(device),
                scale=batch_initial_graph.scale,
                batch=batch_initial_graph.batch,
            )
        else:
            pred_x_scaled, pred_log_ratios = model(
                x=batch_initial_graph.x,
                edge_index=batch_initial_graph.edge_index,
                edge_attr=batch_initial_graph.edge_attr,
                conditions=None,
                scale=batch_initial_graph.scale,
                batch=batch_initial_graph.batch,
            )

    # 6) Unscale the predictions
    pred_x_unscaled = unscale_predictions(
        pred_x_scaled,
        pred_log_ratios,
        batch_initial_graph.scale,
        batch_initial_graph.batch
    )

    # 7) Get original step0 (input) and original step1 (target)
    #    We'll unscale those as well if they are stored in scaled form.
    #    Typically, step0 is already in scaled form in batch_initial_graph.x
    #    Let's unscale both step0 and step1 for fair comparison.
    step0_x_unscaled = unscale_predictions(
        batch_initial_graph.x,
        torch.zeros_like(pred_log_ratios),  # log_ratios = 0 => scale remains initial
        batch_initial_graph.scale,
        batch_initial_graph.batch
    )

    batch_target_graph = batch_target_graph.to(device)
    step1_x_unscaled = unscale_predictions(
        batch_target_graph.x,
        torch.zeros_like(pred_log_ratios),  # similarly, keep scale from step1's graph
        batch_target_graph.scale,
        batch_target_graph.batch
    )

    # 8) Convert tensors to CPU numpy for visualization
    step0_x_unscaled = step0_x_unscaled.cpu().numpy()
    step1_x_unscaled = step1_x_unscaled.cpu().numpy()
    pred_x_unscaled = pred_x_unscaled.cpu().numpy()

    # 9) Visualization: Plot (x,y) for step0, predicted step1, original step1
    #    Adjust indexing if your data is 3D or has different feature ordering.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(step0_x_unscaled[:, 0],
               step0_x_unscaled[:, 1],
               c='blue', label='Original Step 0', alpha=0.7)
    ax.scatter(pred_x_unscaled[:, 0],
               pred_x_unscaled[:, 1],
               c='red', marker='x', label='Predicted Step 1')
    ax.scatter(step1_x_unscaled[:, 0],
               step1_x_unscaled[:, 1],
               c='green', label='Original Step 1', alpha=0.7)

    ax.set_title("Base Case Visualization: Step0 → Predicted Step1 vs. Original Step1")
    ax.legend()
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.show()

if __name__ == "__main__":
    visualize_basecase()
