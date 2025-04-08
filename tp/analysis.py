import gc
from typing import Dict, List, Optional, Tuple, cast

import circuitsvis as cv  # type: ignore
import einops
import matplotlib.pyplot as plt
import numpy as np
import rich
import rich.table
import torch as t
import transformer_lens as tl
from IPython.display import display
from jaxtyping import Float, Int
from reproduce_experiments.plot import imshow  # type: ignore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from causapscal import ORANGE, TURQUOISE, VIOLET
from causapscal.lens import Lens


def filtered_to_str_tokens(
    model: tl.HookedTransformer, raw: List[str]
) -> List[List[str]]:
    return [
        [
            "[NEWLINE]" if x == "\n\n" or x == "\n" else x
            for x in cast(List[str], prompt)
        ]
        for prompt in model.to_str_tokens(raw, prepend_bos=False)
    ]


def get_contrast_directions(
    a_scan: Float[t.Tensor, "n_layers batch d_model"],
    b_scan: Float[t.Tensor, "n_layers batch d_model"],
) -> Float[t.Tensor, "n_layers d_model"]:
    directions = a_scan.mean(dim=1) - b_scan.mean(dim=1)
    return directions / t.linalg.norm(directions, dim=-1, keepdim=True)


def get_pca(
    a_scan: Float[t.Tensor, "n_layers batch d_model"],
    b_scan: Float[t.Tensor, "n_layers batch d_model"],
    n_components: int = 2,
) -> Tuple[
    List[PCA],
    Float[t.Tensor, "n_layers n_samples n_components"],
    Int[t.Tensor, "n_samples"],
]:
    n_layers, n_samples = a_scan.shape[:2]

    activations = t.cat([a_scan, b_scan], dim=1).float().numpy()
    labels = t.cat([t.ones(n_samples), t.zeros(n_samples)]).int()

    pca_list = []
    reduced_activations = []

    for layer in range(n_layers):
        pca = PCA(n_components=n_components)
        x_reduced = pca.fit_transform(activations[layer])

        pca_list.append(pca)
        reduced_activations.append(x_reduced)

    return pca_list, t.Tensor(np.array(reduced_activations)), labels


def plot_pca(
    pca_values: Float[t.Tensor, "n_layers n_samples n_components"],
    pca_labels: Int[t.Tensor, "n_samples"],
    layer: int,
    plot_centers: bool = False,
) -> None:
    a_labels, b_labels = pca_labels.bool(), (1 - pca_labels).bool()
    x_numpy = pca_values[layer].numpy()

    a_center, b_center = (
        x_numpy[a_labels].mean(0),
        x_numpy[b_labels].mean(0),
    )

    fig, ax = plt.subplots()
    ax.scatter(
        x_numpy[a_labels, 0],
        x_numpy[b_labels, 1],
        label="Harmful",
        color=VIOLET,
    )
    ax.scatter(
        x_numpy[a_labels, 0],
        x_numpy[b_labels, 1],
        label="Harmless",
        color=ORANGE,
    )
    if plot_centers:
        ax.scatter(
            [a_center[0], b_center[0]],
            [a_center[1], b_center[1]],
            label="Centers",
            color="grey",
        )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"Layer {layer} activations (PCA)")
    plt.legend()
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plt.show()


def compute_geometric_metrics(
    hf_act: Float[t.Tensor, "n_layers batch_size d_model"],
    hl_act: Float[t.Tensor, "n_layers batch_size d_model"],
    plot: bool = True,
) -> Float[t.Tensor, "3 n_layers"]:
    hf_centers = hf_act.mean(dim=1)
    hl_centers = hf_act.mean(dim=1)

    center_distances = t.norm(hf_centers - hl_centers, dim=1)

    hf_spreads = t.stack([t.pdist(layer.float()).mean() for layer in hf_act])
    hl_spreads = t.stack([t.pdist(layer.float()).mean() for layer in hl_act])

    metrics = t.vstack([center_distances, hf_spreads, hl_spreads])

    if plot:
        table = rich.table.Table(title="Geometric metrics")

        table.add_column("Layer")
        table.add_column("Centroid distance", justify="right", style=TURQUOISE)
        table.add_column("Harmful spread", justify="right", style=VIOLET)
        table.add_column("Harmless spread", justify="right", style=ORANGE)

        for layer in range(hf_act.shape[0]):
            table.add_row(str(layer), *[f"{x:.2f}" for x in metrics[layer]])

        console = rich.console.Console()
        console.print(table)

    return metrics


def compute_logistic_regression(
    hf_act: Float[t.Tensor, "n_layers batch_size d_model"],
    hl_act: Float[t.Tensor, "n_layers batch_size d_model"],
    plot: bool = True,
) -> Tuple[List[LogisticRegression], List[Dict[str, float]]]:
    activations = t.cat([hf_act, hl_act], dim=1).float().numpy()
    labels = t.cat([t.ones(hf_act.shape[1]), t.zeros(hl_act.shape[1])]).int().numpy()

    probes = []
    results = []

    for layer_act in activations:
        probe = LogisticRegression()

        # Compute metrics on test set
        result = cross_validate(
            probe,
            layer_act,
            labels,
            cv=5,
            scoring=["accuracy", "f1", "precision", "recall", "r2"],
            return_train_score=True,
        )
        results.append(result)

        # Learn with full set
        probe.fit(layer_act, labels)
        probes.append(probe)

    results_mean = [{k: v.mean() for k, v in d.items()} for d in results]

    if plot:
        table = rich.table.Table(title="Probes metrics")

        columns = ["Layer"] + list(results_mean[0].keys())[2:]

        for column in columns:
            table.add_column(
                column,
                justify="right",
                style=(
                    ORANGE
                    if "test" in column
                    else (VIOLET if "train" in column else "black")
                ),
            )

        for layer, result in enumerate(results_mean):
            table.add_row(
                *([str(layer)] + [f"{result[column]:.3f}" for column in columns[1:]])
            )

        console = rich.console.Console()
        console.print(table)

    return probes, results_mean


def compute_decision_boundaries(
    probes: List[LogisticRegression],
    plot: bool = True,
) -> Float[t.Tensor, "n_layers d_model"]:
    directions_list = []
    for probe in probes:
        direction = probe.coef_[0]
        direction = direction / np.linalg.norm(direction)
        directions_list.append(t.Tensor(direction))

    directions = t.vstack(directions_list)

    if plot:
        table = rich.table.Table(title="Decision boundary")

        table.add_column("Between layer", justify="right", style="cyan")
        table.add_column("And layer", justify="right", style="dark_violet")
        table.add_column("Similarity", justify="right", style="dark_orange")
        table.add_column("Similarity diff", justify="right", style="dark_orange")

        similarities = t.cosine_similarity(directions[1:], directions[:-1])
        similarities_differences = [
            f"[red]{x:.3f}[/red]" if x > 0 else f"[green]{x:.3f}[/greeen]"
            for x in t.cat([t.zeros(1), similarities[1:] - similarities[:-1]])
        ]

        for layer in range(len(similarities)):
            table.add_row(
                str(layer),
                str(layer + 1),
                f"{similarities:.3f}",
                similarities_differences[layer],
            )

        console = rich.console.Console()
        console.print(table)

    return directions


def plot_attention_patterns(
    pattern: Float[t.Tensor, "n_heads seqQ seqK"],
    tokens: str | List[str],
    layer: Optional[int | str] = None,
) -> None:
    layer = layer if layer is not None else "?"

    display(
        cv.attention.attention_patterns(
            tokens=cast(List[str], tokens),
            attention=pattern,
            attention_head_names=[
                f"L{layer}H{head}" for head in range(pattern.shape[0])
            ],
        )
    )


def get_scores(
    model: tl.HookedTransformer,
    tokens: Float[t.Tensor, "batch seq"],
    patterns: Float[t.Tensor, "n_layers batch n_heads seq seq"],
    x_idx: int = 865,
) -> Float[t.Tensor, "n_layers batch n_heads seq adv_len"]:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    batch = tokens.shape[0]
    seq = tokens.shape[1]

    optim_mask = (1 - (tokens - x_idx * t.ones_like(tokens)).bool().int()).bool()
    optim_mask_expanded = (
        optim_mask.unsqueeze(0)
        .unsqueeze(2)
        .unsqueeze(-2)
        .repeat(n_layers, 1, n_heads, seq, 1)
    )

    return patterns[optim_mask_expanded].view(n_layers, batch, n_heads, seq, -1).sum(-2)


def get_attn_contribution(
    model: tl.HookedTransformer, cache: tl.ActivationCache
) -> Float[t.Tensor, "n_layers batch n_heads d_model"]:
    n_layers = model.cfg.n_layers

    attn_z = t.vstack(
        [cache[f"blocks.{layer}.attn.hook_z"].unsqueeze(0) for layer in range(n_layers)]
    )[:, :, -1, :, :]
    attn_z_cuda = attn_z.to("cuda")
    attn_result_cuda = einops.einsum(
        attn_z_cuda,
        model.W_O,
        "n_layers batch n_heads d_head, n_layers n_heads d_head d_model -> n_layers batch n_heads d_model",
    )
    return attn_result_cuda


def ln_to_pattern(
    Wq: Float[t.Tensor, "q_heads d_model d_heads"],
    activations: Float[t.Tensor, "batch seq_len d_model"],
    k_value: Float[t.Tensor, "batch seq_len k_heads d_heads"],
) -> Float[t.Tensor, "batch q_heads seqQ seqK"]:
    from einops import einsum, rearrange

    q_value = einsum(
        Wq,
        activations,
        "q_heads d_model d_heads, batch seq d_model -> batch seq q_heads d_heads",
    )
    query = rearrange(q_value, "batch seq q_heads d_heads -> batch q_heads seq d_heads")
    key = rearrange(k_value, "batch seq k_heads d_heads -> batch k_heads seq d_heads")

    batch_size, q_heads, seq_len, d_heads = query.shape
    batch_size, k_heads, seq_len, d_heads = key.shape

    query = query / query.size(-1) ** 0.5

    num_head_groups = q_heads // k_heads
    query = rearrange(
        query,
        "batch (k_heads group) seq d_heads -> batch group k_heads seq d_heads",
        group=num_head_groups,
    )
    similarity = einsum(
        query,
        key,
        "batch group k_heads seqQ d_heads, batch k_heads seqK d_heads -> batch group k_heads seqQ seqK",
    )

    mask = t.ones(
        (batch_size, num_head_groups, k_heads, seq_len, seq_len),
        device=query.device,
        dtype=t.bool,
    ).tril_()
    similarity.masked_fill_(~mask, t.finfo(similarity.dtype).min)
    attention = t.nn.functional.softmax(similarity, dim=-1)
    attn_weights = rearrange(
        attention, "batch group k_heads seqQ seqK -> batch seqQ seqK (k_heads group)"
    )
    pattern = rearrange(
        attn_weights, "batch seqQ seqK q_heads -> batch q_heads seqQ seqK"
    )

    return pattern


def compute_solution_space(
    Q: Float[t.Tensor, "d_head d_model"],
    y: Float[t.Tensor, "d_head"],
    tol: float = 1e-10,
) -> Tuple[Float[t.Tensor, "d_model"], Float[t.Tensor, "d_model d_null"]]:
    from torch.linalg import pinv, svd

    _, S, Vh = svd(Q, full_matrices=True)
    rank = t.sum(tol < S)
    x_particular = pinv(Q) @ y

    nullspace_basis = Vh[rank:].T
    return x_particular.reshape(-1), nullspace_basis


def generate_solution(
    x_particular: Float[t.Tensor, "d_model"],
    nullspace_basis: Float[t.Tensor, "d_model d_null"],
    coefficients: Optional[Float[t.Tensor, "d_null"]] = None,
) -> Float[t.Tensor, "d_model"]:
    if coefficients is None:
        coefficients = t.randn(nullspace_basis.shape[1], device=x_particular.device)

    return x_particular + nullspace_basis @ coefficients


def single_target_token(
    lens: Lens,
    pattern: Float[t.Tensor, "batch n_heads seqQ seqK"],
    sentences: List[str],
    target: str,
    source: int = -1,
):
    tokens_list = [
        ["[NEWLINE]" if x == "\n\n" else x for x in lens.model.to_str_tokens(s)]
        for s in sentences
    ]
    target_idx = (
        t.Tensor([t.index(target) for t in tokens_list]).long().to(pattern.device)
    )

    n_heads = lens.model.cfg.n_heads
    batch_size = pattern.shape[0]
    seq_len = pattern.shape[-1]
    n_layers = lens.model.cfg.n_layers

    attns = []
    for _ in range(n_layers):
        attn_last_tok = pattern[:, :, source, :]
        attn = t.gather(
            attn_last_tok,
            -1,
            target_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand((batch_size, n_heads, seq_len)),
        )[:, :, 0]
        attns.append(attn.mean(0))

    return t.stack(attns)


def compute_head_contribution(
    lens: Lens, cache: tl.ActivationCache, token: int, plot: bool = False
):
    n_layers = lens.model.cfg.n_layers

    attn_z = t.vstack(
        [cache[f"blocks.{layer}.attn.hook_z"].unsqueeze(0) for layer in range(n_layers)]
    )[:, :, -1, :, :].mean(1)
    attn_z_cuda = attn_z.to("cuda")
    attn_result_cuda = einops.einsum(
        attn_z_cuda, lens.model.W_O, "... d_head, ... d_head d_model -> ... d_model"
    )
    attn_contrib_cuda = einops.einsum(
        lens.model.W_U[:, token], attn_result_cuda, "d_model, ... d_model -> ..."
    )
    attn_contrib = attn_contrib_cuda.to("cpu")

    del attn_z_cuda, attn_result_cuda, attn_contrib_cuda
    gc.collect()
    t.cuda.empty_cache()

    if plot:
        imshow(attn_contrib)
    else:
        return attn_contrib


def logit_lens(
    lens: Lens,
    activations: Float[t.Tensor, "d_model"],
    k: int = 5,
    title: str = "Logit Lens",
):
    p_lens = t.topk(activations.to("cuda") @ lens.model.W_U, k=k)
    p_lens_values = p_lens.values.cpu().detach().numpy()
    p_lens_indices = p_lens.indices.unsqueeze(1).cpu()
    p_lens_str = lens.model.to_string(p_lens_indices)

    n_lens = t.topk(-activations.to("cuda") @ lens.model.W_U, k=k)
    n_lens_values = n_lens.values.cpu().detach().numpy()
    n_lens_indices = n_lens.indices.unsqueeze(1).cpu()
    n_lens_str = np.array(lens.model.to_string(n_lens_indices))

    table = rich.table.Table(title=title)

    for str_tok in p_lens_str:
        table.add_column(str_tok, justify="left", style=ORANGE)
    for str_tok in n_lens_str[::-1]:
        table.add_column(str_tok, justify="right", style=VIOLET)

    table.add_row(
        *[f"{x:.3f}" for x in np.concatenate([p_lens_values, n_lens_values[::-1]])]
    )
    table.add_row(
        *[
            f"{x:.3f}"
            for x in np.concatenate(
                [np.cumsum(p_lens_values), np.cumsum(n_lens_values)[::-1]]
            )
        ]
    )

    console = rich.console.Console()
    console.print(table)
