import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import plotly.graph_objects as go

from evaluation import  EvalMetrics as evm, compute_top_k_accuracy, compute_map_k, 
                        compute_mar_k, compute_F1_k

# Visualize results for text to image queries
def visualize_t2i_results(query, matches):
    print("Top matches for query: \"" + query + "\"")
    if "image path" in matches[0]:
        plt.figure(figsize=(18, 18))
    for i in range(len(matches)):
        if "image path" in matches[i]:
            path = matches[i]["image path"].numpy().decode('UTF-8')
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(mpimg.imread(path))
            plt.axis("off")
        if "caption" in matches[i]:
            caption = matches[i]["caption"].numpy().decode('UTF-8')
            print(f"{i}) {caption}")
        
# Standard visualization for a multi-purpose plotly graph
def visualize_multigraph(functions, titlexyf=(None, None, None), legend=True):
    fig = go.Figure()
    for function in functions:
        x = function['x']
        y = function['y']
        label = function['label'] if 'label' in function else ""
        color = function['color'] if 'color' in function else None
        linestyle = function['style'] if 'style' in function else "solid"
        marker = go.scatter.Marker(symbol=function['marker'], size=10) if 'marker' in function else None
        opacity = function['opacity'] if 'opacity' in function else 1
        k = len(x)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            line=go.scatter.Line(color=color, dash=linestyle),
            opacity=opacity,
            marker=marker,
            mode="lines+markers+text" if marker else "lines+text",
            name=label,
        ))
    fig.update_xaxes(
        title=titlexyf[0],
        ticks="outside", ticklen=8, minor=dict(dtick=0.5, ticklen=6, tickcolor="black", showgrid=True), ticklabelstep=1, dtick=1, 
        range=(1,k), 
    )
    fig.update_yaxes(
        title=titlexyf[1],
        ticks="outside", ticklen=8, minor=dict(dtick=0.01, ticklen=6, tickcolor="black", showgrid=True), ticklabelstep=1, dtick=0.1,
    )
    fig.update_layout(
        title=titlexyf[2],
        width=900, height=600,
        margin=dict(l=50, r=50, b=20, t=40, pad=4),
        paper_bgcolor="LightSteelBlue",
    )
    fig.show()


# Compute baselines for retrieval
# Assumption of sampling with repetitions, results get more inaccurate as k/l -> inf
def retrieval_baselines(dataset_reference, total_relevant, k, metrics=[]):
    l = len(dataset_reference)
    metrics_out = {}
    for metric in metrics:
        if metric["id"] == evm.METRIC_ACCURACY:
            metrics_out[evm.METRIC_ACCURACY] = sum([ 1 - pow((l - n_el) / l, k) for n_el in total_relevant]) / l
        elif metric["id"] == evm.METRIC_MAP:
            metrics_out[evm.METRIC_MAP] = sum([ n_el / l for n_el in total_relevant]) / l
        elif metric["id"] == evm.METRIC_MAR:
            metrics_out[evm.METRIC_MAR] = sum([ k / l for n_el in total_relevant]) / l
        elif metric["id"] == evm.METRIC_F1:
            metrics_out[evm.METRIC_F1] = compute_F1_k(metrics_out[evm.METRIC_MAP], metrics_out[evm.METRIC_MAR])
    return metrics_out
    
# Computation of a retrieval report containing metrics
def retrieval_report(
    results, reference, relevant,   # Task results, dataset reference and relevant hits at k for task
    tot_relevant=None,              # Rotal number of relevant elements for each dataset element
    k=None,                         # k for metrics computation (should be less or equal than k of retrieval)
    baselines=True,                 # Calculate baselines alongside metrics
    metrics=[],                     # Metrics to take into consideration
    output=True,                    # Print outputs to stdout
    title="Retrieval Report",       # Title of the report
    decimal_precision=4,            # Decimal precision of values
):
    if not k:
        k = len(results[0])
    metrics_out = {}
    
    for metric in metrics:
        if metric["id"] == evm.METRIC_ACCURACY:
            metrics_out[evm.METRIC_ACCURACY] = compute_top_k_accuracy(results, reference, relevant)
        elif metric["id"] == evm.METRIC_MAP:
            metrics_out[evm.METRIC_MAP] = compute_map_k(results, reference, relevant, k=k)
        elif metric["id"] == evm.METRIC_MAR:
            metrics_out[evm.METRIC_MAR] = compute_mar_k(results, reference, relevant, tot_relevant)
        elif metric["id"] == evm.METRIC_F1:
            metrics_out[evm.METRIC_F1] = compute_F1_k(metrics_out[evm.METRIC_MAP], metrics_out[evm.METRIC_MAR])
            
    if baselines:
            baselines = retrieval_baselines(reference, tot_relevant, k, metrics=metrics)
            
    if output:
        print(f"\n ### {title} ###")
        for metric in metrics:
            string = f"{metric['name']:<30}: {round(metrics_out[metric['id']] * 100, decimal_precision):10}%"
            if baselines:
                string += f"{'   Baseline':<8}: {round(baselines[metric['id']] * 100, decimal_precision):10}%"
            print(string)
    
    if baselines:
        return metrics_out, baselines
    return metrics_out
        
# Computation of a retrieval report in graph form containing metrics 
def retrieval_graph_report(
    results, reference,                 # Task results and dataset reference
    tot_relevant=None,                  # Total number of relevant elements for each dataset element
    k_range=(1, 10),                    # k range for metrics computation (maximum value shoul not be greater than k of retrieval)
    baselines=True,                     # Calculate baselines alongside metrics
    metrics=[],                         # Metrics to take into consideration
    titlexyf=(None, None, None),        # Tuple containing: (title of x axis, title of y axis, figure title)
    reference_preprocess=lambda x: x,   # Function to preprocess data contained in the reference dataset
    relevance=lambda m, o: m == o,      # Function to compare elements
    functions=None,                     # Plot pre-existing function data
):
    if not functions:
        functions = {metric["id"]: {"x": [], "y": [], "label": metric["id"], "color": metric["color"], "marker": "0", "opacity": 0.8} for metric in metrics}
        if baselines:
            functions |= {metric["id"] + "_base": {"x": [], "y": [], "label": metric["id"] + " Baseline", "color": metric["color"], "style": "dash", "opacity": 0.5} for metric in metrics}
        for k in range(k_range[0], k_range[1] + 1):
            relevant = compute_relevant_at_k(results, reference, k=k, reference_preprocess=reference_preprocess, relevance=relevance)
            report = retrieval_report(results, reference, relevant, tot_relevant, k=k, baselines=baselines, metrics=metrics, output=False)
            metrics_out = report[0] if baselines else report
            baselines = report[1] if baselines else None
            for metric in metrics_out:
                functions[metric]["x"].append(k)
                functions[metric]["y"].append(metrics_out[metric])
                if baselines:
                    functions[metric + "_base"]["x"].append(k)
                    functions[metric + "_base"]["y"].append(baselines[metric])
    visualize_multigraph(functions.values(), titlexyf)
    return functions

# Computation of a retrieval report in graph form containing metrics, comparing multiple models on the same dataset
def retrieval_graph_compare(
    multi_results, reference,           # Per-model task results and dataset reference
    model_ids,                          # List of ordered model ids and labels in the form {"id": id, "label": label} 
    tot_relevant=None,                  # Total number of relevant elements for each dataset element
    k_range=(1, 10),                    # k range for metrics computation (maximum value shoul not be greater than k of retrieval)
    metrics=[],                         # Metrics to take into consideration
    titlexyf=(None, None, None),        # Tuple containing: (title of x axis, title of y axis, figure title)
    reference_preprocess=lambda x: x,   # Function to preprocess data contained in the reference dataset
    relevance=lambda m, o: m == o,      # Function to compare elements
    functions=None,                     # Plot pre-existing function data
):
    if not functions:
        # Add random markers
        markers = ["0", "1", "2", "3", "17", "26"]
        if len(markers) < len(model_ids):
            print("Too many models!")
            return None
        markers = random.sample(markers, len(model_ids))
        model_ids = [model | {"marker": marker} for model, marker in zip(model_ids, markers)]
        # Generate function models
        functions = {
            metric["id"] + model["id"]: 
            {"x": [], "y": [], "label": model["label"] + " " + metric["id"], "color": metric["color"], "marker": model["marker"], "opacity": 0.8} 
            for model in model_ids for metric in metrics
        }
        # Fill functions
        for model, results in zip(model_ids, multi_results):
            for k in range(k_range[0], k_range[1] + 1):
                relevant = compute_relevant_at_k(results, reference, k=k, reference_preprocess=reference_preprocess, relevance=relevance)
                metrics_out = retrieval_report(results, reference, relevant, tot_relevant, k=k, baselines=False, metrics=metrics, output=False)
                for metric in metrics_out:
                    functions[metric + model["id"]]["x"].append(k)
                    functions[metric + model["id"]]["y"].append(metrics_out[metric])
    visualize_multigraph(functions.values(), titlexyf)
    return functions
    
# Manually compute some text to image queries
def manual_t2i_queries(queries, text_encoder, image_embeddings, dataset_reference, k=10, normalize=True):
    results = find_t2i_matches(queries, clip_text_encoder, test_image_embeddings, k=k, normalize=normalize)
    results = index_to_reference(results, test_dataset_reference)
    for query, matches in zip(queries, results):
        visualize_t2i_results(query, matches)