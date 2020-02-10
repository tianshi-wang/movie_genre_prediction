import plotly.graph_objects as go


def plot_genre(labels):
    label_to_count = {}
    for label in labels:
        if label in label_to_count:
            label_to_count[label] += 1
        else:
            label_to_count[label] = 1
    x = ['very_bad', 'bad', 'ordinary', 'good', 'very_good']
    y = [label_to_count[key] for key in x]
    fig = go.Figure([go.Bar(x=x, y=y)])
    fig.show()