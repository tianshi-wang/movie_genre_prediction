import plotly.graph_objects as go


def plot_learning_curve(train_sizes, train_scores, valid_scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores,
                             mode='lines',
                             name='training_scores'))
    fig.add_trace(go.Scatter(x=train_sizes, y=valid_scores,
                             mode='lines',
                             name='cv_scores'))
    fig.update_layout(title='Learning curve',
                      xaxis_title='Number of examples',
                      yaxis_title='Accuracy')
    fig.show()