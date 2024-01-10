import torch
from adaml import AdamL

def test_adaml_initialization():
    """
    Test the initialization of the AdamL optimizer.
    """
    model = torch.nn.Linear(10, 2)  # A simple model
    optimizer = AdamL(model.parameters())
    assert optimizer is not None, "Failed to initialize AdamL optimizer"

def test_adaml_updates_parameters():
    """
    Test if AdamL optimizer updates model parameters.
    """
    model = torch.nn.Linear(10, 2)
    optimizer = AdamL(model.parameters())
    initial_params = [param.clone() for param in model.parameters()]

    # Define a simple loss and perform an optimization step
    output = model(torch.randn(1, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    for initial, updated in zip(initial_params, model.parameters()):
        assert not torch.equal(initial, updated), "Parameter was not updated"
