from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from conf import log_path

acc = EventAccumulator( log_path )
acc.Reload()

# print(acc.Tags())

loss = pd.DataFrame(acc.Scalars("loss"))
tr_loss = pd.DataFrame(acc.Scalars("Training Loss/Epoch"))

print(tr_loss.value)