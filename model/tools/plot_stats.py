import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

STATS_FILE = "/home/lydiachan/cs329s/model/output/cleft/fed_compare/val_stats.npy"

dt = np.load(STATS_FILE)
num_points = np.nonzero(dt[:,1])[0].shape[0]
clients = (dt.shape[1] - 2)//2

epochs = dt[:num_points, 0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))

for i in range(clients):
    val_loss = dt[:num_points, 1+i*2]
    val_nme = dt[:num_points, 2+i*2]
    label = f"client{i}"
    ax1.plot(epochs, val_loss, label=label)
    ax2.plot(epochs, val_nme, label=label)
ax1.plot(epochs, dt[:num_points, -2], label='server')
ax2.plot(epochs, dt[:num_points, -1], label='server')

ax1.set_title("Loss")
ax2.set_title("NME")
ax1.set_xlabel("Epochs")
ax2.set_xlabel("Epochs")
ax1.legend()
ax2.legend()


st.title("Validation Set Statistics")
st.pyplot(fig)
#display_chart = dt[:num_points,:]
#st.line_chart(display_chart)
