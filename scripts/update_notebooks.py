import json
from pathlib import Path

# Paths
nb4_path = Path("notebooks/04_dani_dl_dummy_baseline.ipynb")
nb5_path = Path("notebooks/05_dani_dl_cyclical.ipynb")

plot_func_code = '''def plot_model_evaluation(
    history,
    y_true,
    y_pred,
    model_name,
    target_name,
    unit="gCO₂eq/kWh",
    plot_hours=500
):
    mae = np.mean(np.abs(y_true - y_pred))
    plot_hours = min(plot_hours, len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Loss curve
    axes[0].plot(
        history.history['loss'],
        label='Training Loss',
        color='#3498db'
    )
    axes[0].plot(
        history.history['val_loss'],
        label='Validation Loss',
        color='#e74c3c'
    )
    axes[0].set_title(
        f'Training vs Validation Loss — {model_name}',
        fontsize=14,
        fontweight='bold'
    )
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')

    # Actual vs Predicted
    axes[1].plot(
        y_true[:plot_hours],
        label=f'Actual {target_name}',
        color='#2c3e50',
        linewidth=1.5
    )
    axes[1].plot(
        y_pred[:plot_hours],
        label=f'Predicted {target_name} (MAE={mae:.2f})',
        color='#e74c3c',
        linewidth=1,
        alpha=0.8
    )
    axes[1].set_title(
        f'Actual vs Predicted — {model_name} ({plot_hours} jam)',
        fontsize=14,
        fontweight='bold'
    )
    axes[1].set_xlabel('Timestep (jam)')
    axes[1].set_ylabel(unit)
    axes[1].legend()
    axes[1].grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()
'''

# ============================================================
# UPDATE NOTEBOOK 04
# ============================================================
with open(nb4_path, 'r', encoding='utf-8') as f:
    nb4 = json.load(f)

# 1. Update build_lstm / build_bilstm cell (or add helper plot func in cell 18)
nb4['cells'][18]['source'].append('\n\n' + plot_func_code)

# 2. In Cell 53 (BiLSTM Multi evaluation), rename variable names to avoid overwriting univariate RE
cell53_src = '''from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# =========================
# CI
# =========================
y_true_ci = y_test_multi_original[:, 0]
y_pred_bilstm_multi_ci = y_pred_bilstm_multi[:, 0]

rmse_ci = np.sqrt(mean_squared_error(y_true_ci, y_pred_bilstm_multi_ci))
mae_ci = mean_absolute_error(y_true_ci, y_pred_bilstm_multi_ci)
mse_ci = mean_squared_error(y_true_ci, y_pred_bilstm_multi_ci)
mape_ci = np.mean(
    np.abs((y_true_ci - y_pred_bilstm_multi_ci) / y_true_ci)
) * 100

print("=== BASELINE BiLSTM — MULTIVARIATE CI RESULTS ===")
print(f"RMSE: {rmse_ci:.4f}")
print(f"MAE: {mae_ci:.4f}")
print(f"MAPE (%): {mape_ci:.4f}")
print(f"MSE: {mse_ci:.4f}")


# =========================
# RE
# =========================
y_true_re = y_test_multi_original[:, 1]
y_pred_bilstm_multi_re = y_pred_bilstm_multi[:, 1]

rmse_re_bilstm_multi = np.sqrt(mean_squared_error(y_true_re, y_pred_bilstm_multi_re))
mae_re_bilstm_multi = mean_absolute_error(y_true_re, y_pred_bilstm_multi_re)
mse_re_bilstm_multi = mean_squared_error(y_true_re, y_pred_bilstm_multi_re)
mape_re_bilstm_multi = np.mean(
    np.abs((y_true_re - y_pred_bilstm_multi_re) / y_true_re)
) * 100

print("\\n=== BASELINE BiLSTM — MULTIVARIATE RE RESULTS ===")
print(f"RMSE: {rmse_re_bilstm_multi:.4f}")
print(f"MAE: {mae_re_bilstm_multi:.4f}")
print(f"MAPE (%): {mape_re_bilstm_multi:.4f}")
print(f"MSE: {mse_re_bilstm_multi:.4f}")
'''
nb4['cells'][53]['source'] = [cell53_src]

# 3. Add visualization cells & Save cells to NB4
vis_cell_nb4 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# MODEL EVALUATION VISUALIZATIONS (Loss & Actual vs Predicted)\n",
        "# ============================================================\n\n",
        "# 1. Univariate LSTM — CI\n",
        "plot_model_evaluation(\n",
        "    history=history_lstm_ci,\n",
        "    y_true=y_test_ci_actual,\n",
        "    y_pred=y_pred_ci,\n",
        "    model_name=\"Univariate LSTM — CI\",\n",
        "    target_name=\"CI\",\n",
        "    unit=\"gCO₂eq/kWh\"\n",
        ")\n\n",
        "# 2. Univariate LSTM — RE\n",
        "plot_model_evaluation(\n",
        "    history=history_lstm_re,\n",
        "    y_true=y_test_re_original,\n",
        "    y_pred=y_pred_re,\n",
        "    model_name=\"Univariate LSTM — RE\",\n",
        "    target_name=\"RE\",\n",
        "    unit=\"%\"\n",
        ")\n\n",
        "# 3. Multivariate LSTM — CI\n",
        "plot_model_evaluation(\n",
        "    history=history_lstm_multi,\n",
        "    y_true=y_test_multi_original[:, 0],\n",
        "    y_pred=y_pred_multi[:, 0],\n",
        "    model_name=\"Multivariate LSTM — CI\",\n",
        "    target_name=\"CI\",\n",
        "    unit=\"gCO₂eq/kWh\"\n",
        ")\n\n",
        "# 4. Multivariate LSTM — RE\n",
        "plot_model_evaluation(\n",
        "    history=history_lstm_multi,\n",
        "    y_true=y_test_multi_original[:, 1],\n",
        "    y_pred=y_pred_multi[:, 1],\n",
        "    model_name=\"Multivariate LSTM — RE\",\n",
        "    target_name=\"RE\",\n",
        "    unit=\"%\"\n",
        ")\n\n",
        "# 5. BiLSTM — CI\n",
        "plot_model_evaluation(\n",
        "    history=history_bilstm_ci,\n",
        "    y_true=y_test_ci,\n",
        "    y_pred=y_pred_bilstm_ci,\n",
        "    model_name=\"BiLSTM — CI\",\n",
        "    target_name=\"CI\",\n",
        "    unit=\"gCO₂eq/kWh\"\n",
        ")\n\n",
        "# 6. BiLSTM — RE\n",
        "plot_model_evaluation(\n",
        "    history=history_bilstm_re,\n",
        "    y_true=y_test_re_original,\n",
        "    y_pred=y_pred_bilstm_re,\n",
        "    model_name=\"BiLSTM — RE\",\n",
        "    target_name=\"RE\",\n",
        "    unit=\"%\"\n",
        ")\n\n",
        "# 7. Multivariate BiLSTM — CI\n",
        "plot_model_evaluation(\n",
        "    history=history_bilstm_multi,\n",
        "    y_true=y_test_multi_original[:, 0],\n",
        "    y_pred=y_pred_bilstm_multi[:, 0],\n",
        "    model_name=\"Multivariate BiLSTM — CI\",\n",
        "    target_name=\"CI\",\n",
        "    unit=\"gCO₂eq/kWh\"\n",
        ")\n\n",
        "# 8. Multivariate BiLSTM — RE\n",
        "plot_model_evaluation(\n",
        "    history=history_bilstm_multi,\n",
        "    y_true=y_test_multi_original[:, 1],\n",
        "    y_pred=y_pred_bilstm_multi[:, 1],\n",
        "    model_name=\"Multivariate BiLSTM — RE\",\n",
        "    target_name=\"RE\",\n",
        "    unit=\"%\"\n",
        ")\n"
    ]
}

save_metrics_cell_nb4 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Save evaluation results\n",
        "results_dir = (\n",
        "    project_root\n",
        "    / \"results\"\n",
        "    / \"deep_learning\"\n",
        ")\n\n",
        "results_dir.mkdir(\n",
        "    parents=True,\n",
        "    exist_ok=True\n",
        ")\n\n",
        "results = [\n",
        "    {\"Model\": \"Uni-LSTM CI\", \"Target\": \"CI\", \"RMSE\": score_ci_lstm[\"RMSE\"], \"MAE\": score_ci_lstm[\"MAE\"], \"MAPE (%)\": score_ci_lstm[\"MAPE (%)\"], \"MSE\": score_ci_lstm[\"MSE\"]},\n",
        "    {\"Model\": \"Uni-LSTM RE\", \"Target\": \"RE\", \"RMSE\": rmse_re, \"MAE\": mae_re, \"MAPE (%)\": mape_re, \"MSE\": mse_re},\n",
        "    {\"Model\": \"Multi-LSTM\", \"Target\": \"CI\", \"RMSE\": metrics_multi_ci[\"RMSE\"], \"MAE\": metrics_multi_ci[\"MAE\"], \"MAPE (%)\": metrics_multi_ci[\"MAPE (%)\"], \"MSE\": metrics_multi_ci[\"MSE\"]},\n",
        "    {\"Model\": \"Multi-LSTM\", \"Target\": \"RE\", \"RMSE\": metrics_multi_re[\"RMSE\"], \"MAE\": metrics_multi_re[\"MAE\"], \"MAPE (%)\": metrics_multi_re[\"MAPE (%)\"], \"MSE\": metrics_multi_re[\"MSE\"]},\n",
        "    {\"Model\": \"BiLSTM CI\", \"Target\": \"CI\", \"RMSE\": rmse_bilstm_ci, \"MAE\": mae_bilstm_ci, \"MAPE (%)\": mape_bilstm_ci, \"MSE\": mse_bilstm_ci},\n",
        "    {\"Model\": \"BiLSTM RE\", \"Target\": \"RE\", \"RMSE\": rmse_bilstm_re, \"MAE\": mae_bilstm_re, \"MAPE (%)\": mape_bilstm_re, \"MSE\": mse_bilstm_re},\n",
        "    {\"Model\": \"BiLSTM Multi\", \"Target\": \"CI\", \"RMSE\": rmse_ci, \"MAE\": mae_ci, \"MAPE (%)\": mape_ci, \"MSE\": mse_ci},\n",
        "    {\"Model\": \"BiLSTM Multi\", \"Target\": \"RE\", \"RMSE\": rmse_re_bilstm_multi, \"MAE\": mae_re_bilstm_multi, \"MAPE (%)\": mape_re_bilstm_multi, \"MSE\": mse_re_bilstm_multi},\n",
        "]\n\n",
        "results_df = pd.DataFrame(results)\n\n",
        "results_path = results_dir / \"04_dani_dl_dummy_baseline_metrics.csv\"\n",
        "results_df.to_csv(results_path, index=False)\n\n",
        "print(\"Results saved to:\")\n",
        "print(results_path)\n",
        "display(results_df)\n"
    ]
}

save_models_cell_nb4 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Save trained models\n",
        "models_dir = (\n",
        "    project_root\n",
        "    / \"models\"\n",
        "    / \"deep_learning\"\n",
        ")\n\n",
        "models_dir.mkdir(\n",
        "    parents=True,\n",
        "    exist_ok=True\n",
        ")\n\n",
        "model_lstm_ci.save(models_dir / \"04_lstm_ci_dummy.keras\")\n",
        "model_lstm_re.save(models_dir / \"04_lstm_re_dummy.keras\")\n",
        "model_lstm_multi.save(models_dir / \"04_lstm_multi_dummy.keras\")\n",
        "model_bilstm_ci.save(models_dir / \"04_bilstm_ci_dummy.keras\")\n",
        "model_bilstm_re.save(models_dir / \"04_bilstm_re_dummy.keras\")\n",
        "model_bilstm_multi.save(models_dir / \"04_bilstm_multi_dummy.keras\")\n\n",
        "print(\"All six trained models saved.\")\n"
    ]
}

nb4['cells'].extend([vis_cell_nb4, save_metrics_cell_nb4, save_models_cell_nb4])

with open(nb4_path, 'w', encoding='utf-8') as f:
    json.dump(nb4, f, indent=1)

print("Updated 04_dani_dl_dummy_baseline.ipynb successfully.")


# ============================================================
# UPDATE NOTEBOOK 05
# ============================================================
with open(nb5_path, 'r', encoding='utf-8') as f:
    nb5 = json.load(f)

# Cell 17 in NB5 is the actual vs prediction plot. Let's update Cell 17 to include plot_model_evaluation for each model.
cell17_src = '''# Actual vs prediction plots for each model
def plot_model_evaluation(
    history,
    y_true,
    y_pred,
    model_name,
    target_name,
    unit="gCO₂eq/kWh",
    plot_hours=500
):
    mae = np.mean(np.abs(y_true - y_pred))
    plot_hours = min(plot_hours, len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Loss curve
    axes[0].plot(
        history.history['loss'],
        label='Training Loss',
        color='#3498db'
    )
    axes[0].plot(
        history.history['val_loss'],
        label='Validation Loss',
        color='#e74c3c'
    )
    axes[0].set_title(
        f'Training vs Validation Loss — {model_name}',
        fontsize=14,
        fontweight='bold'
    )
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')

    # Actual vs Predicted
    axes[1].plot(
        y_true[:plot_hours],
        label=f'Actual {target_name}',
        color='#2c3e50',
        linewidth=1.5
    )
    axes[1].plot(
        y_pred[:plot_hours],
        label=f'Predicted {target_name} (MAE={mae:.2f})',
        color='#e74c3c',
        linewidth=1,
        alpha=0.8
    )
    axes[1].set_title(
        f'Actual vs Predicted — {model_name} ({plot_hours} jam)',
        fontsize=14,
        fontweight='bold'
    )
    axes[1].set_xlabel('Timestep (jam)')
    axes[1].set_ylabel(unit)
    axes[1].legend()
    axes[1].grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()

# 1. Univariate LSTM — CI
plot_model_evaluation(
    history=history_lstm_ci,
    y_true=y_lstm_ci_inv,
    y_pred=pred_lstm_ci_inv,
    model_name="Univariate LSTM — CI",
    target_name="CI",
    unit="gCO₂eq/kWh"
)

# 2. Univariate LSTM — RE
plot_model_evaluation(
    history=history_lstm_re,
    y_true=y_lstm_re_inv,
    y_pred=pred_lstm_re_inv,
    model_name="Univariate LSTM — RE",
    target_name="RE",
    unit="%"
)

# 3. Multivariate LSTM — CI
plot_model_evaluation(
    history=history_lstm_multi,
    y_true=y_multi_inv[:, 0],
    y_pred=pred_lstm_multi_inv[:, 0],
    model_name="Multivariate LSTM — CI",
    target_name="CI",
    unit="gCO₂eq/kWh"
)

# 4. Multivariate LSTM — RE
plot_model_evaluation(
    history=history_lstm_multi,
    y_true=y_multi_inv[:, 1],
    y_pred=pred_lstm_multi_inv[:, 1],
    model_name="Multivariate LSTM — RE",
    target_name="RE",
    unit="%"
)

# 5. BiLSTM — CI
plot_model_evaluation(
    history=history_bilstm_ci,
    y_true=y_lstm_ci_inv,
    y_pred=pred_bilstm_ci_inv,
    model_name="BiLSTM — CI",
    target_name="CI",
    unit="gCO₂eq/kWh"
)

# 6. BiLSTM — RE
plot_model_evaluation(
    history=history_bilstm_re,
    y_true=y_lstm_re_inv,
    y_pred=pred_bilstm_re_inv,
    model_name="BiLSTM — RE",
    target_name="RE",
    unit="%"
)

# 7. Multivariate BiLSTM — CI
plot_model_evaluation(
    history=history_bilstm_multi,
    y_true=y_multi_inv[:, 0],
    y_pred=pred_bilstm_multi_inv[:, 0],
    model_name="Multivariate BiLSTM — CI",
    target_name="CI",
    unit="gCO₂eq/kWh"
)

# 8. Multivariate BiLSTM — RE
plot_model_evaluation(
    history=history_bilstm_multi,
    y_true=y_multi_inv[:, 1],
    y_pred=pred_bilstm_multi_inv[:, 1],
    model_name="Multivariate BiLSTM — RE",
    target_name="RE",
    unit="%"
)
'''

nb5['cells'][17]['source'] = [cell17_src]

with open(nb5_path, 'w', encoding='utf-8') as f:
    json.dump(nb5, f, indent=1)

print("Updated 05_dani_dl_cyclical.ipynb successfully.")
