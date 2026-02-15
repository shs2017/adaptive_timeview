# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import roc_auc_score

# # Import base components from the core implementation
# from timeview_adaptive import PROJECT_ROOT, TimeviewAdaptive

# # =============================================================================
# # 1. Robust PhysioNet 2012 Data Loader
# # =============================================================================

# # class PhysioNet2012ExtendedLoader:
# #     """
# #     Advanced loader for PhysioNet 2012 that extracts:
# #     - Static features (Age, Gender, Height, Weight)
# #     - Target Trajectory (Mean Arterial Pressure - with fallbacks)
# #     - Dynamic Treatment (Mechanical Ventilation status)
# #     - Outcome Label (In-hospital Mortality)
# #     """
# #     def __init__(self, base_path=None, set_name='set-a'):
# #         self.base_path = base_path or PROJECT_ROOT / "data" / "physionet.org" / "files" / "challenge-2012" / "1.0.0"
# #         self.set_name = set_name
# #         self.outcomes_file = self.base_path / f"Outcomes-{set_name[-1]}.txt"
# #         self.data_dir = self.base_path / set_name

# #     def load(self, limit=None, min_length=5):
# #         if not self.outcomes_file.exists():
# #             raise FileNotFoundError(f"Outcomes file not found: {self.outcomes_file}")

# #         # Load outcomes (RecordID -> In-hospital_death)
# #         outcomes_df = pd.read_csv(self.outcomes_file)
# #         labels_map = dict(zip(outcomes_df['RecordID'].astype(int), outcomes_df['In-hospital_death']))

# #         X_list, T_list, Y_list, A_list, L_list = [], [], [], [], []
# #         static_feats = ['Age', 'Gender', 'Height', 'Weight']

# #         files = list(self.data_dir.glob("*.txt"))
# #         if limit: files = files[:limit]

# #         print(f"Processing {len(files)} files. Broadening search for clinical vitals...")
# #         stats = {"missing_outcome": 0, "no_bp_data": 0, "too_short": 0, "success": 0}

# #         for pfile in files:
# #             try:
# #                 rec_id = int(pfile.stem)
# #                 if rec_id not in labels_map:
# #                     stats["missing_outcome"] += 1
# #                     continue

# #                 df = pd.read_csv(pfile)

# #                 # Convert Time HH:MM to float hours
# #                 def parse_time(t_str):
# #                     if isinstance(t_str, (int, float)): return float(t_str)
# #                     parts = str(t_str).split(':')
# #                     if len(parts) != 2: return 0.0
# #                     return float(parts[0]) + float(parts[1])/60.0

# #                 df['Time'] = df['Time'].apply(parse_time)
# #                 ts_df = df.pivot_table(index='Time', columns='Parameter', values='Value', aggfunc='first')

# #                 # --- BLOOD PRESSURE SEARCH (Broad Fallback Logic) ---
# #                 map_val = None

# #                 # 1. Search for direct MAP (Invasive or Non-Invasive)
# #                 for col in ['MAP', 'NIMAP']:
# #                     if col in ts_df.columns:
# #                         map_val = ts_df[col]
# #                         break

# #                 # 2. Calculate MAP from Systolic/Diastolic if direct MAP is missing
# #                 if map_val is None:
# #                     for sys, dias in [('SysABP', 'DiasABP'), ('NISysABP', 'NIDiasABP')]:
# #                         if sys in ts_df.columns and dias in ts_df.columns:
# #                             map_val = (2 * ts_df[dias] + ts_df[sys]) / 3.0
# #                             break

# #                 if map_val is None:
# #                     stats["no_bp_data"] += 1
# #                     continue

# #                 # Clean trajectory using modern Pandas filling
# #                 map_series = map_val.interpolate().bfill().ffill()

# #                 if map_series.isna().all() or len(map_series) < min_length:
# #                     stats["too_short"] += 1
# #                     continue

# #                 # --- STATIC FEATURES ---
# #                 statics = []
# #                 for feat in static_feats:
# #                     val = df[df['Parameter'] == feat]['Value'].values
# #                     statics.append(float(val[0]) if len(val) > 0 and not np.isnan(val[0]) else np.nan)

# #                 # --- TREATMENT (MechVent) ---
# #                 # Assume 0 (no ventilation) if missing, otherwise forward fill status
# #                 if 'MechVent' in ts_df.columns:
# #                     mech_vent = ts_df['MechVent'].ffill().fillna(0).values
# #                 else:
# #                     mech_vent = np.zeros(len(ts_df))

# #                 X_list.append(statics)
# #                 T_list.append(ts_df.index.values)
# #                 Y_list.append(map_series.values)
# #                 A_list.append(mech_vent)
# #                 L_list.append(labels_map[rec_id])
# #                 stats["success"] += 1

# #             except Exception:
# #                 continue

# #         print(f"Final Stats: {stats}")

# #         if not X_list:
# #             raise ValueError("Zero patients passed filters. Check dataset integrity.")

# #         # Normalize Static Features
# #         X = np.array(X_list, dtype=np.float32)
# #         X = SimpleImputer(strategy='median').fit_transform(X)
# #         X = (X - X.mean(0)) / (X.std(0) + 1e-6)

# #         return (torch.tensor(X, dtype=torch.float32),
# #                 T_list, Y_list, A_list,
# #                 torch.tensor(L_list, dtype=torch.float32))

# # Import base components

# # =============================================================================
# # 1. Multi-Vital Data Loader
# # =============================================================================

# class PhysioNet2012ExtendedLoader:
#     def __init__(self, base_path=None, set_name='set-a'):
#         self.base_path = base_path or PROJECT_ROOT / "data" / "physionet.org" / "files" / "challenge-2012" / "1.0.0"
#         self.set_name = set_name
#         self.outcomes_file = self.base_path / f"Outcomes-{set_name[-1]}.txt"
#         self.data_dir = self.base_path / set_name

#     def load(self, limit=None, min_length=5):
#         outcomes_df = pd.read_csv(self.outcomes_file)
#         labels_map = dict(zip(outcomes_df['RecordID'].astype(int), outcomes_df['In-hospital_death'], strict=True))

#         X_list, T_list, Y_map_list, Y_hr_list, A_list, L_list = [], [], [], [], [], []
#         static_feats = ['Age', 'Gender', 'Height', 'Weight']

#         files = list(self.data_dir.glob("*.txt"))
#         if limit:
#             files = files[:limit]

#         print(f"Processing {len(files)} files for MAP and HR...")
#         stats = {"missing": 0, "success": 0}

#         for pfile in files:
#             try:
#                 rec_id = int(pfile.stem)
#                 if rec_id not in labels_map:
#                     continue

#                 df = pd.read_csv(pfile)
#                 def parse_time(t_str):
#                     parts = str(t_str).split(':')
#                     return float(parts[0]) + float(parts[1])/60.0

#                 df['Time'] = df['Time'].apply(parse_time)
#                 ts_df = df.pivot_table(index='Time', columns='Parameter', values='Value', aggfunc='first')

#                 # --- MAP Logic ---
#                 map_val = None
#                 for col in ['MAP', 'NIMAP']:
#                     if col in ts_df.columns:
#                         map_val = ts_df[col]
#                         break
#                 if map_val is None:
#                     if 'SysABP' in ts_df.columns and 'DiasABP' in ts_df.columns:
#                         map_val = (2 * ts_df['DiasABP'] + ts_df['SysABP']) / 3.0

#                 # --- HR Logic ---
#                 hr_val = ts_df['HR'] if 'HR' in ts_df.columns else None

#                 if map_val is None or hr_val is None:
#                     stats["missing"] += 1
#                     continue

#                 # Clean trajectories
#                 map_s = map_val.interpolate().bfill().ffill()
#                 hr_s = hr_val.interpolate().bfill().ffill()

#                 if len(map_s) < min_length or hr_s.isna().all():
#                     stats["missing"] += 1
#                     continue

#                 # Statics
#                 statics = [df[df['Parameter'] == f]['Value'].values[0] if len(df[df['Parameter'] == f]) > 0 else np.nan for f in static_feats]

#                 # Treatment
#                 mv = ts_df['MechVent'].ffill().fillna(0).values if 'MechVent' in ts_df.columns else np.zeros(len(ts_df))

#                 X_list.append(statics)
#                 T_list.append(ts_df.index.values)
#                 Y_map_list.append(map_s.values)
#                 Y_hr_list.append(hr_s.values)
#                 A_list.append(mv)
#                 L_list.append(labels_map[rec_id])
#                 stats["success"] += 1
#             except:
#                 continue

#         print(f"Final Stats: {stats}")
#         X = SimpleImputer(strategy='median').fit_transform(np.array(X_list))
#         X = (X - X.mean(0)) / (X.std(0) + 1e-6)

#         return torch.tensor(X, dtype=torch.float32), T_list, Y_map_list, Y_hr_list, A_list, torch.tensor(L_list)


# # =============================================================================
# # 2. Idea 1: Response-Aware TIMEVIEW (Causal)
# # =============================================================================

# class TimeviewIntervention(TimeviewAdaptive):
#     """
#     Extends TIMEVIEW to model treatment effects:
#     y(t) = Baseline(x, t) + TreatmentEffect(a, t)
#     """
#     def __init__(self, input_dim, n_basis=7, treatment_dim=1, **kwargs):
#         super().__init__(input_dim, n_basis, **kwargs)
#         # Learnable interaction: How treatment 'a' shifts coefficient distribution
#         self.treatment_weights = nn.Parameter(torch.randn(treatment_dim, n_basis) * 0.05)

#     def get_treatment_effect(self, treatments):
#         """Map treatment intensity to a shift in basis coefficients."""
#         return torch.matmul(treatments, self.treatment_weights)

#     def update_and_predict_counterfactual(self, x, t_obs, y_obs, a_obs, t_pred, counterfactual_a=None):
#         """
#         1. Infer baseline from history.
#         2. Predict future under a hypothetical treatment plan.
#         """
#         mu_0, Sigma_0 = self.encode(x)

#         # Adjust prior by observed treatment history
#         a_mean_obs = a_obs.mean(dim=1, keepdim=True)
#         mu_0_adj = mu_0 + self.get_treatment_effect(a_mean_obs)

#         # Bayesian Update
#         self.updater.sigma2 = self.sigma ** 2
#         Phi_obs = self.get_basis(t_obs).unsqueeze(0).expand(x.shape[0], -1, -1)
#         mu_post, Sigma_post = self.updater.update(mu_0_adj, Sigma_0, Phi_obs, y_obs)

#         # Counterfactual Prediction
#         mu_pred = mu_post
#         if counterfactual_a is not None:
#             # Shift the posterior mean by the delta in treatment
#             delta_eff = self.get_treatment_effect(counterfactual_a - a_mean_obs)
#             mu_pred = mu_post + delta_eff

#         return self.predict(mu_pred, Sigma_post, t_pred)

# # # =============================================================================
# # # 3. Idea 2: Decision Utility Metric
# # # =============================================================================

# # def evaluate_decision_utility(model, X, T_list, Y_list, L, n_obs_window=24):
# #     """
# #     Compares the predictive power of TIMEVIEW's inferred coefficients
# #     vs. Raw Statistical features for Mortality Prediction.
# #     """
# #     print(f"\n--- Evaluating Decision Utility (Observation Window: {n_obs_window} hrs) ---")
# #     model.eval()

# #     feats_model, feats_raw, targets = [], [], []

# #     for i in range(len(X)):
# #         t_arr, y_arr = T_list[i], Y_list[i]
# #         mask = t_arr <= n_obs_window
# #         if mask.sum() < 2: continue

# #         # Prepare inputs
# #         t_win = torch.tensor(t_arr[mask], dtype=torch.float32) / 48.0
# #         y_win = torch.tensor(y_arr[mask], dtype=torch.float32).unsqueeze(0)
# #         x_in = X[i].unsqueeze(0)

# #         # 1. TIMEVIEW Features: Posterior Mean Coefficients
# #         with torch.no_grad():
# #             mu_0, Sigma_0 = model.encode(x_in)
# #             Phi = model.get_basis(t_win).unsqueeze(0)
# #             mu_post, _ = model.updater.update(mu_0, Sigma_0, Phi, y_win)
# #             feats_model.append(mu_post.numpy().flatten())

# #         # 2. Raw Baseline: Stats of the window
# #         y_np = y_win.numpy().flatten()
# #         raw_vec = [y_np.mean(), y_np.std(), y_np[-1], (y_np[-1]-y_np[0])/(len(y_np)+1e-5)]
# #         feats_raw.append(raw_vec)

# #         targets.append(L[i].item())

# #     # Train/Test Evaluation
# #     X_m, X_r, y = np.array(feats_model), np.array(feats_raw), np.array(targets)
# #     split = int(0.8 * len(y))

# #     def get_auc(train_x, train_y, test_x, test_y):
# #         clf = LogisticRegression(max_iter=1000).fit(train_x, train_y)
# #         probs = clf.predict_proba(test_x)[:, 1]
# #         return roc_auc_score(test_y, probs)

# #     auc_m = get_auc(X_m[:split], y[:split], X_m[split:], y[split:])
# #     auc_r = get_auc(X_r[:split], y[:split], X_r[split:], y[split:])

# #     print(f"Results on {len(y)-split} patients:")
# #     print(f"  Raw Stats AUC:  {auc_r:.4f}")
# #     print(f"  TIMEVIEW AUC:   {auc_m:.4f}")
# #     print(f"  Improvement:    {auc_m - auc_r:+.4f}")

# #     return auc_m, auc_r

# # =============================================================================
# # 2. Updated Utility Evaluation
# # =============================================================================

# def evaluate_multi_vital_utility(model, X, T_list, Y_map, Y_hr, L, n_obs_window=24):
#     print("\n--- Evaluating Multi-Vital Utility (MAP Coefficients + HR Stats) ---")
#     model.eval()
#     feats_model, feats_raw, targets = [], [], []

#     # Global HR stats for normalization
#     hr_all = np.concatenate(Y_hr)
#     hr_m, hr_s = hr_all.mean(), hr_all.std()

#     for i in range(len(X)):
#         t_arr, y_m, y_h = T_list[i], Y_map[i], Y_hr[i]
#         mask = t_arr <= n_obs_window
#         if mask.sum() < 3:
#             continue

#         # TIMEVIEW features from MAP
#         t_win = torch.tensor(t_arr[mask], dtype=torch.float32) / 48.0
#         y_m_win = torch.tensor(y_m[mask], dtype=torch.float32).unsqueeze(0)
#         # Normalize MAP for the model
#         y_m_win = (y_m_win - 80) / 20.0

#         with torch.no_grad():
#             mu_0, Sigma_0 = model.encode(X[i:i+1])
#             Phi = model.get_basis(t_win).unsqueeze(0)
#             mu_post, _ = model.updater.update(mu_0, Sigma_0, Phi, y_m_win)
#             map_coeffs = mu_post.numpy().flatten()

#         # HR features (Raw Stats)
#         yh_win = y_h[mask]
#         hr_stats = [(yh_win.mean()-hr_m)/hr_s, yh_win.std()/hr_s, (yh_win[-1]-hr_m)/hr_s]

#         # MAP features (Raw Stats)
#         ym_win = y_m[mask]
#         map_stats = [ym_win.mean(), ym_win.std(), ym_win[-1]]

#         # Combined Feature Vectors
#         feats_model.append(np.concatenate([map_coeffs, hr_stats]))
#         feats_raw.append(np.concatenate([map_stats, hr_stats]))
#         targets.append(L[i].item())

#     X_m, X_r, y = np.array(feats_model), np.array(feats_raw), np.array(targets)
#     split = int(0.8 * len(y))

#     # We use a slightly more robust classifier (Random Forest) to handle the 0.70 target
#     from sklearn.ensemble import RandomForestClassifier
#     def get_auc(tx, ty, vx, vy):
#         clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(tx, ty)
#         return roc_auc_score(vy, clf.predict_proba(vx)[:, 1])

#     auc_m = get_auc(X_m[:split], y[:split], X_m[split:], y[split:])
#     auc_r = get_auc(X_r[:split], y[:split], X_r[split:], y[split:])

#     print(f"Results: Raw (MAP+HR) AUC: {auc_r:.4f} | TIMEVIEW (MAP Coeffs+HR) AUC: {auc_m:.4f}")
#     return auc_m


# # =============================================================================
# # 4. Main Experiment Workflow
# # =============================================================================

# if __name__ == "__main__":
#     # 1. Load Data
#     print("Loading PhysioNet 2012...")
#     loader = PhysioNet2012ExtendedLoader()
#     X, T, Y_map, Y_hr, A, L = loader.load(limit=500)

#     # 2. Preprocessing (normalize MAP trajectories)
#     all_y = np.concatenate(Y_map)
#     mean_y, std_y = all_y.mean(), all_y.std()
#     Y_norm = [(y - mean_y) / (std_y + 1e-6) for y in Y_map]

#     # 3. Model & Hyperparams
#     batch_size = 32
#     model = TimeviewIntervention(input_dim=X.shape[1], n_basis=7)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     # # 4. Training Loop (Batched Encoder)
#     # print(f"\nTraining Timeview-Intervention (Batch Size: {batch_size})...")
#     # model.train()

#     # for epoch in range(31):
#     #     indices = torch.randperm(len(X))
#     #     epoch_loss = 0

#     #     for i in range(0, len(X), batch_size):
#     #         batch_idx = indices[i:i+batch_size]
#     #         optimizer.zero_grad()

#     #         # --- THE FIX: Batch the static features for the encoder ---
#     #         x_batch = X[batch_idx]
#     #         # This calls BatchNorm with size (32, 4), which works!
#     #         mu_batch, Sigma_batch = model.encode(x_batch)

#     #         batch_loss_val = 0
#     #         # Now iterate through the pre-calculated priors for individual trajectories
#     #         for j, idx in enumerate(batch_idx):
#     #             # Slice the j-th prior from the batch
#     #             mu_0 = mu_batch[j:j+1]
#     #             Sigma_0 = Sigma_batch[j:j+1]

#     #             # Setup individual trajectory data
#     #             t_in = torch.tensor(T[idx], dtype=torch.float32) / 48.0
#     #             y_in = torch.tensor(Y_norm[idx], dtype=torch.float32).unsqueeze(0)
#     #             a_mean = torch.tensor([[A[idx].mean()]], dtype=torch.float32)

#     #             # Forward pass using the sliced prior
#     #             mu_adj = mu_0 + model.get_treatment_effect(a_mean)
#     #             y_mean, y_var = model.predict(mu_adj, Sigma_0, t_in)

#     #             # Loss calculation
#     #             nll = (0.5 * torch.log(y_var) + 0.5 * (y_in - y_mean)**2 / y_var).mean()
#     #             loss = nll + 0.01 * model.kl_divergence(mu_0, Sigma_0)

#     #             # Accumulate gradients
#     #             scaled_loss = loss / len(batch_idx)
#     #             scaled_loss.backward()
#     #             batch_loss_val += scaled_loss.item()

#     #         optimizer.step()
#     #         epoch_loss += batch_loss_val

#     #     if epoch % 10 == 0:
#     #         print(f"Epoch {epoch}: Avg Loss {epoch_loss / (len(X)/batch_size):.4f}")
# # 4. Training Loop (Loss Accumulation Fix)
#     print(f"\nTraining Timeview-Intervention (Batch Size: {batch_size})...")
#     model.train()

#     for epoch in range(31):
#         indices = torch.randperm(len(X))
#         epoch_loss = 0

#         for i in range(0, len(X), batch_size):
#             batch_idx = indices[i:i+batch_size]
#             optimizer.zero_grad()

#             # Compute priors for the whole batch (One graph)
#             x_batch = X[batch_idx]
#             mu_batch, Sigma_batch = model.encode(x_batch)

#             total_batch_loss = 0  # We will sum losses here

#             for j, idx in enumerate(batch_idx):
#                 mu_0 = mu_batch[j:j+1]
#                 Sigma_0 = Sigma_batch[j:j+1]

#                 t_in = torch.tensor(T[idx], dtype=torch.float32) / 48.0
#                 y_in = torch.tensor(Y_norm[idx], dtype=torch.float32).unsqueeze(0)
#                 a_mean = torch.tensor([[A[idx].mean()]], dtype=torch.float32)

#                 # Forward pass
#                 mu_adj = mu_0 + model.get_treatment_effect(a_mean)
#                 y_mean, y_var = model.predict(mu_adj, Sigma_0, t_in)

#                 # Calculate individual loss
#                 nll = (0.5 * torch.log(y_var) + 0.5 * (y_in - y_mean)**2 / y_var).mean()
#                 kl = 0.01 * model.kl_divergence(mu_0, Sigma_0)

#                 # Add to total batch loss (normalized by batch size)
#                 total_batch_loss += (nll + kl) / len(batch_idx)

#             # --- THE FIX: Backward once per batch ---
#             total_batch_loss.backward()
#             optimizer.step()

#             epoch_loss += total_batch_loss.item()

#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}: Avg Loss {epoch_loss / (len(X)/batch_size):.4f}")

#     # 5. Visualization & Evaluation
#     model.eval()
#     evaluate_multi_vital_utility(model, X, T, Y_map, Y_hr, L)
