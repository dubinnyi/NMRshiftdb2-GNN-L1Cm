"""
Internationalization (i18n) helper module for the NMR GNN pipeline.

Base language in source code is English. Russian output for console prints,
CLI help descriptions, and plot labels is enabled automatically when the
terminal locale is Russian (e.g. LANG=ru_RU.UTF-8) or when NMR_LANG=ru.
"""

import os


def is_russian_locale() -> bool:
    """
    Return True if the terminal environment specifies a Russian locale
    (e.g., LANG=ru_RU.UTF-8, LC_ALL=ru_RU.UTF-8, or NMR_LANG=ru).
    """
    for env_var in ("NMR_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val.lower().startswith("ru")
    return False


RU_MESSAGES = {
    # === respredict_pipeline.py: CLI arguments & validation ===
    "NMR GNN Training Pipeline":
        "Пайплайн обучения GNN для предсказания химических сдвигов ЯМР",
    "Model architecture: v2 (lightweight Williams-2023 polynomial GNN + 3D RBF + CH2 features + multi-head beta-NLL) or legacy (original GraphVertModel). Default: v2":
        "Архитектура модели: v2 (легкая полиномиальная GNN Williams-2023 + 3D RBF + признаки CH2 + многоголовый beta-NLL) или legacy (оригинальная GraphVertModel). По умолчанию: v2",
    "Training mode: warmup (mu+sigma), molcalib (+c_m), auto (warmup -> molcalib)":
        "Режим: warmup (mu+sigma), molcalib (+c_m), auto (warmup -> molcalib)",
    "Training mode: warmup (mu+sigma), molcalib (+c_m), auto (warmup -> molcalib). Default: warmup for beta_nll, auto for nolog":
        "Режим: warmup (mu+sigma), molcalib (+c_m), auto (warmup -> molcalib). По умолчанию: warmup для beta_nll, auto для nolog",
    "Uncertainty loss function: beta_nll (Seitzer et al., ICLR 2022) or nolog (legacy Jonas & Kuhn). Default: beta_nll":
        "Функция потерь неопределенности: beta_nll (Seitzer et al., ICLR 2022) или nolog (оригинальная Jonas & Kuhn). По умолчанию: beta_nll",
    "Beta exponent in [0, 1] for beta-NLL loss (0=standard NLL, 0.5=default, 1=MSE mean gradient)":
        "Показатель beta в [0, 1] для beta-NLL (0=обычный NLL, 0.5=по умолчанию, 1=градиент MSE по среднему)",
    "Number of probabilistic bootstrap readout heads in v2 architecture (default: 5)":
        "Число вероятностных bootstrap-голов в архитектуре v2 (по умолчанию: 5)",
    "Number of heads trained per molecule in v2 bootstrap subsampling (default: 4)":
        "Число голов, обучаемых на каждую молекулу в bootstrap-подвыборке v2 (по умолчанию: 4)",
    "Hidden GNN width for v2 architecture (default: 256, lightweight as in Williams & Jonas 2023)":
        "Ширина скрытого слоя GNN в архитектуре v2 (по умолчанию: 256, легкая модель как в Williams & Jonas 2023)",
    "Number of GNN message-passing layers for v2 architecture (default: 8)":
        "Число слоев передачи сообщений GNN в архитектуре v2 (по умолчанию: 8)",
    "Training batch size (default: 32 for v2, 16 for legacy)":
        "Размер батча обучения (по умолчанию: 32 для v2, 16 для legacy)",
    "Disable 3D spatial features in v2 (ablation: pure 2D + rich chemical + CH2 features)":
        "Отключить 3D-пространственные признаки в v2 (абляция: чистый 2D + расширенные химические + CH2 признаки)",
    "Dropout rate for v2 GNN and readout heads (default: 0.05)":
        "Коэффициент dropout для слоев GNN и голов v2 (по умолчанию: 0.05)",
    "Adam weight decay (L2 regularization) (default: v2=1e-4, legacy=1e-6)":
        "Параметр weight decay (L2-регуляризация) для Adam (по умолчанию: v2=1e-4, legacy=1e-6)",
    "Initial predicted std (in ppm) for beta-NLL softplus variance head. Default: v2 13C=2.5, legacy 13C=5.0, 1H=1.0":
        "Начальная предсказываемая std (в ppm) для softplus-головы дисперсии beta-NLL. По умолчанию: v2 13C=2.5, legacy 13C=5.0, 1H=1.0",
    "Maximum predicted std clamp (in ppm) for beta-NLL variance head. Default: 13C=100.0, 1H=10.0":
        "Максимальное ограничение std (в ppm) для головы дисперсии beta-NLL. По умолчанию: 13C=100.0, 1H=10.0",
    "ERROR: --arch v2 (NMRResPredictV2) requires --loss beta_nll. Use --arch legacy --loss nolog for the original Jonas & Kuhn model.":
        "ОШИБКА: --arch v2 (NMRResPredictV2) требует --loss beta_nll. Используйте --arch legacy --loss nolog для оригинальной модели Jonas & Kuhn.",
    "ERROR: --beta must be in [0.0, 1.0], got {beta}":
        "ОШИБКА: параметр --beta должен быть в [0.0, 1.0], получено {beta}",
    "ERROR: --loss beta_nll is currently implemented for stage 'warmup' only. Pass --protocol warmup (or omit --protocol), or use --loss nolog for legacy molcalib/auto.":
        "ОШИБКА: --loss beta_nll сейчас реализован только для этапа 'warmup'. Укажите --protocol warmup (или опустите --protocol), либо используйте --loss nolog для старого режима molcalib/auto.",
    "Path to pre-trained weights file (.state or .model)":
        "Путь к файлу предобученных весов (.state или .model)",
    "Target nucleus for chemical shift prediction":
        "Целевое ядро для предсказаний",
    "Minimum uncertainty floor (std_regularize) in ppm. Default: 13C=0.1, 1H=0.01":
        "Минимальная ошибка (std_regularize) в ppm. По умолчанию: 13C=0.1, 1H=0.01",
    "Maximum number of epochs per stage":
        "Максимальное количество эпох на один этап",
    "Dataset cross-validation split index (0, 1, etc.)":
        "Индекс кросс-валидации датасета (0, 1 и т.д.)",
    "Quick smoke test run (2 batches, 2 epochs) to verify pipeline integrity":
        "Быстрый прогон (2 батча, 2 эпохи) для проверки пайплайна на ошибки",
    "Use L1 penalty |C_m| (in ppm) for molecule calibration in molcalib":
        "Использовать L1-штраф |C_m| (в ppm) для калибровки в molcalib",
    "Use L2 penalty C_m^2 (in ppm^2) for molecule calibration in molcalib (default)":
        "Использовать L2-штраф C_m^2 (в ppm^2) для калибровки в molcalib (по умолчанию)",
    "Explicit LAMBDA_PENALTY weight (if omitted, selected automatically by norm and nucleus)":
        "Явный вес штрафа LAMBDA_PENALTY (если не указан, выбирается автоматически под норму и ядро)",
    "Explicit learning rate. Default: warmup=1e-4, molcalib=3e-5":
        "Явный шаг обучения (learning rate). По умолчанию: warmup=1e-4, molcalib=3e-5",
    "Explicit learning rate. Default: v2=5e-4, legacy warmup=1e-4, molcalib=3e-5":
        "Явный шаг обучения (learning rate). По умолчанию: v2=5e-4, legacy warmup=1e-4, molcalib=3e-5",
    "Allowed per-stage increase in Student-t df divergence (df_train - df_test). Default: 0.42 (two-stage cap <= 0.84)":
        "Допустимый прирост разницы степеней свободы Стьюдента (df_train - df_test) на один этап (по умолчанию: 0.42, суммарно за два этапа <= 0.84)",
    "Minimum number of base training epochs before early stopping is armed (default: v2 warmup=10, legacy warmup=50, molcalib=20)":
        "Минимальное число эпох базового обучения до включения критериев остановки (по умолчанию: v2 warmup=10, legacy warmup=50, molcalib=20)",
    "Evaluate validation metrics and Student-t parameters every N epochs (default: v2=1, legacy=5)":
        "Вычислять метрики валидации и параметры Стьюдента каждые N эпох (по умолчанию: v2=1, legacy=5)",
    "Save full prediction CSVs and Z-score/Stats PNG plots every N epochs (default: v2=5, legacy=10)":
        "Сохранять полные CSV предсказаний и PNG-графики Z-score/Stats каждые N эпох (по умолчанию: v2=5, legacy=10)",
    "WARNING: Unrecognized command-line argument: '{arg}'":
        "WARNING: Нераспознанный аргумент командной строки: '{arg}'",
    "ERROR: Stage 'molcalib' requires a pre-trained checkpoint path via --weights <path> or --checkpoint <path> (or use --protocol auto for the full warmup -> molcalib cycle).":
        "ОШИБКА: Для запуска этапа 'molcalib' необходимо указать путь к чекпоинту предобученной модели через --weights <path> или --checkpoint <path> (или используй --protocol auto для полного цикла warmup -> molcalib).",
    "  [Checkpoint Meta] Updating net_params['{key}']: {old_val} -> {new_val}":
        "  [Checkpoint Meta] Обновление net_params['{key}']: {old_val} -> {new_val}",
    "  [Checkpoint Meta] Failed to read {path}: {err}":
        "  [Checkpoint Meta] Не удалось прочитать {path}: {err}",
    "Unknown checkpoint format: {obj_type}":
        "Неизвестный формат чекпоинта: {obj_type}",
    "!!! SMOKE TEST MODE ENABLED (2 batches, 2 epochs) !!!":
        "!!! ВКЛЮЧЕН РЕЖИМ SMOKE TEST (2 батча, 2 эпохи) !!!",
    "Dataset size (>= {min_atoms} target {spect_set} nuclei): Train={n_train}, Test={n_test}":
        "Размер выборки (>= {min_atoms} целевых ядер {spect_set}): Train={n_train}, Test={n_test}",
    "ERROR: Specified checkpoint file not found: {path}":
        "ОШИБКА: Указанный файл чекпоинта не найден: {path}",
    "Loading pre-trained weights from {path}...":
        "Загрузка предобученных весов из {path}...",
    "Final weights for stage {stage} saved to:\n  {state_path}\n  {model_path}":
        "Финальные веса этапа {stage} сохранены :\n  {state_path}\n  {model_path}",
    "\n[SMOKE TEST] Test completed without errors. Automatically cleaning up temporary files...":
        "\n[SMOKE TEST] Тест успешно завершен без ошибок. Автоматическая очистка временных файлов...",
    "[SMOKE TEST] All temporary checkpoints, CSVs, plots, and TensorBoard logs removed.\n":
        "[SMOKE TEST] Все временные чекпоинты, CSV, графики и логи TensorBoard удалены.\n",

    # === graph_conv_many_nuc_util.py: training runner & EarlyStopping ===
    "  [Warning] Skipped {skipped} batches with non-finite loss in epoch {epoch_i}.":
        "  [Внимание] Пропущено {skipped} батчей с неконечным Loss на эпохе {epoch_i}.",
    "\n[WARNING] Non-finite weights/loss at epoch {epoch_i}! Restoring last valid weights, reducing lr to {lr:.2e}, and continuing training.":
        "\n[ВНИМАНИЕ] Неконечные веса/Loss на эпохе {epoch_i}! Откат к последним валидным весам, снижение lr до {lr:.2e} и продолжение обучения.",
    "\n[ERROR] NaN/Inf detected in Train Loss at epoch {epoch_i}! Aborting training and restoring best weights.":
        "\n[ОШИБКА] Обнаружен NaN/Inf в Train Loss на эпохе {epoch_i}! Прерывание обучения и откат к лучшим весам.",
    "\n--- EARLY STOPPING TRIGGERED at epoch {epoch_i} ---":
        "\n--- ОБУЧЕНИЕ ОСТАНОВЛЕНО ДОСРОЧНО на эпохе {epoch_i} ---",
    "Reason: {reason}":
        "Причина: {reason}",
    "Restoring best model weights (epoch {best_epoch}, {metric_name}: {best_loss:.4f}) before finishing stage.":
        "Восстановление лучших весов модели (эпоха {best_epoch}, {metric_name}: {best_loss:.4f}) перед завершением этапа.",
    "  [EarlyStop] WARNING: Val Loss = {val_loss} (NaN/Inf) at epoch {epoch_i}! Ignoring invalid validation step.":
        "  [EarlyStop] ВНИМАНИЕ: Val Loss = {val_loss} (NaN/Inf) на эпохе {epoch_i}! Пропуск невалидной проверки.",
    "  [EarlyStop] WARNING: Val Loss = {val_loss} (NaN/Inf)! Stopping training.":
        "  [EarlyStop] ВНИМАНИЕ: Val Loss = {val_loss} (NaN/Inf)! Остановка обучения.",
    "NaN/Inf in validation Loss at epoch {epoch_i}":
        "NaN/Inf в валидационном Loss на эпохе {epoch_i}",
    "  [EarlyStop Base] Epoch {epoch_i} < min_epochs ({min_epochs}) or scale={sc:.2f} > 3.0: early stopping disarmed during initial convergence.":
        "  [EarlyStop База] Эпоха {epoch_i} < min_epochs ({min_epochs}) или scale={sc:.2f} > 3.0: критерии остановки отключены на время начальной сходимости.",
    "  [EarlyStop DF] Initial divergence {d_nu_key} = {d_nu:.2f} (tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}). Stage '{stage}' threshold: {limit:.2f}":
        "  [EarlyStop DF] Начальное расхождение {d_nu_key} = {d_nu:.2f} (tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}). Порог остановки этапа '{stage}': {limit:.2f}",
    "  [EarlyStop DF] Student-t overfitting threshold exceeded ({detail}, tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}). DF counter: {cnt}/{patience} (best weights locked at epoch {best_epoch}).":
        "  [EarlyStop DF] Превышен порог переобучения Стьюдента ({detail}, tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}). Счетчик DF: {cnt}/{patience} (лучшие веса зафиксированы на эпохе {best_epoch}).",
    "Student-t degrees-of-freedom divergence exceeded threshold ({detail}) for {cnt} consecutive checks.":
        "Расхождение степеней свободы Стьюдента превысило порог ({detail}) {cnt} проверки подряд.",
    "  [EarlyStop DF] Student-t within bounds: {d_nu_key}={d_nu:.2f} <= {limit:.2f} (tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}).":
        "  [EarlyStop DF] Норма Стьюдента: {d_nu_key}={d_nu:.2f} <= {limit:.2f} (tr_nu={tr_nu:.2f}, te_nu={te_nu:.2f}).",
    "  [EarlyStop] Initialized at epoch {epoch_i}: {metric_name} = {best_loss:.4f}":
        "  [EarlyStop] Инициализация на эпохе {epoch_i}: {metric_name} = {best_loss:.4f}",
    "  [EarlyStop] Improved at epoch {epoch_i}! {metric_name} decreased from {old_loss:.4f} to {new_loss:.4f}":
        "  [EarlyStop] Улучшение на эпохе {epoch_i}! {metric_name} снизился с {old_loss:.4f} до {new_loss:.4f}",
    "  [EarlyStop] {metric_name} = {target_val:.4f}, but weights NOT updated due to d_nu overfitting (kept epoch {best_epoch} with {metric_name} = {best_loss:.4f}).":
        "  [EarlyStop] {metric_name} = {target_val:.4f}, но веса НЕ обновлены из-за переобучения по d_nu (сохранена эпоха {best_epoch} с {metric_name} = {best_loss:.4f}).",
    "  [EarlyStop] No improvement in {metric_name} ({target_val:.4f}). Counter: {cnt}/{patience}. Best ({best_loss:.4f}) at epoch {best_epoch}":
        "  [EarlyStop] Нет улучшений {metric_name} ({target_val:.4f}). Счетчик: {cnt}/{patience}. Лучший ({best_loss:.4f}) на эпохе {best_epoch}",
    "{metric_name} did not improve for the last {epochs} epochs.":
        "{metric_name} не улучшался последние {epochs} эпох.",

    # === plot_z_score.py: plots, metrics CSV & standalone CLI ===
    "[{title_prefix}] WARNING: z-score array is empty (model weights may contain NaN). Skipping plot.":
        "[{title_prefix}] ПРЕДУПРЕЖДЕНИЕ: Массив z-score пуст (возможно, веса модели содержат NaN). Пропуск отрисовки.",
    "Data (z)":
        "Данные (z)",
    "Student-t\n(df={nu:.2f}, loc={loc:.4f}, scale={scale:.4f})":
        "t-Стьюдент\n(df={nu:.2f}, loc={loc:.4f}, scale={scale:.4f})",
    "Ideal: N(0,1)":
        "Идеал: N(0,1)",
    " ({n_atoms} atoms, {n_mols} molecules)":
        " ({n_atoms} атомов, {n_mols} молекул)",
    "{prefix_str}{title_prefix}: Z-score Histogram{counts_str}":
        "{prefix_str}{title_prefix}: Гистограмма z-score{counts_str}",
    "Probability Density":
        "Плотность вероятности",
    "Empirical CDF":
        "Эмпирическая CDF",
    "Student-t CDF":
        "t-Стьюдент CDF",
    "Ideal: N(0,1) CDF":
        "Идеал: N(0,1) CDF",
    "{title_prefix}: Cumulative Distribution (CDF)":
        "{title_prefix}: Интегральное распределение",
    "Probability (P(X < z))":
        "Вероятность (P(X < z))",
    "{prefix_str}{title_prefix}: Sigma Distribution [ppm]":
        "{prefix_str}{title_prefix}: Распределение Sigma [ppm]",
    "Predicted Uncertainty (sigma) [ppm]":
        "Предсказанная ошибка (sigma) [ppm]",
    "Frequency (atoms)":
        "Частота (атомы)",
    "{title_prefix}: Per-Molecule Shift (C_m) [ppm] [1-99 percentile]":
        "{title_prefix}: Сдвиг молекулы (C_m) [ppm] [1-99 перцентиль]",
    "Frequency (molecules)":
        "Частота (молекулы)",
    "  [Metrics Ep {epoch_i:04d}] Appended metrics row to: {path} (te_mae_cal={te_mae_cal:.4f}, d_nu_cal={d_nu_cal:.2f}, ratio_sc_cal={ratio_sc_cal:.2f})":
        "  [Metrics Ep {epoch_i:04d}] Записана строка метрик в: {path} (te_mae_cal={te_mae_cal:.4f}, d_nu_cal={d_nu_cal:.2f}, ratio_sc_cal={ratio_sc_cal:.2f})",
    "  [Plot] Skipping plot save {path}: insufficient valid data (possible NaN).":
        "  [Plot] Пропуск сохранения графиков {path}: недостаточно валидных данных (возможно, NaN).",
    "Z-score plots saved to: '{path}'":
        "Графики Z-score сохранены в: '{path}'",
    "Statistics plots saved to: '{path}'":
        "Графики статистики сохранены в: '{path}'",
    "  [Plot/CSV Ep {epoch_i:04d}] Saved CSVs: {train_path}, {test_path}":
        "  [Plot/CSV Ep {epoch_i:04d}] Сохранены CSV: {train_path}, {test_path}",
    "Unknown object type in {path}: {obj_type}":
        "Неизвестный тип объекта в {path}: {obj_type}",
    "Loading metadata: {path}":
        "Загрузка метаданных: {path}",
    "YES":
        "ДА",
    "NO":
        "НЕТ",
    "Nucleus: {spect_set} | Protocol: {protocol} (subtract C_m={apply_str}) | std_reg={std_reg}":
        "Ядро: {spect_set} | Протокол: {protocol} (вычитание C_m={apply_str}) | std_reg={std_reg}",
    "Nucleus: {spect_set} | Loss: {loss_type} | Protocol: {protocol} (subtract C_m={apply_str}) | std_reg={std_reg}":
        "Ядро: {spect_set} | Лосс: {loss_type} | Протокол: {protocol} (вычитание C_m={apply_str}) | std_reg={std_reg}",
    "  [Metrics Ep {epoch_i:04d}] Appended metrics row to: {path} (te_mae={te_mae:.4f}, te_rmse={te_rmse:.4f}, te_nll={te_nll:.4f}, cov68={cov68:.3f}, cov95={cov95:.3f}, d_nu_raw={d_nu_raw:.2f})":
        "  [Metrics Ep {epoch_i:04d}] Записана строка метрик в: {path} (te_mae={te_mae:.4f}, te_rmse={te_rmse:.4f}, te_nll={te_nll:.4f}, cov68={cov68:.3f}, cov95={cov95:.3f}, d_nu_raw={d_nu_raw:.2f})",
    "Explicit loss type override (if omitted, inferred from metadata or filename)":
        "Явное указание типа лосса (если не указано, определяется из метаданных или имени файла)",
    "Found existing prediction CSV files (model evaluation not required):":
        "Найдены готовые CSV-файлы предсказаний (расчет по модели не требуется):",
    "Neither prediction CSV files ({csv_path}) nor model weights file ({state_path}) were found.":
        "Не найдены ни CSV-файлы ({csv_path}), ни файл весов модели ({state_path}).",
    "A .meta or .json file is required to compute predictions from model weights ({state_path}).":
        "Для расчета предсказаний из весов модели ({state_path}) необходим .meta или .json файл.",
    "Dataset file not found: {path}":
        "Не найден файл датасета: {path}",
    "Reading source dataset: {path}":
        "Чтение исходного датасета: {path}",
    "Initializing model from checkpoint: {path}":
        "Инициализация модели из чекпоинта: {path}",
    "Computing predictions for training set...":
        "Вычисление предсказаний для тренировочной выборки...",
    "Computing predictions for test set...":
        "Вычисление предсказаний для тестовой выборки...",
    "Saving results to {train_path} and {test_path}...":
        "Сохранение результатов в {train_path} и {test_path}...",
    "Plot Z-score and Sigma/C_m distributions from a checkpoint or prediction CSV":
        "Отрисовка Z-score и распределений Sigma/C_m по чекпоинту или CSV",
    "Path to model file (.state/.model) OR prediction CSV file (*_preds_train.csv)":
        "Путь к файлу модели (.state/.model) ИЛИ к CSV-файлу предсказаний (*_preds_train.csv)",
    "Path to .meta or .json file (optional, auto-detected if omitted)":
        "Путь к .meta или .json файлу (необязательно, определяется автоматически)",
    "Path to .pickle dataset file (required only when prediction CSVs are absent)":
        "Путь к .pickle датасету (требуется только если нет готовых CSV)",
    "Explicit protocol override (if omitted, inferred from metadata or filename)":
        "Явное указание протокола (если не указано, определяется из метаданных или имени файла)",
    "Explicit std_regularize override (defaults to value in metadata)":
        "Явное указание std_regularize (по умолчанию берется из метаданных)",
    "Output directory for plots (default: plots_13C or plots_1H)":
        "Папка для сохранения графиков (по умолчанию: plots_13C или plots_1H)",
    "Force recomputation of predictions from the model even if CSVs exist":
        "Принудительно пересчитать предсказания по модели даже при наличии CSV",

    # === convert_13C_to_1H_state.py ===
    "Convert 13C state to initialize 1H model (Partial Transfer Learning)":
        "Конвертация 13C state для инициализации 1H модели (Partial Transfer Learning)",
    "Path to trained 13C .state file":
        "Путь к готовому .state файлу обученной модели 13C",
    "Path to 13C .meta or .json file (feature extractor architectures are identical)":
        "Путь к .meta или .json файлу от 13C (архитектуры экстрактора идентичны)",
    "Output path for the new .state file ready for 1H training":
        "Путь для сохранения нового .state файла, готового для запуска 1H",
    "Loading 13C metadata from {path}...":
        "Загрузка метаданных 13C из {path}...",
    "Loading 13C weights from {path}...":
        "Загрузка весов 13C из {path}...",
    "Successfully transferred graph feature extractor layers (gml.*): {count}":
        "Успешно перенесено слоев графового экстрактора (gml.*): {count}",
    "Done! New state saved to: {path}":
        "Готово! Новый state сохранен в: {path}",

    # === sd_to_pickle.py ===
    "No data found for nucleus {nucleus}!":
        "Нет данных для ядра {nucleus}!",
    "Saving {nucleus}... Train: {n_train}, Test: {n_test}":
        "Сохранение {nucleus}... Train: {n_train}, Test: {n_test}",
    "-> File created: {path}":
        "-> Файл создан: {path}",
    "Streaming {path}...":
        "Чтение {path} потоком...",
    "Processing SDF":
        "Обработка SDF",
    "\n--- BUILD SUMMARY ---":
        "\n--- ИТОГИ СБОРКИ ---",
}


def _(msg: str, **kwargs) -> str:
    """
    Translate `msg` to Russian if the current terminal locale is Russian,
    otherwise keep the base English string. Formats with `kwargs` if given.
    """
    text = RU_MESSAGES.get(msg, msg) if is_russian_locale() else msg
    if kwargs:
        return text.format(**kwargs)
    return text
