-- injlab 数据库 schema —— 全系统唯一真源
-- ER 图由本文件生成（python -m injlab er），界面与 API 都是它的视图。
PRAGMA foreign_keys = ON;

-- 1) 数据集版本：一次快照一行，保证任何实验结果都能追溯到一个不可变的数据划分
CREATE TABLE IF NOT EXISTS dataset_version (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    source_mix    TEXT,                       -- 各来源条数，如 {"toy":90,"wildguard":1200}
    split_rule    TEXT,                       -- 如 "stratified,test_ratio=0.3,seed=20260101"
    sha256        TEXT,                       -- 全量样本哈希，用于发现"数据被悄悄改过"
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 2) 样本
CREATE TABLE IF NOT EXISTS sample (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_version_id INTEGER NOT NULL REFERENCES dataset_version(id),
    text               TEXT    NOT NULL,
    label              TEXT    NOT NULL,      -- benign / direct_injection / indirect_injection
    source             TEXT,                  -- 语料来源，跨集泛化实验要用
    split              TEXT    NOT NULL,      -- train / test
    sha256             TEXT    NOT NULL,
    is_hard_negative   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sample_label_split ON sample(label, split);
CREATE UNIQUE INDEX IF NOT EXISTS ux_sample_hash ON sample(dataset_version_id, sha256);

-- 3) 攻击方法（攻击库的注册表）
CREATE TABLE IF NOT EXISTS attack (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    family       TEXT    NOT NULL,            -- obfuscation / paraphrase / gradient / structural
    params_json  TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 4) 攻击生成样本：is_success 一列让实验具备"攻击成功率"这个维度
CREATE TABLE IF NOT EXISTS attack_sample (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    attack_id        INTEGER NOT NULL REFERENCES attack(id),
    source_sample_id INTEGER NOT NULL REFERENCES sample(id),
    text             TEXT    NOT NULL,
    achieved_evasion INTEGER,                 -- NULL = 未评测；1 = 该样本逃逸成功
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 5) 防御策略：阈值、是否做输入规范化、分块参数 —— 可被实验矩阵逐项扫描
CREATE TABLE IF NOT EXISTS defense_policy (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    threshold    REAL    NOT NULL,
    normalize_on INTEGER NOT NULL DEFAULT 1,
    chunk_size   INTEGER NOT NULL DEFAULT 256,
    chunk_stride INTEGER NOT NULL DEFAULT 128,
    params_json  TEXT
);

-- 6) 模型版本：红队回路每迭代一轮就多一行
CREATE TABLE IF NOT EXISTS model_version (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    arch       TEXT    NOT NULL,              -- keyword / tfidf_lr / distilbert
    ckpt_path  TEXT,
    hyper_json TEXT,
    seed       INTEGER,
    trained_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 7) 实验运行：config 与 metrics 全落库，论文里每张表都能回溯到一次 run
CREATE TABLE IF NOT EXISTS experiment_run (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    config_json  TEXT,
    metrics_json TEXT,
    git_commit   TEXT,
    policy_id    INTEGER REFERENCES defense_policy(id),
    started_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at  TEXT
);

-- 8) 检测日志：哈希链，防篡改审计（prev_hash 自引用形成一条链）
CREATE TABLE IF NOT EXISTS detection_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version_id  INTEGER NOT NULL REFERENCES model_version(id),
    experiment_run_id INTEGER REFERENCES experiment_run(id),
    input_kind        TEXT    NOT NULL,       -- sample / attack_sample / live
    input_ref         INTEGER,
    text_sha256       TEXT    NOT NULL,
    score             REAL    NOT NULL,
    pred              TEXT    NOT NULL,
    latency_ms        REAL,
    prev_id           INTEGER REFERENCES detection_log(id),
    prev_hash         TEXT,
    cur_hash          TEXT    NOT NULL,
    ts                TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detlog_model ON detection_log(model_version_id);
