from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import mlflow
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
import json
import re
from pathlib import Path
from src.datascience.components.configurations import ConfigurationManager
from src.datascience.entity.config_entity import DataCategorisationConfig
from src.datascience import logger






class DataCategorisation:
    def __init__(self):
        pass

    def _init(self, data_categorisation_config:DataCategorisationConfig):
        self.data_categorisation_config = data_categorisation_config

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        enc = SentenceTransformer(self.data_categorisation_config.model_name)
        E = enc.encode(texts, batch_size=self.data_categorisation_config.batch_size, show_progress_bar=False, normalize_embeddings=self.data_categorisation_config.normalise)
        return np.asarray(E, dtype=np.float32)

    
    def _ctfidf_top_terms(self, texts_by_cluster, top_n=6, ngram=(1,3)):
        docs = [" ".join(t) for t in texts_by_cluster]
        cv = CountVectorizer(ngram_range=ngram, min_df=1)
        X = cv.fit_transform(docs)
        idf = TfidfTransformer(use_idf=True, norm=None).fit(X).idf_
        ctfidf = X.multiply(idf).tocsr()
        tokens = np.array(cv.get_feature_names_out())
        out = []
        for i in range(ctfidf.shape[0]):
            row = ctfidf[i].toarray().ravel()
            idx = row.argsort()[-top_n:][::-1]
            out.append(tokens[idx].tolist())
        return out    
    
    
    def _l2(self, X: np.ndarray):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    
    def _load_seeds(self) -> dict:
        js_path = Path(self.data_categorisation_config.seeds)
        text = js_path.read_text(encoding="utf-8-sig")
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fixed = re.sub(r'([\]\}])\s*(")', r'\1,\n\2', text)

        try:
            data = json.loads(fixed)
            p_fixed = js_path.with_suffix(".fixed.json")
            p_fixed.write_text(fixed, encoding="utf-8")
            print(f"[fixed] wrote {p_fixed}")
            return data
        except json.JSONDecodeError as e:
            logger.exception(e)
            raise e
    
    def _make_seed_centroids(self, seeds: dict):
        items = {k: v for k, v in seeds.items()}
        labels = list(items.keys())
        centroids = []
        for lbl in labels:
            phrases = items[lbl]
            E_seed = self._embed_texts(phrases)
            centroids.append(E_seed.mean(axis=0, keepdims=False))
        C = self._l2(np.vstack(centroids))
        return C, labels

    def _assign_to_seeds(self, texts: list[str], C: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            E = self._embed_texts(texts)
            E = self._l2(E)
            S = E @ C.T
            tau = self.data_categorisation_config.tau
            idx = S.argmax(axis=1)
            best_sim = S[np.arange(len(E)), idx]
            chosen = np.array(labels, dtype=object)[idx]
            chosen = np.where(best_sim >= tau, chosen, "Other")
            return chosen, best_sim, S

        except Exception as e:
            logger.exception(e)
            raise e
            

    def initiate_data_categorisation(self):
        try:
            logger.info("\n<<<<<< Data Categorisation Initiated >>>>>>\n")
            config = ConfigurationManager()
            self.data_categorisation_config = config.get_data_categorisation_config()
            tau = self.data_categorisation_config.tau

            mlflow.set_experiment(self.data_categorisation_config.experiment)
            with mlflow.start_run(run_name=f"{self.data_categorisation_config.run_name}__seed-centroids"):
                mlflow.log_param("mode", "seed-centroids")
                mlflow.log_param("model_name", self.data_categorisation_config.model_name)
                mlflow.log_param("tau", tau)
                mlflow.log_param("seeds_path", str(self.data_categorisation_config.seeds))

                df = pd.read_csv(self.data_categorisation_config.source)
                if "Item Name" not in df.columns:
                    raise ValueError("Missing 'Item Name' column in source data.")
                texts = (df["Item Name"].fillna("")
                        .astype(str).str.strip().str.replace(r"\s+", " ", regex=True).tolist())

                seeds = self._load_seeds()
                mlflow.log_param("n_predef_clusters", int(len(seeds)))
                C, labels = self._make_seed_centroids(seeds)

                chosen, best_sim, _ = self._assign_to_seeds(texts, C, labels)

                label_to_id = {lbl: i for i, lbl in enumerate(labels)}
                cluster_id = np.array([label_to_id.get(lbl, -1) for lbl in chosen], dtype=int)

                keep_cols = [c for c in ("Item ID", "Item Name") if c in df.columns]
                if "Item ID" not in keep_cols:
                    df = df.reset_index().rename(columns={"index": "Item ID"})
                    keep_cols = [c for c in ("Item ID", "Item Name") if c in df.columns]
                clusters_df = df[keep_cols].copy()
                clusters_df["cluster_id"] = cluster_id
                clusters_df["confidence"] = best_sim.clip(0, 1)

                p_clusters = Path(self.data_categorisation_config.root_dir,"clusters.csv")
                clusters_df.to_csv(p_clusters, index=False)
                mlflow.log_artifact(str(p_clusters))

                rows, texts_by_c = [], []
                for lbl, cid in label_to_id.items():
                    mask = (cluster_id == cid)
                    size = int(mask.sum())
                    if size > 0:
                        rows.append({"cluster_id": cid, "label": lbl, "size": size})
                        texts_by_c.append(list(pd.Series(texts)[mask]))

                tops = self._ctfidf_top_terms(texts_by_c, top_n=10, ngram=(1,3)) if texts_by_c else []
                for r, terms in zip(rows, tops):
                    r["top_terms"] = terms

                labels_df = pd.DataFrame(rows).sort_values("size", ascending=False)
                p_labels = Path(self.data_categorisation_config.root_dir, "cluster_labels.csv")
                labels_df.to_csv(p_labels, index=False)
                mlflow.log_artifact(str(p_labels))

                coverage = float((cluster_id != -1).mean())
                mlflow.log_metric("coverage_non_other", coverage)
                mlflow.log_metric("n_clusters_effective", int((labels_df["size"] > 0).sum()))

                logger.info(f"[SEED-ASSIGN] τ={tau:.2f}, coverage={coverage:.1%}")
                logger.info(f"Saved: {p_clusters}, {p_labels}")
                logger.info("\n<<<<<< Data Categorisation Completed >>>>>>\n")
                return p_clusters, p_labels

        except Exception as e:
            logger.exception(e)
            raise e