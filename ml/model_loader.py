import joblib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class ModelBundle:
    def __init__(self, cls_path: str, reg_path: Optional[str] = None):
        self.cls_path = cls_path
        self.reg_path = reg_path
        self.cls = None
        self.reg = None
        self.loaded = False

    def load(self):
        if Path(self.cls_path).exists():
            try:
                self.cls = joblib.load(self.cls_path)
                logger.info(f"[MODEL] Classification model loaded: {self.cls_path}")
            except Exception as e:
                logger.error(f"[MODEL] Classification load error: {e}")
        else:
            logger.warning(f"[MODEL] Classification model not found: {self.cls_path}")

        if self.reg_path:
            if Path(self.reg_path).exists():
                try:
                    self.reg = joblib.load(self.reg_path)
                    logger.info(f"[MODEL] Regression model loaded: {self.reg_path}")
                except Exception as e:
                    logger.error(f"[MODEL] Regression load error: {e}")
            else:
                logger.warning(f"[MODEL] Regression model not found: {self.reg_path}")

        self.loaded = self.cls is not None

    def predict_proba(self, feature_df):
        if not self.cls:
            return 0.55
        try:
            return float(self.cls.predict_proba(feature_df)[:, 1][0])
        except Exception as e:
            logger.error(f"[MODEL] predict_proba error: {e}")
            return 0.55

    def predict_reg(self, feature_df):
        if not self.reg:
            return None
        try:
            return float(self.reg.predict(feature_df)[0])
        except Exception as e:
            logger.error(f"[MODEL] regression predict error: {e}")
            return None
