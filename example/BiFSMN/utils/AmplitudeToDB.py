import paddle
import math
import paddle.nn.functional as F
from typing import Optional

class AmplitudeToDB(paddle.nn.Layer):
    def __init__(self, stype: str = 'power', top_db: Optional[float] = None) -> None:
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = top_db
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        amin = paddle.to_tensor(self.amin)
        x = paddle.maximum(x, amin)
        
        db = self.multiplier * paddle.log10(x / self.ref_value)
        
        if self.top_db is not None:
            db = paddle.minimum(db, self.top_db)
        
        return db