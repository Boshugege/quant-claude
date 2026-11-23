# -*- coding: utf-8 -*-
"""本地模拟的 bigmodule：提交代码里的自测单元（M.jyc_eval._latest）不会报错。

评分机自己调用 main 并评分，自测单元会被跳过；如果代码在别处直接调用 M.jyc_eval，这里只记录收到的因子数据，
返回 None，真正的评分以评分机报告为准。
"""
CAPTURED = []


class _JycEval(object):
    def _latest(self, factor_data=None, show=False, **kwargs):
        CAPTURED.append(factor_data)
        print("[bigmodule] 已截获 M.jyc_eval 调用（本地模拟，评分以评分机报告为准）")
        return None

    def __call__(self, *a, **kw):
        return self._latest(*a, **kw)


class _M(object):
    jyc_eval = _JycEval()


M = _M()
