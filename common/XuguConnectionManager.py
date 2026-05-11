import xgcondb
from core.config import cfg

class XuguConnectionManager:
    def __init__(self, config: dict = None):
        # 如果传入了 config，使用传入的；否则使用 .env 中的配置
        db_config = config if config else cfg.xg_db_info
        self.conn = xgcondb.connect(**db_config)
        self.closed = False

    def get_conn(self):
        return self.conn

    def close(self):
        self.conn.commit()
        self.conn.close()
        self.closed = True

    def execute(self, sql):
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            self.conn.commit()
            res = cur.fetchall()
            return res
        except Exception as e:
            self.conn.rollback()
            raise Exception(f'执行sql失败！: {e}')
        finally:
            cur.close()

    async def async_execute(self, sql):
        cur = await self.conn.cursor()
        try:
            await cur.execute(sql)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise Exception(f'执行sql失败！: {e}')
        finally:
            cur.close()

    def execute_list(self, sql_list):
        '''
        批量sql操作，一次性执行多条sql，再次进行插入
        :param sql_list:
        :return:
        '''
        cur = self.conn.cursor()
        try:
            data_list = []
            for sql in sql_list:
                cur.execute(sql)
                self.conn.commit()
                res = cur.fetchall()
                # data_list.append(res)
                if cur.description:  # 如果有列信息（查询语句才有）
                    # 获取所有字段名
                    columns = [col[0] for col in cur.description]
                    # 转成 [ {字段名:值}, {...} ]
                    res_dict = [dict(zip(columns, row)) for row in res]
                    data_list.append(res_dict)
                else:
                    # 没有列的情况（插入/更新/删除），直接返回原结果
                    data_list.append(res)
            return data_list
        except Exception as e:
            self.conn.rollback()
            raise Exception(f'执行sql失败！: {e}')
        finally:
            cur.close()

    def execute_many(self, sql, rows):
        cur = self.conn.cursor()
        try:
            cur.executemany(sql, rows)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise Exception(f'执行sql失败！: {e}')
        finally:
            cur.close()

    def execute_param(self, sql, params=None):
        """
        执行单条SQL（支持参数化查询，解决无引号+SQL注入问题）
        :param sql: SQL语句模板（支持占位符，如 %s 或 :1，取决于xgcondb适配）
        :param params: SQL查询参数（元组/列表），可选，默认None
        :return: 查询结果（二维列表）
        """
        if self.closed:
            raise Exception("数据库连接已关闭，无法执行SQL")

        cur = self.conn.cursor()
        try:
            # 核心改造：支持参数化查询，自动处理参数引号和转义
            if params is not None and isinstance(params, (tuple, list)):
                cur.execute(sql, params)
            else:
                cur.execute(sql)

            self.conn.commit()
            res = cur.fetchall()
            return res
        except Exception as e:
            self.conn.rollback()
            raise Exception(f'执行sql失败！: {e}')
        finally:
            cur.close()

    def create(self, create_sql):
        self.execute(create_sql)

    def insert(self, sql):
        self.execute(sql)

    def inserts(self, sql, val):
        cur = self.conn.cursor()
        try:
            cur.executemany(sql, val)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise Exception(f'插入数据失败！: {e}')
        finally:
            cur.close()

    def delete(self, sql):
        self.execute(sql)

    def update(self, sql):
        self.execute(sql)

    def select(self, sql):
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            col = cur.description
            results = cur.fetchall()
            return results, col
        except Exception as e:
            raise Exception(f'查询数据失败！: {e}')
        finally:
            cur.close()

    def show(self, table_name):
        show_sql = "show create table  {}".format(table_name)
        cur = self.conn.cursor()
        try:
            cur.execute(show_sql)
            results = cur.fetchall()
            return results
        except Exception as e:
            raise Exception(f'查询数据失败！: {e}')
        finally:
            cur.close()

if __name__ == '__main__':
    import config as mconfig
    import time
    for i in range(50):
        a = GbaseConnectionManager(mconfig.gbase_db_info)
        time.sleep(0.1)
        a.close()
        print(a)
