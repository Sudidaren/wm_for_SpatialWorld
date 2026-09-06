# LightWM 数据盘点报告

> 生成时间：本地运行；frame_index 共 72484 帧，fd_index 共 3618 帧，合计 76102 帧。

## 各数据池总览

| 池 | 场景数 | episode 数 | 帧数 | 带 visible 帧数 |
|---|---|---|---|---|
| lightwm_data | 89 | 1027 | 33325 | 30066 |
| cov | 12 | 12 | 5316 | 4794 |
| objviews | 88 | 1151 | 25719 | 25719 |
| procthor | 5 | 5 | 7224 | 7224 |
| virtualhome | 10 | 15 | 900 | 870 |
| fd | 81 | 309 | 3618 | 0 |

## 轨迹类型分布（帧数）

| 池 | 类型 | 帧数 | episode 数 |
|---|---|---|---|
| lightwm_data | target | 12015 | 89 |
| lightwm_data | action | 5311 | 89 |
| lightwm_data | task | 4165 | 582 |
| lightwm_data | sweep | 4005 | 89 |
| lightwm_data | focus | 4005 | 89 |
| lightwm_data | random | 3824 | 89 |
| cov | coverage | 5316 | 12 |
| objviews | objviews | 25719 | 1151 |
| procthor | procthor_coverage | 7224 | 5 |
| virtualhome | virtualhome_coverage | 900 | 15 |

## AI2-THOR 场景族分布（FloorPlan 场景数）

| 族 | lightwm_data | cov | objviews |
|---|---|---|---|
| FloorPlan 1-30 | 30 | 12 | 29 |
| FloorPlan 201-230 | 17 | 0 | 17 |
| FloorPlan 301-330 | 26 | 0 | 26 |
| FloorPlan 401-430 | 16 | 0 | 16 |
| 合计(去重) | 89 | 12 | 88 |

## 稀有类覆盖（全局 visible 出现次数最低的 14 类，按池分）

| 类别 | 全局 | lightwm_data | cov | objviews | procthor | virtualhome |
|---|---|---|---|---|---|---|
| AluminumFoil | 61 | 42 | 0 | 19 | 0 | 0 |
| VacuumCleaner | 99 | 53 | 0 | 46 | 0 | 0 |
| TableTopDecor | 207 | 98 | 0 | 109 | 0 | 0 |
| RoomDecor | 312 | 186 | 0 | 126 | 0 | 0 |
| DogBed | 337 | 34 | 0 | 70 | 233 | 0 |
| Footstool | 380 | 156 | 0 | 224 | 0 | 0 |
| Dumbbell | 381 | 84 | 0 | 297 | 0 | 0 |
| Desktop | 384 | 157 | 0 | 227 | 0 | 0 |
| BathtubBasin | 445 | 112 | 0 | 333 | 0 | 0 |
| LaundryHamper | 575 | 327 | 0 | 245 | 3 | 0 |
| Boots | 607 | 310 | 0 | 297 | 0 | 0 |
| Ottoman | 666 | 266 | 0 | 195 | 205 | 0 |
| WateringCan | 760 | 437 | 0 | 323 | 0 | 0 |
| GarbageBag | 777 | 272 | 0 | 346 | 159 | 0 |

## 词表外类型（未计入 117 类覆盖；多为结构件或未映射物体）

| 池 | 类型 | 出现次数 |
|---|---|---|
| procthor | wall | 27745 |
| procthor | room | 7145 |
| procthor | door | 5741 |
| procthor | window | 3544 |
| procthor | ClothesDryer | 105 |
| procthor | TableTop | 94 |

## 本地文件完整性

- 检查执行：False
- 缺失 rgb 帧数：0
