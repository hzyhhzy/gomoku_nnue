# ai主动写的，我也懒得删了，不保证准确


# NNUEBoardHistory - 替代NNUEBoard的功能

## 概述

`NNUEBoardHistory` 类已成功创建，用于替代 `NNUEBoard` 的功能，并继承自 `BoardHistory` 类。该类集成了NNUE神经网络评估功能，同时维护了完整的棋盘历史记录。

## 主要功能

### 1. 继承BoardHistory功能
- 继承了 `BoardHistory` 的所有功能
- 保持了游戏历史、规则管理等核心功能
- 兼容现有的 `BoardHistory` 接口

### 2. NNUE神经网络评估
- 集成了两个 `Eva_nnuev2` 评估器（黑白双方）
- 提供神经网络评估方法：
  - `evaluateFull()` - 完整评估（价值+策略）
  - `evaluatePolicy()` - 策略评估
  - `evaluateValue()` - 价值评估

### 3. 历史棋盘记录
- **核心功能**：维护 `std::vector<Board> historicalBoards` 记录所有历史棋盘状态
- 每次下棋时自动保存当前棋盘状态
- 支持高效的撤销操作

### 4. NNUEBoard功能替代
- `play(Color color, Loc loc)` - 下棋方法
- `undo(Color color, Loc loc)` - 撤销方法
- `clearCache(Color color)` - 清除缓存
- `updateInputBuf(Color nextPlayer)` - 更新输入缓冲区

## 类结构

```cpp
class NNUEBoardHistory : public BoardHistory {
public:
    // NNUE评估器
    Eva_nnuev2 blackEvaluator;
    Eva_nnuev2 whiteEvaluator;
    
    // 神经网络输入缓冲区
    float gfInputBuf[NNUEV2::globalFeatureNum];
    bool illegalMapBuf[MaxBS * MaxBS];
    
    // 历史棋盘记录 - 核心功能
    std::vector<Board> historicalBoards;
    
    // 移动缓存用于高效撤销
    struct MoveCache {
        bool isUndo;
        Color color;
        Loc loc;
    };
    
    MoveCache moveCacheB[MaxBS * MaxBS], moveCacheW[MaxBS * MaxBS];
    int moveCacheBlength, moveCacheWlength;
};
```

## 主要方法

### 构造函数
- `NNUEBoardHistory()` - 默认构造函数
- `NNUEBoardHistory(const ModelWeight* weights)` - 带权重的构造函数
- `NNUEBoardHistory(const Board& board, Player pla, const Rules& rules, const ModelWeight* weights)` - 完整构造函数

### 核心方法
- `clear(const Board& board, Player pla, const Rules& rules)` - 清除并重置
- `play(Color color, Loc loc)` - 下棋并更新历史
- `undo(Color color, Loc loc)` - 撤销并恢复历史
- `evaluateFull/Policy/Value()` - 神经网络评估

## 历史记录机制

1. **初始化**：`clear()` 方法会将初始棋盘状态添加到 `historicalBoards`
2. **下棋**：`play()` 方法会：
   - 复制当前棋盘状态
   - 在副本上执行移动
   - 将新状态添加到历史记录
3. **撤销**：`undo()` 方法会：
   - 从历史记录中移除最后一个状态
   - 恢复到前一个状态

## 与NNUEBoard的对比

| 功能 | NNUEBoard | NNUEBoardHistory |
|------|-----------|------------------|
| 神经网络评估 | ✓ | ✓ |
| 移动缓存 | ✓ | ✓ |
| 游戏历史管理 | ✗ | ✓ (继承自BoardHistory) |
| 棋盘历史记录 | ✗ | ✓ (historicalBoards) |
| 高效撤销 | 有限 | ✓ (基于历史记录) |
| 规则管理 | 基础 | ✓ (完整支持) |

## 使用示例

```cpp
// 创建NNUEBoardHistory实例
NNUEBoardHistory hist(weights);

// 初始化
Board board(15, 15);
Rules rules;
hist.clear(board, P_BLACK, rules);

// 下棋
hist.play(C_BLACK, centerLoc);
hist.play(C_WHITE, anotherLoc);

// 检查历史记录
cout << "历史棋盘数量: " << hist.historicalBoards.size() << endl;

// 撤销
hist.undo(C_WHITE, anotherLoc);

// 神经网络评估
NNUE::PolicyType policy[MaxBS * MaxBS];
NNUE::ValueType value = hist.evaluateFull(C_BLACK, policy);
```

## 文件结构

- `nnue/NNUEBoardHistory.h` - 头文件定义
- `nnue/NNUEBoardHistory.cpp` - 实现文件
- `tests/testNNUEBoardHistory.cpp` - 测试文件
- `test_nnue_board_history_simple.cpp` - 简单测试程序

## 总结

`NNUEBoardHistory` 成功实现了以下目标：

1. ✅ **替代NNUEBoard功能** - 提供了所有NNUEBoard的核心功能
2. ✅ **维护历史棋盘** - 使用 `vector<Board>` 记录所有历史状态
3. ✅ **高效撤销操作** - 基于历史记录的快速撤销
4. ✅ **继承BoardHistory** - 保持了完整的游戏历史管理功能
5. ✅ **神经网络集成** - 完整的NNUE评估功能

该类现在可以作为NNUEBoard的完全替代品使用，并提供了更强大的历史管理和撤销功能。