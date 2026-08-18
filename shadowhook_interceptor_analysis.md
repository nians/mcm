# ShadowHook Interceptor 执行流程深度分析

分析对象：`bytedance/android-inline-hook` 的 `shadowhook` 模块，重点是 interceptor（拦截器）的运行时执行链路，尤其是 `shadowhook_interceptor_caller`。

分析基于的提交：`0976d6f`（HEAD，`SHADOWHOOK_VERSION "2.0.1"`）。

涉及的核心文件：
- `shadowhook/src/main/cpp/sh_switch.c` —— `shadowhook_interceptor_caller` 及 switch 状态机
- `shadowhook/src/main/cpp/arch/arm64/sh_glue.S` / `arch/arm/sh_glue.S` —— 保存/恢复 CPU 上下文的汇编 glue
- `shadowhook/src/main/cpp/arch/arm64/sh_inst.c` / `arch/arm/sh_inst.c` —— glue launcher 与 enter/exit trampoline 构造
- `shadowhook/src/main/cpp/include/shadowhook.h` —— `shadowhook_cpu_context_t` 与 flags 定义

---

## 1. 执行流程回顾

一次 interceptor 触发的完整跳转链（以 arm64、函数首地址拦截为例）：

```
[目标函数首指令]  b shadow_exit
        │
        ▼
[shadow_exit @ELF gap]  stp x16,x17,[sp,#-0x10]  // 保存 IP 寄存器（interceptor 专用出口）
                        ldr x16,#8 ; br x16
        │
        ▼
[glue_launcher @mmap]   ldr x16, =shadowhook_interceptor_glue
                        ldr x17, =ctx(=sh_switch_t*)
                        br  x16
        │
        ▼
[shadowhook_interceptor_glue @.text (sh_glue.S)]
        · sub sp,#0x340，建立 fp-chain
        · 读取 sh_switch_t.intercept_flags_union（偏移 0）
        · 若 READ 位置位 → 保存 q0-q31/fpsr/fpcr
        · 保存 x0-x30、sp、pstate(NZCV) 进 shadowhook_cpu_context_t
        · bl shadowhook_interceptor_caller(ctx, &cpu_context, &next_hop)
        · 若 WRITE 位置位 → 从 cpu_context 恢复 q0-q31/fpsr/fpcr
        · 恢复 x0-x30、pstate；用 x16 装载 next_hop
        · br x16
        │
        ▼
[next_hop]  proxy_addr（有 hook 时）  或  resume_addr（仅拦截时，= enter trampoline）
```

`shadowhook_interceptor_caller`（`sh_switch.c:387`）本身逻辑很短：

```c
void shadowhook_interceptor_caller(void *ctx, shadowhook_cpu_context_t *cpu_context, void **next_hop) {
  sh_switch_t *self = (sh_switch_t *)ctx;
#if defined(__aarch64__)
  cpu_context->pc = self->target_addr;
#elif defined(__arm__)
  cpu_context->regs[15] = SH_UTIL_CLEAR_BIT0(self->target_addr);
#endif
  sh_switch_interceptor_t *interceptor;
  for (interceptor = __atomic_load_n(&SLIST_FIRST(&self->interceptors), __ATOMIC_ACQUIRE); interceptor;
       interceptor = __atomic_load_n(&SLIST_NEXT(interceptor, link), __ATOMIC_RELAXED)) {
    if (__atomic_load_n(&interceptor->enabled, __ATOMIC_RELAXED))
      interceptor->pre(cpu_context, interceptor->data);
  }
  uintptr_t proxy_addr = __atomic_load_n(&self->proxy_addr, __ATOMIC_ACQUIRE);
  *next_hop = (void *)(0 != proxy_addr ? proxy_addr : __atomic_load_n(&self->resume_addr, __ATOMIC_RELAXED));
}
```

**结论先行：** `shadowhook_interceptor_caller` 的**控制流骨架**（GP 上下文重建、无锁链表遍历的内存序、`next_hop` 选择）是正确的。但它是一个**透明前置回调的调度器**——在里面调用了用户任意的 `pre()`，因此必须像信号处理函数一样，对被拦截现场的“旁路状态”保持透明。有两处透明性被破坏：**(P0) 线程局部 `errno` 未跨回调保存/恢复**、**(P1) 浮点/向量参数寄存器在默认 flags 下被破坏**。下面按严重程度排列。

---

## 2. 发现的问题

### P0 — `errno` 未跨拦截器回调保存/恢复（最严重，且是彻底的 bug）

**位置：** `shadowhook_interceptor_caller`（`sh_switch.c:387`），`interceptor->pre(...)` 调用循环的前后。

**机制：**
`errno` 在 bionic 里是 `(*__errno())`，是**线程局部（TLS）内存**，不是 CPU 寄存器。汇编 glue 只保存/恢复通用寄存器、向量寄存器和 `pstate`——**完全不触及 errno 这块 TLS**。而循环里的 `interceptor->pre(cpu_context, data)` 是**用户任意代码**，几乎必然会改写 `errno`（调用任何可能 set errno 的 libc、日志、`malloc`、syscall 包装等都会）。

interceptor 的契约是**透明前置回调**：文档明确“当拦截器函数返回时，被拦截的指令将继续执行”。它相当于往指令流里注入一段代码，语义上等同于**信号处理函数**——信号可能在“syscall 返回”和“检查 errno”之间到达，所以 signal handler 必须保存/恢复 errno；interceptor 完全同理。当前实现没有做这件事，于是：

> 用户的 `pre` 回调污染的 `errno` 会**泄漏进被拦截的原始代码**，原始代码（或其调用者）之后读到的 `errno` 就是错的。

**触发示例（非常容易命中）：**
```c
// 在某个函数中间某条指令上拦截；该函数返回失败时其调用者会读 errno
void my_pre(shadowhook_cpu_context_t *ctx, void *data) {
  __android_log_print(ANDROID_LOG_INFO, "tag", "hit");  // 内部可能改动 errno
}
// 原始函数正常置 errno=EAGAIN 后返回，但调用者读到的却是被 log 改过的 errno → 逻辑错乱
```
这个 bug 与目标函数是否用浮点无关、与 flags 无关、几乎任何“做点实事”的拦截器回调都会触发，而且极其隐蔽难查。

**与 hook 的区别（为什么只有 interceptor 需要修）：** hook 的 proxy 是函数的**完整替换**，errno 语义本就是用户 proxy 自己的责任；而 interceptor 是透明注入，原始代码仍在跑，基础设施必须保证 errno 透明。所以修复精确地只落在 `shadowhook_interceptor_caller`。

**修复（已验证的最小改动）：** 在调用回调前保存 `errno`、循环后恢复。

```diff
--- a/shadowhook/src/main/cpp/sh_switch.c
+++ b/shadowhook/src/main/cpp/sh_switch.c
@@ -23,6 +23,7 @@
 
 #include "sh_switch.h"
 
+#include <errno.h>
 #include <inttypes.h>
 #include <pthread.h>
@@ shadowhook_interceptor_caller @@
   cpu_context->regs[15] = SH_UTIL_CLEAR_BIT0(self->target_addr);
 #endif
 
+  // interceptor 是透明前置回调：pre() 返回后被拦截指令要继续执行。
+  // pre() 是用户任意代码，可能改动线程局部 errno，需在回调前后保存/恢复，
+  // 使拦截器对恢复执行的目标不可见（与信号处理函数必须保护 errno 同理）。
+  int saved_errno = errno;
+
   sh_switch_interceptor_t *interceptor;
   for (interceptor = __atomic_load_n(&SLIST_FIRST(&self->interceptors), __ATOMIC_ACQUIRE); interceptor;
        interceptor = __atomic_load_n(&SLIST_NEXT(interceptor, link), __ATOMIC_RELAXED)) {
     if (__atomic_load_n(&interceptor->enabled, __ATOMIC_RELAXED))
       interceptor->pre(cpu_context, interceptor->data);
   }
 
+  errno = saved_errno;
+
   uintptr_t proxy_addr = __atomic_load_n(&self->proxy_addr, __ATOMIC_ACQUIRE);
```

正确性要点：从被拦截点到 `shadowhook_interceptor_caller` 入口，中间只经过 `shadow_exit`/`glue_launcher`/`glue` 三段纯汇编（无 libc、无 syscall），因此入口处读到的 `errno` 就是目标现场的 `errno`；恢复之后 glue 也只有纯汇编再跳到 `next_hop`。故这对 save/restore 精确复原了目标的 errno。

> 同源问题：FPSR 里的 FP 累积异常标志 / QC 饱和位也是这种“旁路状态”，但它只在设置了 fpsimd flags 时才被保存/恢复（见 P1）。errno 则完全没有对应机制，因此更严重。

---

### P1 — 默认 flags 下，目标函数的 FP/SIMD 参数寄存器被静默破坏

**位置：** `arch/arm64/sh_glue.S:112 / :150`（`tbnz ...,#0 / #1`），`arch/arm/sh_glue.S` 的 `vfpv3d16/d32` 变体，以及 `sh_glue.S:146` 处 `bl shadowhook_interceptor_caller`。

**机制：**
glue 只有在 `intercept_flags_union` 对应位置位时才保存/恢复向量寄存器：READ 位（bit0）→ 调用前保存 `q0–q31`；WRITE 位（bit1）→ 调用后从 `cpu_context` 恢复 `q0–q31`。而 `bl shadowhook_interceptor_caller` 是标准 AAPCS 调用，内部又调用用户任意的 `pre(...)`。按 AAPCS64，被调方可自由破坏 **caller-saved** 的 `v0–v7`、`v16–v31`（只有 `d8–d15` 是 callee-saved）。关键点：`v0–v7` 正是**浮点/向量函数参数**的传递寄存器。

| flags | 保存 v0–v31 | 恢复 v0–v31 | 目标函数拿到的 fp 参数 |
|---|---|---|---|
| DEFAULT (0) | 否 | 否 | **被破坏** |
| READ_ONLY (1) | 是 | 否 | **被破坏**（只存进 cpu_context，没写回寄存器） |
| WRITE_ONLY (2) | 否 | 是 | **被破坏**（把未初始化的 cpu_context.vregs 写回 → 垃圾） |
| READ_WRITE (3) | 是 | 是 | 正确保留 |

即**只有 `FPSIMD_READ_WRITE` 能让 fp 参数透传**。GP 参数（`x0–x7`）不受影响，因为 glue 无条件保存并恢复 `x0–x30`——这个**不对称**正是坑点。文档（`doc/manual.md:1126`）只说默认下 fpsimd“修改无效”，会让人误以为“不碰 fpsimd 就原样透传”，但实际是 caller-saved 子集被销毁。

**修复方向：** 让 glue 无条件保存并恢复 `v0–v7`（fp 实参寄存器）；READ/WRITE flags 只再决定 `v8–v31`/fpsr/fpcr 是否暴露给拦截器。或在文档显著警告“带浮点/向量参数的函数必须用 `FPSIMD_READ_WRITE`，READ_ONLY/WRITE_ONLY 同样会破坏参数”。

---

### P2 — FPSIMD flags 实际是“每地址”生效（union），而非“每拦截器”

**位置：** `sh_switch.c:423 / :439`（`self->intercept_flags_union |= flags`），glue 读取 `intercept_flags_union`（结构体偏移 0）。

同一 target 上的多个 interceptor 共享一次 glue 调用，是否保存/恢复向量寄存器由所有 flags 的**并集**决定，而它们又共享同一个 `cpu_context`。后果：拦截器 A 用 DEFAULT、B 用 READ_WRITE 同挂一个地址时，A 也会看到“真实”的 vregs（而非文档承诺的随机值），且 **A 对 `cpu_context->vregs` 的写会因 B 的 WRITE 位被真正写回物理寄存器**。某拦截器的 fpsimd 可见/可写行为会被同地址其他拦截器的 flags 悄悄改变。API 上 flags 看似 per-interceptor，实现却是 per-address。实际危害通常有限，但属真实语义不一致。

---

### P3 — `intercept_flags_union` 存在数据竞争（data race）

**位置：** 写侧 `sh_switch.c:423/439/472`（非原子 `|=`/赋值，持 `sh_ref_lock`）；读侧 glue 中 `ldr IP_1,[IP_1]`（无锁快路径）。

glue 无锁读、写侧持锁但用非原子写，构成对同一 `size_t` 的数据竞争（技术上 UB）。arm/arm64 对齐字读写不撕裂，读到的要么旧值要么新值，都是合法 flag 集合，实际后果只是某次调用用了略陈旧的 flag 决策，基本无害。严格应改 `__atomic_load/__atomic_or_fetch`（RELAXED 即可），与本文件其它字段（`proxy_addr`、`enabled`）保持一致。

---

### P4 — 无锁移除竞争：`unintercept` 后 `pre` 可能再触发一次，`data` 生命周期需谨慎

**位置：** `sh_switch_interceptor_del`（`sh_switch.c:453`，仅置 `enabled=false` 不摘链），caller 读 `enabled` 用 `__ATOMIC_RELAXED`。

移除是无锁软删除，节点保留（真正 free 在 15s 延迟销毁）。caller 侧 `enabled` 是 RELAXED 读，与 del 的 RELEASE 写无严格 happens-before，因此已在 glue 内飞行的线程 `unintercept` 完成后仍可能再调用一次 `pre(cpu_context, data)`。节点 15s 内不释放，故非内部 UAF；但调用方若在 `shadowhook_unintercept` 返回后立即释放 `data`，这次残留调用会拿到已释放的 `data` → 用户侧 UAF。属无锁 hook 库固有语义，但文档未提示“unintercept 返回不代表 `pre` 不会再被调用”，建议补充（`data` 延迟释放或引用计数）。

---

### P5 — ARM 上 `cpsr` 的 T 位 / IT 状态不反映目标指令集（次要）

**位置：** `arch/arm/sh_glue.S` `m_prolog` 的 `mrs IP_0, cpsr`。

glue 本体是 `.arm` 代码，`mrs cpsr` 抓到的 T 位恒为 0，即使目标是 Thumb 函数。NZCV 是对的，但 `cpu_context->cpsr` 的 T 位、Thumb IT-block 的 ITSTATE 位不代表目标现场。若拦截器依赖 `cpsr.T` 判断指令集会得到错误结论（此时 `regs[15]` 又已被 `CLEAR_BIT0` 抹掉）。边角限制，建议文档注明。

---

### P6 — 拦截器执行顺序是 LIFO（后注册先执行）（行为提示）

**位置：** `sh_switch_interceptor_add`（`sh_switch.c:445`）头插链表；caller 从表头遍历。

多个拦截器运行顺序是注册的逆序。文档只说“按顺序依次执行”（`manual.md:1124`）未明方向，使用者可能默认 FIFO。录制顺序与执行顺序一致（内部自洽），但对外语义建议写清楚。

---

## 3. 确认正确、设计良好的部分

为避免误导，以下几处经核对是**正确**的，不是 bug：

- **GP 上下文重建：** default/CORRUPT_IP_REGS 两种配置下都能正确还原被 trampoline 借用的 `x0/x1`（或 `x16/x17`）原值再写入 `cpu_context`，`x0–x30`/sp/pstate 完整保存恢复。
- **`next_hop` 选择：** `proxy_addr!=0 ? proxy_addr : resume_addr`，且 `resume_addr` 在首次 hook 安装（重定向目标之前）就已 RELEASE 写入，不存在“读到 0”的窗口。
- **无锁链表发布：** 新节点字段先写、再对表头做 RELEASE 存储，读者 ACQUIRE 读表头；已存在节点的 `next` 永不改动（只头插），遍历安全。
- **内存安全的延迟释放：** enter/glue_launcher trampoline 经 `sh_trampo` 以时间戳延迟复用（enter 15s），switch 销毁也延迟 15s，缓解“线程仍在 trampoline 中执行时被释放”的经典 UAF。
- **`is_proc_start==false` 的 x16/x17 透传：** glue 把（可能被拦截器改写的）`x16/x17` 存回红区，enter trampoline 的 `ldp x16,x17,[sp,#-0x10]` 再取回，修改可生效。
- **ARM 忽略 `is_to_interceptor`（`sh_inst.c` 的 `(void)is_to_interceptor`）：** 并非 bug。ARM 在 glue_launcher 里 `str IP_0,[sp,#-4]` 统一保存 IP，出口桩无需区分；arm64 才需在 `shadow_exit` 出口桩里 `stp x16,x17` 区分拦截器/普通 hook 出口。
- **栈对齐：** arm64 `sub sp,#0x340` 保持 16 字节对齐；arm 在 `blx` 处经多次 push 累加后恰好 8 字节对齐。

---

## 4. 建议优先级

1. **P0（必修）**：`shadowhook_interceptor_caller` 在回调循环前后保存/恢复 `errno`（补丁见上，最小改动）。这是彻底的透明性 bug，与 flags 无关、几乎所有非平凡拦截器都会触发。
2. **P1（应修）**：无条件保留 fp 实参寄存器（至少 `v0–v7`/`s0–s15`），或在文档显著警告只有 `FPSIMD_READ_WRITE` 能透传 fp 参数。
3. **P2（应厘清）**：明确 FPSIMD flags 是 per-address 语义，或改为逐拦截器决定其可见/可写范围。
4. **P3/P4（建议修）**：`intercept_flags_union` 改原子访问；文档补充 `unintercept` 后 `pre` 可能残留触发、`data` 需延迟释放。
5. **P5/P6（文档）**：注明 ARM cpsr T/IT 限制与拦截器 LIFO 执行顺序。
