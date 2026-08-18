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

**结论先行：** `shadowhook_interceptor_caller` 函数本身（GP 寄存器上下文的重建、无锁链表遍历的内存序、`next_hop` 的选择）是**正确**的。真正的问题出在它所处的“执行流程契约”上——尤其是 FP/SIMD 寄存器的处理。下面按严重程度排列。

---

## 2. 发现的问题

### P0 — 默认 flags 下，目标函数的 FP/SIMD 参数寄存器被静默破坏（最严重）

**位置：** `arch/arm64/sh_glue.S:112 / :150`（`tbnz ...,#0 / #1`），`arch/arm/sh_glue.S` 的 `vfpv3d16/d32` 变体，以及 `sh_glue.S:146` 处 `bl shadowhook_interceptor_caller`。

**机制：**
glue 只有在 `intercept_flags_union` 对应位置位时才会保存/恢复向量寄存器：
- READ 位（bit0）→ 调用 interceptor **之前** 保存 `q0–q31`；
- WRITE 位（bit1）→ 调用 interceptor **之后** 从 `cpu_context` 恢复 `q0–q31`。

而 `bl shadowhook_interceptor_caller` 是一次标准 AAPCS 调用，`shadowhook_interceptor_caller` 内部又会调用**用户任意的** `interceptor->pre(...)`。按 AAPCS64，被调用方可以自由破坏 **caller-saved** 的向量寄存器 `v0–v7`、`v16–v31`（只有 `d8–d15` 是 callee-saved 会被保留）。

关键点在于：`v0–v7` 正是**浮点/向量函数参数**的传递寄存器。于是：

| flags | 保存 v0–v31 | 恢复 v0–v31 | 目标函数拿到的 fp 参数 |
|---|---|---|---|
| DEFAULT (0) | 否 | 否 | **被破坏**（pre 的残留值） |
| READ_ONLY (1) | 是 | 否 | **被破坏**（保存只写进 cpu_context，没写回寄存器） |
| WRITE_ONLY (2) | 否 | 是 | **被破坏**（把未初始化的 cpu_context.vregs 写回 → 垃圾） |
| READ_WRITE (3) | 是 | 是 | 正确保留 |

也就是说：**只有 `SHADOWHOOK_INTERCEPT_WITH_FPSIMD_READ_WRITE` 才能让 fp 参数透传**。GP 参数（`x0–x7`）不受影响，因为 glue 无条件保存并恢复 `x0–x30`——这个**不对称**正是坑点所在。

**触发示例：** 对任何带浮点/向量参数的函数下 interceptor（哪怕拦截器函数体什么都不做）：
```c
// 目标：void mix(float gain, int* buf)  —— gain 在 v0/s0，buf 在 x0
shadowhook_intercept_sym_addr(mix, my_pre, NULL, SHADOWHOOK_INTERCEPT_DEFAULT, ...);
// 运行时 mix() 收到的 gain 变成垃圾值。改成 ..._WITH_FPSIMD_READ_WRITE 才正常。
```

**为什么算 bug 而不只是“性能取舍”：** 文档（`doc/manual.md:1126`）只说默认下“fpsimd 寄存器的值是随机的，且修改无效（modifications ineffective）”。“修改无效”这句会让使用者误以为“不碰 fpsimd，则 fpsimd 原样透传”。但实现的实际行为是 fpsimd 的 caller-saved 子集被**销毁**。拦截器契约是“`pre` 返回后被拦截指令继续执行”（即透明前置回调），而这个透明性对 fp 参数并不成立，文档也未告警。这对任何拦截“带浮点参数函数”的使用者都是静默的正确性 bug。

**修复方向（任选其一）：**
- 让 glue **无条件保存并恢复** `v0–v7`（fp 实参寄存器）——代价是每次拦截多几条 `stp/ldp`，但保证透明；READ/WRITE flags 只再额外决定 `v8–v31`/fpsr/fpcr 是否暴露给拦截器；
- 或在文档中明确：“拦截可能收到浮点/向量参数的函数，必须使用 `FPSIMD_READ_WRITE`”，并说明 READ_ONLY/WRITE_ONLY 同样会破坏参数。

---

### P1 — FPSIMD flags 实际是“每地址”生效（union），而非“每拦截器”

**位置：** `sh_switch.c:423 / :439`（`self->intercept_flags_union |= flags`），glue 读取 `intercept_flags_union`（结构体偏移 0）。

同一个 target 上的多个 interceptor 共享**一次** glue 调用，是否保存/恢复向量寄存器由所有拦截器 flags 的**并集** `intercept_flags_union` 决定，而它们又共享**同一个** `cpu_context`。后果：

- 拦截器 A 用 DEFAULT，拦截器 B 用 READ_WRITE，同挂一个地址 → A 也会看到“真实”的 vregs（而非文档承诺的随机值），且 **A 对 `cpu_context->vregs` 的写会因为 B 的 WRITE 位而被真正写回物理寄存器**。
- 即某个拦截器的可见/可写 fpsimd 行为，会被**同地址的其他拦截器**的 flags 悄悄改变。

单看 API（flags 是每次 `intercept_*` 调用传入的）会认为 flags 是 per-interceptor 语义，实现却是 per-address。实际危害通常有限（“行为良好”的 DEFAULT 拦截器不会去写 vregs），但这是一个真实的语义不一致，且与 P0 的修复取向相关。

---

### P2 — `intercept_flags_union` 存在数据竞争（data race）

**位置：** 写侧 `sh_switch.c:423/439/472`（普通 `|=` 和赋值，持 `sh_ref_lock`）；读侧 glue 中 `ldr IP_1,[IP_1]`（无锁快路径）。

glue 在无锁快路径里读 `intercept_flags_union`，而 add/unintercept 在持锁时用非原子 `|=` / 赋值修改它。二者构成对同一 `size_t` 的数据竞争（技术上是 UB）。在 arm/arm64 上对齐字的读写不会撕裂，读到的要么是旧值要么是新值，都是合法 flag 集合，因此实际后果只是“某一次调用用了略微陈旧的 flag 决策”，基本无害。但严格讲应改为 `__atomic_load/__atomic_or_fetch`（RELAXED 即可），与本文件中其它字段（`proxy_addr`、`enabled` 等都用了 `__atomic_*`）保持一致。

---

### P3 — 无锁移除竞争：`unintercept` 后 `pre` 可能再触发一次，`data` 生命周期需谨慎

**位置：** `sh_switch_interceptor_del`（`sh_switch.c:453`，仅置 `enabled=false`，不摘链），`shadowhook_interceptor_caller` 读 `enabled` 用 `__ATOMIC_RELAXED`。

拦截器移除是无锁软删除：只把 `enabled` 置 false，节点保留（真正 free 在 15s 延迟销毁时）。而 caller 侧对 `enabled` 是 RELAXED 读，与 del 的 RELEASE 写之间没有严格 happens-before。因此存在这样的窗口：一个已在 glue 内飞行的线程，刚 `unintercept` 完成后仍可能读到旧的 `enabled==true`，从而**再调用一次** `pre(cpu_context, data)`。

节点本身 15s 内不会释放，所以不是 shadowhook 内部的 UAF；但如果调用方在 `shadowhook_unintercept` 返回后**立即释放了 `data`**，这一次残留调用会拿到已释放的 `data` → 用户侧 UAF。这属于无锁 hook 库的固有语义，但文档未明确“unintercept 返回不代表 `pre` 不会再被调用”，建议补充说明（`data` 应延迟释放或采用引用计数）。

---

### P4 — ARM 上 `cpsr` 的 T 位 / IT 状态不反映目标指令集（次要）

**位置：** `arch/arm/sh_glue.S` `m_prolog` 的 `mrs IP_0, cpsr`。

glue 本体是 `.arm` 代码，`mrs cpsr` 抓到的 T 位恒为 0（ARM 态），即使被拦截的目标是 Thumb 函数。条件标志（NZCV）在整条分支链里不被改动，是正确的；但 `cpu_context->cpsr` 的 T 位、以及 Thumb IT-block 的 ITSTATE 位不代表目标现场。若拦截器依赖 `cpsr.T` 判断指令集，会得到错误结论（此时应改用“目标地址 bit0”判断，但 `regs[15]` 又已被 `CLEAR_BIT0` 抹掉）。属于边角限制，影响面小，建议文档注明。

---

### P5 — 拦截器执行顺序是 LIFO（后注册先执行）（行为提示）

**位置：** `sh_switch_interceptor_add`（`sh_switch.c:445`）头插链表；caller 从表头遍历。

多个拦截器的运行顺序是**注册的逆序**（后加的先跑）。文档只说“按顺序依次执行”（`manual.md:1124`），未明确方向；使用者可能默认是 FIFO。录制顺序（`sh_switch_record_proxy_and_interceptor`）与执行顺序一致，所以内部自洽，但对外语义建议写清楚。

---

## 3. 确认正确、设计良好的部分

为避免误导，以下几处经核对是**正确**的，不是 bug：

- **GP 上下文重建：** glue 在 default/CORRUPT_IP_REGS 两种配置下，都能正确还原被 trampoline 借用的 `x0/x1`（或 `x16/x17`）原值再写入 `cpu_context`，`x0–x30`/sp/pstate 完整保存与恢复。
- **`next_hop` 选择：** `proxy_addr!=0 ? proxy_addr : resume_addr`，且 `resume_addr` 在首次 hook 安装（重定向目标之前）就已 RELEASE 写入，不存在“读到 0”的窗口。
- **无锁链表发布：** 新节点字段先写、再对表头做 RELEASE 存储，读者 ACQUIRE 读表头；已存在节点的 `next` 永不改动（只头插），遍历安全。
- **内存安全的延迟释放：** enter/glue_launcher trampoline 经 `sh_trampo` 以时间戳标记延迟复用（enter 15s），switch 销毁也延迟 15s，缓解“线程仍在 trampoline 中执行时被释放”的经典 UAF。
- **`is_proc_start==false` 的 x16/x17 透传：** glue 把（可能被拦截器改写的）`x16/x17` 存回红区，enter trampoline 的 `ldp x16,x17,[sp,#-0x10]` 再取回，修改可生效。
- **ARM 忽略 `is_to_interceptor`（`sh_inst.c` 的 `(void)is_to_interceptor`）：** 并非 bug。ARM 在 glue_launcher 里 `str IP_0,[sp,#-4]` 统一保存 IP，故出口桩无需区分；arm64 才需要在 `shadow_exit` 出口桩里 `stp x16,x17` 区分拦截器/普通 hook 出口。
- **栈对齐：** arm64 `sub sp,#0x340` 保持 16 字节对齐；arm 在 `blx` 处经多次 push 累加后恰好 8 字节对齐。

---

## 4. 建议优先级

1. **P0（应修）**：无条件保留 fp 实参寄存器（至少 `v0–v7` / `s0–s15`），或在文档中显著警告“带浮点/向量参数的函数必须用 `FPSIMD_READ_WRITE`，READ_ONLY/WRITE_ONLY 同样会破坏参数”。
2. **P1（应厘清）**：明确 FPSIMD flags 是 per-address 语义，或改为逐拦截器分别决定其可见/可写范围。
3. **P2/P3（建议修）**：`intercept_flags_union` 改原子访问；文档补充 `unintercept` 后 `pre` 可能残留触发、`data` 需延迟释放。
4. **P4/P5（文档）**：注明 ARM cpsr T/IT 限制与拦截器 LIFO 执行顺序。
