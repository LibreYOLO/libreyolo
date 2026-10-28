"""All CSS: brand tokens (dark slate+cyan / light cream), layout, components."""

PART = r"""  :root{
    /* LibreYOLO brand - dark (slate + cyan), matching libreyolo.com */
    --bg:#020617; --bg2:#0b1120;
    --s1:#0f172a; --s2:#1e293b; --s3:#334155;
    --line:#1e293b; --line2:#334155;
    --tx:#e2e8f0; --tx2:#94a3b8; --tx3:#64748b;
    --ac:#06b6d4; --ac-ink:#012a33; --ai:#22d3ee;
    --ok:#10b981; --warn:#fbbf24; --danger:#ef4444;
    --r:12px; --r2:9px; --sh:0 18px 44px -14px rgba(2,6,23,.62); --shs:0 2px 6px rgba(2,6,23,.42);
    --stage1:#0b1120; --stage2:#020617; --topbar1:#0f172a; --topbar2:#0b1120;
    --acg1:#22d3ee; --acg2:#0891b2; --glass:rgba(15,23,42,.85);
  }
  :root.light{
    /* LibreYOLO brand - warm light, inspired by the marketing carousels (cream + cyan) */
    --bg:#faf6ec; --bg2:#fffdf6;
    --s1:#fffdf7; --s2:#f4eedd; --s3:#e9e1cd;
    --line:#ece4d2; --line2:#dccfb4;
    --tx:#1c1917; --tx2:#57534e; --tx3:#8a8175;
    --ac:#0891b2; --ac-ink:#ffffff; --ai:#0e7490;
    --ok:#059669; --warn:#d97706; --danger:#dc2626;
    --sh:0 18px 44px -18px rgba(28,25,23,.18); --shs:0 1px 3px rgba(28,25,23,.08);
    --stage1:#f1ead8; --stage2:#e6dcc5; --topbar1:#fffdf7; --topbar2:#f7f1e2;
    --acg1:#22c3e0; --acg2:#0891b2; --glass:rgba(255,253,247,.85);
  }
  *{box-sizing:border-box}
  [hidden]{display:none !important}   /* author display rules must not undo the hidden attribute (wizard panes/footer) */
  html,body{margin:0;height:100%;background:var(--bg);color:var(--tx);
    font:13.5px/1.55 "Outfit",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
  button{font:inherit;color:inherit;cursor:pointer}
  .ic{width:16px;height:16px;display:block;flex:none}
  #app{display:grid;grid-template-rows:52px 1fr;height:100vh}
  /* topbar */
  .topbar{display:flex;align-items:center;gap:12px;padding:0 14px;
    background:var(--topbar1);border-bottom:1px solid var(--line)}
  .brand{display:flex;align-items:center;gap:8px;font-weight:650;letter-spacing:.2px}
  .brand .ic{width:21px;height:21px;color:var(--ac)}
  .brand b{color:var(--ac)}
  .topbar .sep{width:1px;height:20px;background:var(--line2)}
  .topbar .ds{color:var(--tx3);font-size:12px;max-width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .topbar .counter{color:var(--tx2);font-variant-numeric:tabular-nums;font-size:12.5px}
  .topbar .counter b{color:var(--tx)}
  .grow{flex:1}
  .btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;height:32px;padding:0 13px;
    border-radius:var(--r2);border:1px solid transparent;font-weight:560;transition:.15s;white-space:nowrap}
  .btn .ic{width:15px;height:15px}
  .btn-primary{background:var(--ac);color:var(--ac-ink);
    box-shadow:0 1px 2px rgba(0,0,0,.25)}
  .btn-primary:hover{background:var(--ai);transform:translateY(-1px)}
  .btn-primary:active{transform:translateY(0)}
  .btn-ghost{background:var(--s2);border-color:var(--line2);color:var(--tx)}
  .btn-ghost:hover{background:var(--s3)}
  .btn-sm{height:30px;padding:0 11px;font-size:12px}
  .btn-icon{display:grid;place-items:center;width:32px;height:32px;border-radius:var(--r2);
    background:transparent;border:1px solid transparent;color:var(--tx2);transition:.15s}
  .btn-icon:hover{background:var(--s2);color:var(--tx);border-color:var(--line)}
  .ai{display:flex;align-items:center;gap:7px;padding:4px 6px;border-radius:12px;
    background:color-mix(in srgb,var(--ac) 11%,transparent);border:1px solid color-mix(in srgb,var(--ac) 28%,transparent)}
  :root.light .ai{background:color-mix(in srgb,var(--ac) 9%,transparent);border-color:color-mix(in srgb,var(--ac) 30%,transparent)}
  .tgroup{display:inline-flex;align-items:center;gap:3px}
  .ai .field{display:flex;align-items:center;gap:8px;height:32px;padding:0 11px;border-radius:var(--r2);
    background:var(--s1);border:1px solid var(--line);color:var(--tx3);font-size:12px}
  .ai .field b{color:var(--tx);font-variant-numeric:tabular-nums;min-width:28px;text-align:right}
  .ai input[type=range]{-webkit-appearance:none;appearance:none;width:92px;height:4px;border-radius:9px;background:var(--s3);outline:none}
  .ai input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;
    background:var(--ai);cursor:pointer;box-shadow:0 0 0 3px color-mix(in srgb,var(--ac) 25%,transparent)}
  .select{height:32px;border-radius:var(--r2);background:var(--s1);color:var(--tx2);
    border:1px solid var(--line);padding:0 8px;font-size:12px;max-width:170px}
  .laprompt{height:32px;width:210px;border-radius:var(--r2);background:var(--s1);color:var(--tx);
    border:1px solid var(--line);padding:0 11px;font-size:12px;outline:none}
  .laprompt:focus{border-color:var(--ac)}
  .save{display:inline-flex;align-items:center;gap:7px;height:30px;padding:0 12px;border-radius:999px;
    border:1px solid var(--line);color:var(--tx3);font-size:12px;font-weight:540}
  .save::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
  .save.dirty{color:var(--warn);border-color:rgba(245,177,61,.35);background:rgba(245,177,61,.08)}
  .save.saved{color:var(--ok);border-color:rgba(45,212,167,.35);background:rgba(45,212,167,.08)}
  .save.fail{color:var(--danger);border-color:color-mix(in srgb,var(--danger) 40%,transparent);background:rgba(239,68,68,.08)}
  @keyframes pop{0%{transform:scale(1)}40%{transform:scale(1.14)}100%{transform:scale(1)}}
  .save.flash{animation:pop .42s ease-out}
  /* main */
  main{display:grid;grid-template-columns:300px 1fr 234px;min-height:0}   /* regions column always reserved: adding/removing a label never reflows or deforms the canvas */
  .regions{display:flex;flex-direction:column;min-height:0;background:var(--bg2);border-left:1px solid var(--line)}
  .rp-head{padding:13px 13px;border-bottom:1px solid var(--line);font-size:12px;font-weight:600;color:var(--tx2);display:flex;gap:7px;align-items:center}
  .rp-head b{color:var(--tx);font-variant-numeric:tabular-nums}
  .rp-list{flex:1;overflow-y:auto;padding:8px}
  .rprow{display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:8px;border:1px solid transparent;cursor:pointer;transition:.1s}
  .rprow:hover{background:var(--s2);border-color:var(--line)}
  .rprow.on{background:var(--s3);border-color:var(--ac)}
  .rprow .sw{width:12px;height:12px;border-radius:3px;flex:none}
  .rprow .rpn{flex:1;font-size:12.5px;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .rprow .rpi{font-size:11px;color:var(--tx3);font-variant-numeric:tabular-nums}
  .rprow .rpx{opacity:0;color:var(--tx3);font-size:16px;line-height:1;padding:0 3px;border-radius:5px}
  .rprow:hover .rpx{opacity:.55} .rprow .rpx:hover{opacity:1;color:var(--danger);background:var(--s1)}
  .rp-empty{padding:18px 12px;color:var(--tx3);font-size:12px;text-align:center}
  /* Responsive: the canvas keeps priority - side panels shrink as the window narrows,
     and the regions panel yields below ~820px so the image never becomes a sliver. */
  @media (max-width:1180px){ main{grid-template-columns:232px 1fr 200px} }
  @media (max-width:980px){ main{grid-template-columns:200px 1fr 184px} }
  @media (max-width:820px){ main{grid-template-columns:172px 1fr} .regions{display:none} }
  .sidebar{display:flex;flex-direction:column;min-height:0;background:var(--bg2);border-right:1px solid var(--line)}
  .side-head{padding:12px 12px 10px;border-bottom:1px solid var(--line)}
  .seg{display:flex;gap:2px;padding:3px;background:var(--s1);border:1px solid var(--line);border-radius:var(--r2)}
  .seg button{flex:1;height:28px;border:0;border-radius:6px;background:transparent;color:var(--tx3);font-size:12px;font-weight:560;transition:.12s}
  .seg button.on{background:var(--s3);color:var(--tx);box-shadow:var(--shs)}
  .seg button:hover:not(.on){color:var(--tx2)}
  .list{flex:1;overflow:auto;padding:8px;display:flex;flex-direction:column;gap:6px}
  .list::-webkit-scrollbar{width:11px}
  .list::-webkit-scrollbar-thumb{background:var(--s3);border-radius:9px;border:3px solid var(--bg2)}
  .card{display:flex;gap:10px;align-items:center;padding:7px;border-radius:var(--r2);
    background:var(--s1);border:1px solid transparent;text-align:left;transition:.12s;width:100%}
  .card:hover{background:var(--s2);border-color:var(--line)}
  .card.sel{background:var(--s2);border-color:var(--ac);box-shadow:0 0 0 1px var(--ac)}
  .card .thumb{width:54px;height:40px;border-radius:6px;object-fit:cover;background:var(--s3);flex:none}
  .card .meta{display:flex;flex-direction:column;gap:3px;min-width:0;flex:1}
  .card .fn{font-size:12.5px;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .card .st{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--tx3);text-transform:capitalize}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--tx3);flex:none;display:inline-block}
  .dot.labeled{background:var(--ok)} .dot.empty{background:#54607a}
  .dot.unlabeled{background:var(--tx3)} .dot.suggested{background:var(--ai)}
  .empty{padding:34px 12px;text-align:center;color:var(--tx3);font-size:12px}
  .side-stats{border-top:1px solid var(--line);padding:11px 12px 13px;max-height:232px;overflow:auto;background:var(--bg2)}
  .side-stats .sh{display:flex;justify-content:space-between;align-items:baseline;font-size:10.5px;color:var(--tx3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px}
  .side-stats .sh b{color:var(--tx2);letter-spacing:0;text-transform:none;font-size:11.5px}
  .statrow{display:flex;align-items:center;gap:8px;margin-bottom:6px}
  .statrow .sw{width:9px;height:9px;border-radius:3px;flex:none}
  .statrow .nm{font-size:11.5px;color:var(--tx2);width:72px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:none}
  .statrow .barwrap{flex:1;height:7px;background:var(--s1);border-radius:9px;overflow:hidden}
  .statrow .bar{height:100%;border-radius:9px;transition:width .3s ease}
  .statrow .ct{font-size:11px;color:var(--tx3);font-variant-numeric:tabular-nums;width:30px;text-align:right;flex:none}
  .side-stats .none{color:var(--tx3);font-size:11.5px;text-align:center;padding:6px 0}
  .modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(8,9,12,.82);backdrop-filter:blur(4px);z-index:20}
  .modal.show{display:flex}
  .mcard{width:min(680px,92vw);max-height:84vh;display:flex;flex-direction:column;background:var(--s1);border:1px solid var(--line2);border-radius:16px;box-shadow:var(--sh);overflow:hidden}
  .mhead{display:flex;align-items:center;justify-content:space-between;padding:15px 20px;border-bottom:1px solid var(--line)}
  .mhead h3{margin:0;font-size:15px}
  .mx{display:grid;place-items:center;width:30px;height:30px;border-radius:8px;background:var(--s2);border:1px solid var(--line);color:var(--tx2);font-size:17px;line-height:1}
  .mx:hover{background:var(--s3);color:var(--tx)}
  .mbody{padding:18px 20px;overflow:auto}
  .iload{color:var(--tx3);text-align:center;padding:24px}
  .igrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(108px,1fr));gap:10px;margin-bottom:18px}
  .icard{background:var(--s2);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
  .icard .ik{font-size:10.5px;color:var(--tx3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}
  .icard .iv{font-size:21px;font-weight:680;font-variant-numeric:tabular-nums;letter-spacing:-.2px}
  .isec{margin-bottom:18px}
  .isec .ititle{font-size:10.5px;color:var(--tx3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px}
  .ibar{display:flex;align-items:center;gap:10px;margin-bottom:6px;font-size:12px}
  .ibar .il{width:96px;color:var(--tx2);font-variant-numeric:tabular-nums}
  .ibar .it{flex:1;height:8px;background:var(--s2);border-radius:9px;overflow:hidden}
  .ibar .it span{display:block;height:100%;background:linear-gradient(90deg,var(--ac),var(--ai));border-radius:9px}
  .ibar .ic{width:34px;text-align:right;color:var(--tx3);font-variant-numeric:tabular-nums}
  .iok{display:flex;align-items:center;gap:8px;color:var(--ok);font-size:13px}
  .iok .ic{width:16px;height:16px}
  .iwarn{padding:10px 12px;border-radius:9px;background:var(--s2);border:1px solid var(--line);color:var(--tx2);font-size:12.5px;margin-bottom:12px}
  .iwarn.leak{background:rgba(251,113,113,.09);border-color:rgba(251,113,113,.35);color:var(--danger);font-weight:560}
  .idup{display:flex;align-items:center;gap:5px;margin-bottom:8px;flex-wrap:wrap}
  .ithumb{width:46px;height:36px;object-fit:cover;border-radius:5px;background:var(--s3)}
  .idsplit{margin-left:4px;font-size:11px;color:var(--tx3)}
  .insbtn{display:grid;place-items:center;width:32px;height:32px;border-radius:8px;background:transparent;border:1px solid transparent;color:var(--tx2)}
  .insbtn:hover{background:var(--s2);color:var(--tx);border-color:var(--line)}
  .traincta{display:none;align-items:center;justify-content:space-between;gap:8px;padding:10px 12px;
    border-top:1px solid var(--line);background:linear-gradient(180deg,rgba(45,212,167,.07),transparent)}
  .traincta .t-l{display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--ok);font-weight:560}
  .traincta .t-l .ic{width:14px;height:14px}
  .t-cmd{display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:7px;
    background:var(--s2);border:1px solid var(--line2);color:var(--tx2);transition:.13s}
  .t-cmd:hover{background:var(--s3);color:var(--tx)}
  .t-cmd code{font:11px ui-monospace,monospace}
  .t-cmd .ic{width:13px;height:13px}
  .t-cmd.copied{color:var(--ok);border-color:rgba(45,212,167,.4);background:rgba(45,212,167,.08)}
  /* stage */
  .stage{position:relative;min-width:0;overflow:hidden;
    background:radial-gradient(130% 130% at 50% 0%,var(--stage1),var(--stage2))}
  canvas{display:block;width:100%;height:100%;touch-action:none;cursor:crosshair}
  .glass{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--line2)}
  .toolbar{position:absolute;top:14px;right:14px;display:flex;flex-direction:column;gap:5px;
    padding:6px;border-radius:13px;box-shadow:var(--sh)}
  .tool{display:grid;place-items:center;width:36px;height:36px;border-radius:9px;background:transparent;
    border:1px solid transparent;color:var(--tx2);transition:.12s}
  .tool:hover{background:var(--s2);color:var(--tx)}
  .tool.ai{color:var(--ai)} .tool.ai:hover{background:rgba(34,211,238,.16)}
  .tdiv{height:1px;background:var(--line);margin:2px 5px}
  .hud{position:absolute;top:14px;left:14px;padding:7px 12px;border-radius:10px;font-size:12px;
    color:var(--tx2);box-shadow:var(--shs);font-variant-numeric:tabular-nums}
  .classbar{position:absolute;left:50%;bottom:16px;transform:translateX(-50%)}
  .classchip{display:inline-flex;align-items:center;gap:9px;height:40px;padding:0 16px;border-radius:999px;
    color:var(--tx);box-shadow:var(--sh);font-weight:560;transition:.14s}
  .classchip:hover{border-color:var(--ac)}
  .classchip .sw{width:14px;height:14px;border-radius:4px}
  .classchip .cc-h{color:var(--tx3);font-size:10.5px;text-transform:uppercase;letter-spacing:.7px}
  .picker{position:absolute;left:50%;bottom:66px;transform:translateX(-50%) translateY(8px);
    width:min(540px,88vw);max-height:48vh;display:none;flex-direction:column;opacity:0;transition:.16s;
    background:var(--s1);border:1px solid var(--line2);border-radius:15px;box-shadow:var(--sh);overflow:hidden;z-index:4}
  .picker.show{display:flex;opacity:1;transform:translateX(-50%) translateY(0)}
  .psearch{display:flex;align-items:center;gap:9px;padding:12px 14px;border-bottom:1px solid var(--line)}
  .psearch .ic{width:15px;height:15px;color:var(--tx3)}
  .psearch input{flex:1;background:transparent;border:0;outline:none;color:var(--tx);font-size:13px}
  #pal{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:3px;padding:10px;overflow:auto}
  .pclass{display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:8px;background:transparent;border:1px solid transparent;text-align:left;transition:.1s}
  .pclass:hover{background:var(--s2)} .pclass.on{background:var(--s3);border-color:var(--ac)}
  .pclass .sw{width:12px;height:12px;border-radius:3px;flex:none}
  .pclass .pn{flex:1;font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .pclass .pk{color:var(--tx3);font-size:11px;font-variant-numeric:tabular-nums;background:var(--s3);border-radius:4px;padding:0 5px}
  .banner{position:absolute;left:50%;top:14px;transform:translateX(-50%);display:none;align-items:center;gap:8px;
    max-width:min(680px,84vw);padding:9px 14px;border-radius:10px;font-size:12.5px;
    background:var(--s1);color:var(--warn);border:1px solid color-mix(in srgb,var(--warn) 40%,transparent);box-shadow:var(--sh)}
  .progress{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
    background:rgba(8,9,12,.78);backdrop-filter:blur(3px);z-index:6}
  .pcard{width:384px;padding:26px;border-radius:16px;background:var(--s1);border:1px solid var(--line2);box-shadow:var(--sh);text-align:center}
  .pcard .ic{width:30px;height:30px;color:var(--ai);margin:0 auto 10px}
  .ptitle{font-weight:650;font-size:15px;margin-bottom:5px}
  .ptxt{color:var(--tx3);font-size:12.5px;font-variant-numeric:tabular-nums;margin-bottom:15px;min-height:18px}
  .ptrack{height:7px;border-radius:99px;background:var(--s3);overflow:hidden}
  .pbar{height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,var(--ac),var(--ai));transition:width .25s ease}
  .help{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(8,9,12,.8);backdrop-filter:blur(4px);z-index:30}
  .help .card2{width:min(720px,92vw);max-height:86vh;overflow:auto;background:var(--s1);border:1px solid var(--line2);border-radius:16px;padding:20px 22px;box-shadow:var(--sh)}
  .help .hh{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}
  .help h3{margin:0;font-size:15px;display:flex;align-items:center;gap:8px}
  .help-sub{color:var(--tx3);font-size:12.5px;margin:0 0 16px}
  .kgrid{display:grid;grid-template-columns:1fr 1fr;gap:16px 24px}
  @media (max-width:680px){ .kgrid{grid-template-columns:1fr} }
  .kgroup h4{margin:0 0 6px;font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;color:var(--tx3)}
  .krow{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:5px 0;border-bottom:1px solid var(--line);font-size:12.5px;color:var(--tx2)}
  .krow:last-child{border-bottom:0}
  .krow .keys{display:flex;align-items:center;gap:4px;flex:none}
  .help kbd{display:inline-block;background:var(--s3);border:1px solid var(--line2);border-bottom-width:2px;border-radius:6px;padding:1px 7px;font:11px ui-monospace,monospace;color:var(--tx)}
  .help kbd.k-edit{cursor:pointer;min-width:14px;text-align:center;transition:.12s}
  .help kbd.k-edit:hover{border-color:var(--ac);color:var(--ac)}
  .help kbd.k-live{border-color:var(--ac);color:var(--ac);box-shadow:0 0 0 3px color-mix(in srgb,var(--ac) 22%,transparent)}
  .khint-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:16px}
  .khint{font-size:12px;color:var(--tx3)} .khint.err{color:var(--danger)}
  :focus-visible{outline:2px solid var(--ac);outline-offset:2px}
  /* --- data-quality + AI superpowers: Radar, Boost, Map, dup-fixer --- */
  .chip{display:none;align-items:center;gap:7px;height:30px;padding:0 12px;border-radius:999px;
    font-size:11.5px;font-weight:560;border:1px solid var(--line2);background:var(--s2);color:var(--tx2);max-width:300px}
  .chip.show{display:inline-flex}
  .chip.good{color:var(--ok);border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.09)}
  .chip.run{color:var(--ai);border-color:rgba(34,211,238,.35);background:rgba(34,211,238,.09)}
  .chip.bad{color:var(--danger);border-color:rgba(239,68,68,.4);background:rgba(239,68,68,.09)}
  .chip .ic{width:14px;height:14px}
  .chip .x{margin-left:1px;opacity:.55;font-size:14px;line-height:1}
  .chip .x:hover{opacity:1}
  .spin{animation:spin 1s linear infinite;transform-origin:50% 50%}
  @keyframes spin{to{transform:rotate(360deg)}}
  .rsummary{margin-bottom:13px}
  .rlast{display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:11.5px;color:var(--tx3)}
  .rrow{display:flex;align-items:center;gap:11px;padding:8px 10px;border-radius:10px;background:var(--s2);
    border:1px solid var(--line);margin-bottom:7px;width:100%;text-align:left;transition:.12s}
  .rrow:hover{border-color:var(--ac);background:var(--s3)}
  .rrow.cur{border-color:var(--ac)}
  .rrow .rthumb{width:62px;height:46px;object-fit:cover;border-radius:6px;background:var(--s3);flex:none}
  .rrow .rmeta{flex:1;min-width:0}
  .rrow .rfn{font-size:12.5px;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .rrow .rb{display:inline-flex;align-items:center;gap:5px;margin-top:5px;flex-wrap:wrap}
  .rbadge{font-size:10.5px;font-weight:600;padding:2px 7px;border-radius:999px}
  .rbadge.class{background:rgba(251,191,36,.16);color:var(--warn)}
  .rbadge.miss{background:rgba(34,211,238,.16);color:var(--ai)}
  .rbadge.phantom{background:rgba(239,68,68,.16);color:var(--danger)}
  .rscore{font-size:11px;color:var(--tx3);font-variant-numeric:tabular-nums;flex:none;width:30px;text-align:right}
  .rsev{width:44px;height:5px;background:var(--s3);border-radius:9px;overflow:hidden;flex:none}
  .rsev span{display:block;height:100%;border-radius:9px}
  .ifix{display:inline-flex;align-items:center;gap:6px;height:26px;padding:0 10px;border-radius:7px;margin-left:auto;
    background:var(--ac);color:var(--ac-ink);border:0;font-size:11.5px;font-weight:600;white-space:nowrap}
  .ifix:hover{filter:brightness(1.08)} .ifix:disabled{opacity:.6}
  .ifix.done{background:transparent;color:var(--ok);border:1px solid rgba(16,185,129,.4)}
  .qrow{display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:8px;background:var(--s2);margin-bottom:6px;width:100%;text-align:left}
  .qrow:hover{background:var(--s3)}
  .qrow .qt{font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;padding:2px 6px;border-radius:5px;background:var(--s3);color:var(--tx2);flex:none}
  .qrow .qt.tiny{color:var(--danger)} .qrow .qt.sliver{color:var(--warn)} .qrow .qt.fullframe{color:var(--ai)}
  .qrow .qn{font-size:12px;color:var(--tx);flex:none;width:118px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .qrow .qm{font-size:11.5px;color:var(--tx2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .mapcard{width:min(880px,95vw);height:min(700px,92vh)}
  .mapstage{position:relative;flex:1;min-height:0;background:radial-gradient(120% 120% at 50% 0%,var(--stage1),var(--stage2));overflow:hidden}
  #mapcv{display:block;width:100%;height:100%;cursor:crosshair}
  .maphint{position:absolute;left:14px;bottom:12px;font-size:11.5px;color:var(--tx2);background:var(--glass);padding:6px 10px;border-radius:8px;border:1px solid var(--line2)}
  .maplegend{position:absolute;right:14px;top:12px;display:flex;gap:14px;font-size:11px;color:var(--tx2);background:var(--glass);padding:6px 11px;border-radius:8px;border:1px solid var(--line2)}
  .maplegend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
  /* --- project home --- */
  .home{position:fixed;inset:0;z-index:40;display:none;overflow:auto;
    background:radial-gradient(120% 90% at 50% -10%,var(--bg2),var(--bg))}
  .home.show{display:block}
  .home-theme{position:absolute;top:16px;right:18px}
  .home-inner{max-width:880px;margin:0 auto;padding:62px 24px 48px}
  .home-hero{text-align:center;margin-bottom:28px}
  .home-brand{display:inline-flex;align-items:center;gap:11px;font-size:30px;font-weight:680;letter-spacing:.2px}
  .home-brand .ic{width:34px;height:34px;color:var(--ac)}
  .home-brand b{color:var(--ac)}
  .home-tag{color:var(--tx2);font-size:14px;margin:12px auto 0;max-width:540px}
  .home-sec{max-width:840px;margin:32px auto 12px;color:var(--tx3);font-size:11px;text-transform:uppercase;letter-spacing:.7px}
  .home-grid{max-width:840px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(244px,1fr));gap:12px}
  .prj{position:relative;text-align:left;background:var(--s1);border:1px solid var(--line2);border-radius:13px;padding:15px 15px 14px;transition:.14s;width:100%;box-shadow:var(--shs)}
  .prj:hover{border-color:var(--ac);transform:translateY(-2px);box-shadow:var(--sh)}
  .prj-name{font-size:15px;font-weight:640;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-right:18px;letter-spacing:-.1px}
  .prj-path{font-size:11px;color:var(--tx3);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .prj-barwrap{height:7px;background:var(--s3);border-radius:9px;overflow:hidden;margin:14px 0 8px}
  .prj-bar{height:100%;border-radius:9px;background:linear-gradient(90deg,var(--ac),var(--ai))}
  .prj-meta{display:flex;justify-content:space-between;gap:6px;font-size:11.5px;color:var(--tx2);font-variant-numeric:tabular-nums}
  .prj-menu{position:absolute;top:8px;right:8px;width:24px;height:24px;border-radius:7px;display:grid;place-items:center;
    background:var(--s2);border:1px solid var(--line);color:var(--tx3);opacity:0;transition:.12s;cursor:pointer}
  .prj:hover .prj-menu{opacity:1}
  .prj-menu:hover{color:var(--tx);border-color:var(--line2)}
  .prj-menu .ic{width:15px;height:15px}
  .prj-actions{position:absolute;top:36px;right:8px;z-index:6;display:none;flex-direction:column;min-width:158px;
    background:var(--s1);border:1px solid var(--line2);border-radius:10px;box-shadow:var(--sh);padding:4px}
  .prj-actions.show{display:flex}
  .prj-act{text-align:left;background:transparent;border:0;color:var(--tx);font-size:12.5px;padding:7px 10px;border-radius:7px;cursor:pointer}
  .prj-act:hover{background:var(--s2)}
  .prj-act.danger{color:var(--danger)}
  .prj-act.danger:hover{background:color-mix(in srgb,var(--danger) 13%,transparent)}
  .home-empty{grid-column:1/-1;text-align:center;color:var(--tx3);font-size:13px;padding:28px}
  .rdy{margin-bottom:18px;padding:14px 15px;border-radius:11px;border:1px solid var(--line2);background:var(--s2)}
  .rdy.go{border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.07)}
  .rdy-h{display:flex;align-items:center;gap:9px;font-size:16px;font-weight:670;margin-bottom:12px;letter-spacing:-.2px}
  .rdy-h .ic{width:20px;height:20px} .rdy.go .rdy-h{color:var(--ok)}
  .rdy-row{display:flex;align-items:center;gap:9px;font-size:12.5px;color:var(--tx2);margin-bottom:7px}
  .rdy-row .ic{width:15px;height:15px;flex:none}
  .rdy-row.ok .ic{color:var(--ok)} .rdy-row.bad .ic{color:var(--danger)} .rdy-row.warn .ic{color:var(--warn)}
  .rdy .t-cmd{margin-top:12px}
  .sidesearch{display:flex;align-items:center;gap:8px;margin-top:8px;background:var(--s1);border:1px solid var(--line);border-radius:8px;padding:0 10px;height:30px}
  .sidesearch:focus-within{border-color:var(--ac)}
  .sidesearch .ic{width:14px;height:14px;color:var(--tx3);flex:none}
  .sidesearch input{flex:1;background:transparent;border:0;outline:none;color:var(--tx);font-size:12px}
  /* share popover */
  .pop{position:fixed;top:54px;right:14px;z-index:25;width:330px;display:none;
    background:var(--s1);border:1px solid var(--line2);border-radius:13px;box-shadow:var(--sh);padding:15px 16px}
  .pop.show{display:block}
  .pop h4{margin:0 0 5px;font-size:13.5px}
  .pop p{margin:0 0 12px;font-size:12px;color:var(--tx2);line-height:1.5}
  .pop p code{background:var(--s3);border-radius:4px;padding:0 5px;font:11px ui-monospace,monospace}
  .urlrow{display:flex;align-items:center;gap:8px;background:var(--s2);border:1px solid var(--line);border-radius:9px;padding:8px 10px}
  .urlrow code{flex:1;font:12px ui-monospace,monospace;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .urlrow button{flex:none;display:grid;place-items:center;width:28px;height:28px;border-radius:7px;background:var(--s3);border:1px solid var(--line2);color:var(--tx2)}
  .urlrow button:hover{color:var(--tx)}
  .urlrow button.copied{color:var(--ok);border-color:rgba(16,185,129,.4)}
  .cecard{width:min(460px,92vw)}
  .ce-note{color:var(--tx2);font-size:12.5px;line-height:1.5;margin:0 0 14px}
  .ce-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
  .ce-row .sw{width:14px;height:14px;border-radius:4px;flex:none}
  .ce-in{flex:1;background:var(--s2);border:1px solid var(--line2);border-radius:9px;padding:8px 11px;color:var(--tx);font-size:13px;outline:none}
  .ce-in:focus{border-color:var(--ac);box-shadow:0 0 0 3px color-mix(in srgb,var(--ac) 22%,transparent)}
  .ce-rm{flex:none;width:28px;height:28px;border-radius:8px;background:var(--s2);border:1px solid var(--line2);color:var(--tx2);font-size:16px;line-height:1;cursor:pointer}
  .ce-rm:hover{color:var(--danger);border-color:var(--danger)}
  .ce-fixed{flex:none;width:34px;text-align:center;color:var(--tx3);font:11px ui-monospace,monospace}
  .ce-add{margin-top:4px}
  .ce-err{color:var(--danger);font-size:12.5px;min-height:16px;margin-top:10px}
  .ce-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:12px}
  .classchip.empty{border-style:dashed;color:var(--tx2)}
  .classbar{display:none}   /* class selection now lives in the bottom label bar */
  .bottombar{position:absolute;left:0;right:0;bottom:0;display:flex;align-items:flex-end;gap:10px;padding:12px 14px;z-index:5;pointer-events:none}
  .bottombar>*{pointer-events:auto}
  .labelbar{display:flex;align-items:center;gap:7px;flex:1;min-width:0;overflow-x:auto;padding:3px;scrollbar-width:thin}
  .lchip{display:inline-flex;align-items:center;gap:7px;height:33px;padding:0 11px;border-radius:9px;flex:none;background:var(--s1);border:1px solid var(--line2);color:var(--tx2);font-size:12.5px;font-weight:540;white-space:nowrap;box-shadow:var(--sh);transition:.1s}
  .lchip:hover{border-color:var(--ac);color:var(--tx)}
  .lchip.on{border-color:var(--ac);background:var(--s3);color:var(--tx);box-shadow:0 0 0 1px var(--ac) inset,var(--sh)}
  .lchip .sw{width:12px;height:12px;border-radius:3px;flex:none}
  .lchip .ln{overflow:hidden;text-overflow:ellipsis;max-width:160px}
  .lchip .lk{font:11px ui-monospace,monospace;color:var(--tx3);background:var(--s3);border-radius:4px;padding:1px 5px;font-variant-numeric:tabular-nums}
  .lchip.add{color:var(--tx3);border-style:dashed;box-shadow:none}
  .submitbtn{display:inline-flex;align-items:center;gap:8px;height:39px;padding:0 18px;border-radius:10px;flex:none;background:var(--ac);color:var(--ac-ink);font-size:13px;font-weight:650;border:1px solid var(--ac);box-shadow:var(--sh);transition:.12s}
  .submitbtn:hover{filter:brightness(1.08)}
  .submitbtn:disabled{opacity:.45;cursor:default;filter:none}
  .submitbtn .ic{width:16px;height:16px}
  .submitbtn kbd{background:rgba(255,255,255,.22);border-radius:5px;padding:1px 6px;font:11px ui-monospace,monospace}
  /* ===== New Project wizard ===== */
  .wz{position:fixed;inset:0;z-index:60;display:none;align-items:center;justify-content:center;
    background:rgba(6,6,8,.66);backdrop-filter:blur(6px)}
  .wz.show{display:flex}
  .wz-card{width:min(640px,94vw);max-height:90vh;display:flex;flex-direction:column;
    background:var(--s1);border:1px solid var(--line2);border-radius:18px;box-shadow:var(--sh);overflow:hidden}
  .wz-head{display:flex;align-items:center;gap:14px;padding:16px 18px;border-bottom:1px solid var(--line)}
  .wz-title{font-size:15px;font-weight:680;letter-spacing:-.2px}
  .wz-steps{display:flex;align-items:center;gap:4px;margin-left:auto}
  .wz-step{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--tx3);padding:4px}
  .wz-step i{display:grid;place-items:center;width:20px;height:20px;border-radius:50%;font-style:normal;font-size:11px;
    font-weight:700;background:var(--s3);color:var(--tx2);transition:.15s}
  .wz-step.on{color:var(--tx)} .wz-step.on i{background:var(--ac);color:var(--ac-ink)}
  .wz-step.done i{background:color-mix(in srgb,var(--ac) 34%,var(--s3));color:var(--tx)}
  .wz-body{padding:20px 20px 8px;overflow:auto}
  .wz-pane{display:flex;flex-direction:column}
  .wz-lbl{font-size:12.5px;font-weight:600;color:var(--tx);margin:14px 0 7px}
  .wz-lbl:first-child{margin-top:0}
  .wz-lbl .wz-hint{color:var(--tx3);font-weight:400;margin-left:5px}
  .wz-lbl .req{color:var(--danger);margin-left:3px;font-weight:700}
  .wz-in{width:100%;background:var(--s2);border:1px solid var(--line2);border-radius:10px;
    padding:10px 12px;color:var(--tx);font-size:13.5px;outline:none;transition:.12s}
  .wz-in::placeholder{color:var(--tx3)}
  .wz-in:focus{border-color:var(--ac);box-shadow:0 0 0 3px color-mix(in srgb,var(--ac) 22%,transparent)}
  .wz-ta{min-height:62px;resize:vertical;font-family:inherit}
  .wz-tabs{display:flex;gap:3px;padding:3px;background:var(--s2);border:1px solid var(--line);border-radius:10px;margin-bottom:14px}
  .wz-tab{flex:1;height:32px;border:0;border-radius:7px;background:transparent;color:var(--tx3);font-size:12.5px;font-weight:580}
  .wz-tab.on{background:var(--s1);color:var(--tx);box-shadow:var(--shs)}
  .wz-folder{display:flex;gap:8px} .wz-folder .wz-in{flex:1}
  .wz-drop{margin-top:12px;border:1.5px dashed var(--line2);border-radius:12px;padding:26px 16px;text-align:center;
    background:var(--s2);transition:.12s;cursor:pointer}
  .wz-drop:hover,.wz-drop.over{border-color:var(--ac);background:color-mix(in srgb,var(--ac) 8%,var(--s2))}
  .wz-drop.disabled{opacity:.5}
  .wz-drop svg{width:26px;height:26px;color:var(--ac);margin:0 auto 8px;display:block}
  .wz-drop-t{font-size:13.5px;font-weight:600;color:var(--tx)}
  .wz-drop-s{font-size:12px;color:var(--tx3);margin-top:4px}
  .wz-link{background:none;border:0;color:var(--ac);font:inherit;font-weight:600;padding:0;cursor:pointer}
  .wz-files{margin-top:12px;max-height:148px;overflow:auto;display:flex;flex-direction:column;gap:5px}
  .wz-frow{display:flex;align-items:center;gap:9px;font-size:12px;color:var(--tx2);padding:5px 9px;background:var(--s2);border-radius:8px}
  .wz-frow .wz-fn{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .wz-frow .wz-fs{font-variant-numeric:tabular-nums;color:var(--tx3);flex:none}
  .wz-frow.ok .wz-fs{color:var(--ok)} .wz-frow.err{color:var(--danger)} .wz-frow.err .wz-fn{color:var(--danger)}
  .wz-note{font-size:12px;color:var(--tx3);line-height:1.5;margin:12px 0 0}
  .wz-tasks{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .wz-task{display:flex;align-items:center;gap:8px;padding:11px 13px;border-radius:10px;background:var(--s2);
    border:1px solid var(--line2);color:var(--tx2);font-size:13px;font-weight:560;text-align:left;transition:.1s}
  .wz-task:hover:not([disabled]){border-color:var(--ac);color:var(--tx)}
  .wz-task.on{border-color:var(--ac);background:color-mix(in srgb,var(--ac) 12%,var(--s2));color:var(--tx);box-shadow:0 0 0 1px var(--ac) inset}
  .wz-task[disabled]{opacity:.5}
  .wz-task .soon{margin-left:auto;font-size:9px;text-transform:uppercase;letter-spacing:.5px;color:var(--tx3);background:var(--s3);border-radius:4px;padding:1px 5px}
  .wz-storage{display:grid;grid-template-columns:1fr;gap:8px}
  .wz-storage .wz-task{display:block}
  .wz-storage .sub{display:block;font-size:11px;color:var(--tx3);font-weight:400;margin-top:3px;line-height:1.45}
  .wz-classes{display:flex;flex-direction:column;gap:8px}
  .wz-clsrow{display:flex;align-items:center;gap:9px}
  .wz-clsrow input[type=color]{flex:none;width:34px;height:34px;padding:2px;border:1px solid var(--line2);border-radius:9px;background:var(--s2);cursor:pointer}
  .wz-clsrow .wz-in{flex:1}
  .wz-clsrow .wz-rm{flex:none;display:grid;place-items:center;width:34px;height:34px;border-radius:9px;background:var(--s2);border:1px solid var(--line2);color:var(--tx3);font-size:17px;line-height:1}
  .wz-clsrow .wz-rm:hover{color:var(--danger);border-color:var(--danger)}
  .wz-addcls{margin-top:10px;align-self:flex-start}
  .wz-check{display:inline-flex;align-items:center;gap:9px;margin-top:16px;font-size:12.5px;color:var(--tx2);cursor:pointer}
  .wz-check input{width:16px;height:16px;accent-color:var(--ac)}
  .wz-foot{display:flex;align-items:center;gap:12px;padding:14px 18px;border-top:1px solid var(--line);background:var(--bg2)}
  .wz-err{flex:1;color:var(--danger);font-size:12.5px;min-height:16px}
  .wz-nav{display:flex;gap:9px}
  .exp-ratios{display:flex;gap:14px;margin-top:10px;flex-wrap:wrap}
  .exp-ratios label{display:flex;flex-direction:column;gap:4px;font-size:11.5px;color:var(--tx2)}
  .exp-ratios input{width:92px;background:var(--s2);border:1px solid var(--line2);border-radius:8px;padding:6px 9px;color:var(--tx);font-size:13px;outline:none}
  .exp-ratios input:focus{border-color:var(--ac)}
  .exp-fmts{display:flex;gap:18px;flex-wrap:wrap;margin-top:2px}
  .exp-result{font-size:12px;color:var(--ok);margin-top:8px;word-break:break-all;line-height:1.5}
  .home-actions{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin:6px auto 0}
  .home-actions .btn{height:46px;padding:0 24px;font-size:14px;font-weight:640;border-radius:12px}
  /* in-app feedback (first-party) */
  .fb{position:fixed;right:16px;bottom:14px;z-index:70}
  .fb-btn{display:inline-flex;align-items:center;gap:7px;height:34px;padding:0 14px;border-radius:999px;
    color:var(--tx2);font-size:12.5px;font-weight:560;box-shadow:var(--sh)}
  .fb-btn:hover{color:var(--tx);border-color:var(--ac)}
  .fb-btn .ic{width:15px;height:15px}
  .fb-panel{position:absolute;right:0;bottom:44px;width:300px;display:none;background:var(--s1);
    border:1px solid var(--line2);border-radius:13px;box-shadow:var(--sh);padding:13px 14px}
  .fb-panel.show{display:block}
  .fb-head{display:flex;align-items:center;justify-content:space-between;font-size:13px;font-weight:640;margin-bottom:9px}
  .fb-panel textarea{width:100%;box-sizing:border-box;min-height:88px;resize:vertical;background:var(--s2);
    border:1px solid var(--line2);border-radius:9px;padding:9px 11px;color:var(--tx);font:12.5px/1.5 inherit;outline:none}
  .fb-panel textarea:focus{border-color:var(--ac)}
  .fb-foot{display:flex;align-items:center;justify-content:space-between;gap:9px;margin-top:9px}
  .fb-status{flex:1;font-size:11.5px;color:var(--tx3);min-height:14px}
  .fb-status.ok{color:var(--ok)} .fb-status.err{color:var(--danger)}
  .fb-status a{color:inherit;font-weight:600}
  .fb-note{margin-top:8px;font-size:10.5px;color:var(--tx3)}
"""
