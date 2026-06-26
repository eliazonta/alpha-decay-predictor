\p 5050

system "c 200 200";
show "Starting LOBSTER KDB+ Server on port 5050...";

load_lobster:{[symbol; msg_file; ob_file]
    show "Loading LOBSTER Message File...";
    // No enlist on delimiter to prevent treating first row as headers
    msg_raw: ("FIJFFI"; ",") 0: hsym msg_file;
    msg: flip `time`event_type`order_id`size`price`direction ! msg_raw;
    
    show "Loading LOBSTER Orderbook File...";
    ob_raw: (40#"F"; ",") 0: hsym ob_file;
    
    // Dynamically generate column names for all 10 levels
    ob_cols: raze {`$("ask_price_",string x;"ask_size_",string x;"bid_price_",string x;"bid_size_",string x)} each 1+til 10;
    ob: flip ob_cols ! ob_raw;
    
    // Rename Level 1 columns to our standard names
    ob: `best_ask_price`best_ask_qty`best_bid_price`best_bid_qty xcol ob;
    
    // Fix price scaling natively across all price columns
    price_cols: cols[ob] where (string cols ob) like "*price*";
    ob: ![ob; (); 0b; price_cols ! { (%; x; 10000.0) } each price_cols];
    
    show "Joining and Computing Multi-Level Features Vectorized...";
    quotes:: msg,'ob;
    update sym: symbol from `quotes;
    
    // Level 1 Features
    update
        micro_price: ((best_bid_price * best_ask_qty) + (best_ask_price * best_bid_qty)) % (best_bid_qty + best_ask_qty),
        mid_price: (best_ask_price + best_bid_price) % 2,
        obi: (best_bid_qty - best_ask_qty) % (best_bid_qty + best_ask_qty),
        spread: best_ask_price - best_bid_price
    from `quotes;

    // Multi-Level OBI (Top 5 and Top 10 levels)
    update 
        sum_ask_5: best_ask_qty + ask_size_2 + ask_size_3 + ask_size_4 + ask_size_5,
        sum_bid_5: best_bid_qty + bid_size_2 + bid_size_3 + bid_size_4 + bid_size_5,
        sum_ask_10: best_ask_qty + ask_size_2 + ask_size_3 + ask_size_4 + ask_size_5 + ask_size_6 + ask_size_7 + ask_size_8 + ask_size_9 + ask_size_10,
        sum_bid_10: best_bid_qty + bid_size_2 + bid_size_3 + bid_size_4 + bid_size_5 + bid_size_6 + bid_size_7 + bid_size_8 + bid_size_9 + bid_size_10
    from `quotes;
    
    update 
        obi_5: (sum_bid_5 - sum_ask_5) % (sum_bid_5 + sum_ask_5),
        obi_10: (sum_bid_10 - sum_ask_10) % (sum_bid_10 + sum_ask_10)
    from `quotes;

    update rolling_vol: 100 mdev micro_price from `quotes;

    // Target forward returns.
    //   Primary target = MID-price return -- the tradable mid.
    //   Micro-price returns are kept ONLY as a mechanical baseline: the
    //   micro-price is by construction pulled toward the heavier queue, so it
    //   co-moves with OBI almost algebraically (corr ~0.9). Predicting micro
    //   returns from OBI therefore overstates the tradable signal (IC ~0.33 vs
    //   ~0.10 on the mid). See README "Micro-price is a mechanical baseline".
    update
        ret_10c:  (((-10)  xprev mid_price) - mid_price) % mid_price,
        ret_50c:  (((-50)  xprev mid_price) - mid_price) % mid_price,
        ret_200c: (((-200) xprev mid_price) - mid_price) % mid_price,
        ret_micro_10c:  (((-10)  xprev micro_price) - micro_price) % micro_price,
        ret_micro_50c:  (((-50)  xprev micro_price) - micro_price) % micro_price,
        ret_micro_200c: (((-200) xprev micro_price) - micro_price) % micro_price
    from `quotes;

    quotes:: select from quotes where not null rolling_vol,
        not null ret_10c, not null ret_50c, not null ret_200c;
    
    show "Data ingested and engineered successfully. Ready for PyKX IPC queries!";
 };
