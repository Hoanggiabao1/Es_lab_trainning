upper_bound = sum(2**j for j in range(n_bits))
        # Create PB constraint: sum(power_terms) - sum(binary_terms) <= 0
        # This is equivalent to: sum(power_terms) <= sum(binary_terms)
        pb_clauses = PBEnc.leq(lits=lits, weights=coeffs, bound=UB,
                                 top_id=start)
            
        # Update variable counter
        if pb_clauses.nv > start:
            start = pb_clauses.nv + 1
            
        # Add the encoded clauses to WCNF
        for clause in pb_clauses.clauses:
            wcnf.append(clause)