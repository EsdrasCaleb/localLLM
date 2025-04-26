package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ExecutionFilter_equals_0_1_Test {

    @Test
    public void testEquals() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "acct1", "time1", "sym1", "sec1", "exch1", "side1");
        ExecutionFilter filter2 = new ExecutionFilter(1, "acct1", "time1", "sym1", "sec1", "exch1", "side1");
        assertTrue(filter1.equals(filter2));
        ExecutionFilter filter3 = new ExecutionFilter(2, "acct2", "time2", "sym2", "sec2", "exch2", "side2");
        assertFalse(filter1.equals(filter3));
        filter3 = new ExecutionFilter(1, "acct3", "time3", "sym3", "sec3", "exch3", "side3");
        assertFalse(filter1.equals(filter3));
        filter3 = new ExecutionFilter(1, "acct1", "time4", "sym4", "sec4", "exch4", "side4");
        assertFalse(filter1.equals(filter3));
        filter3 = new ExecutionFilter(1, "acct1", "time1", "sym5", "sec5", "exch5", "side5");
        assertFalse(filter1.equals(filter3));
        filter3 = null;
        assertFalse(filter1.equals(filter3));
    }
}
