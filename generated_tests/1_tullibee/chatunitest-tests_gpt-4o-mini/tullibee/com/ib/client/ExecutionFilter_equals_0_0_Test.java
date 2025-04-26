package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ExecutionFilter_equals_0_0_Test {

    private ExecutionFilter filter1;

    private ExecutionFilter filter2;

    private ExecutionFilter filter3;

    @BeforeEach
    public void setUp() {
        filter1 = new ExecutionFilter(1, "Acct1", "12:00", "AAPL", "STK", "NASDAQ", "BUY");
        // same values, different case
        filter2 = new ExecutionFilter(1, "acct1", "12:00", "aapl", "stk", "nasdaq", "buy");
        filter3 = new ExecutionFilter(2, "Acct2", "12:30", "GOOG", "STK", "NYSE", "SELL");
    }

    @Test
    public void testEquals_Null() {
        assertFalse(filter1.equals(null), "Should return false when comparing to null");
    }

    @Test
    public void testEquals_SameInstance() {
        assertTrue(filter1.equals(filter1), "Should return true when comparing the same instance");
    }

    @Test
    public void testEquals_SameValuesDifferentCase() {
        assertTrue(filter1.equals(filter2), "Should return true when values are the same ignoring case");
    }

    @Test
    public void testEquals_DifferentValues() {
        assertFalse(filter1.equals(filter3), "Should return false when values are different");
    }
}
