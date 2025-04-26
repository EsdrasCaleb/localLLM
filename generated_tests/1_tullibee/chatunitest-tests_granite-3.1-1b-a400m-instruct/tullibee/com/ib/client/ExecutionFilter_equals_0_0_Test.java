package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ExecutionFilter_equals_0_0_Test {

    @Test
    public void testEquals() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "ABC", "2023-04-01", "AAPL", "NYSE", "BUY", "NYSE");
        ExecutionFilter filter2 = new ExecutionFilter(1, "ABC", "2023-04-01", "AAPL", "NYSE", "BUY", "NYSE");
        ExecutionFilter filter3 = new ExecutionFilter(2, "XYZ", "2023-04-02", "GOOGL", "NASDAQ", "SELL", "NASDAQ");
        assertTrue(filter1.equals(filter2));
        assertFalse(filter1.equals(filter3));
    }
}
