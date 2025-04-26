package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ExecutionFilter_equals_0_0_Test {

    @Test
    public void testEquals() throws Exception {
        // Create two ExecutionFilter objects with the same data
        ExecutionFilter filter1 = new ExecutionFilter(1234, "ACCT1", "2023-10-05", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(1234, "ACCT1", "2023-10-05", "AAPL", "STK", "NYSE", "BUY");
        // Check if they are equal
        assertTrue(filter1.equals(filter2));
        // Create two ExecutionFilter objects with different data
        ExecutionFilter filter3 = new ExecutionFilter(1234, "ACCT1", "2023-10-05", "AAPL", "STK", "NYSE", "SELL");
        ExecutionFilter filter4 = new ExecutionFilter(1234, "ACCT1", "2023-10-06", "GOOGL", "STK", "NASDAQ", "BUY");
        // Check if they are not equal
        assertFalse(filter3.equals(filter4));
    }
}
