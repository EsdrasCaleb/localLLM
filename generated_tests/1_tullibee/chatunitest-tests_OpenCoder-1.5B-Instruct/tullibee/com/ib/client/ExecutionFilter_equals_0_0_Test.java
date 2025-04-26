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
        ExecutionFilter filter1 = new ExecutionFilter(123, "ACC123", "2023-01-01 10:00:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter2 = new ExecutionFilter(123, "ACC123", "2023-01-01 10:00:00", "AAPL", "STK", "NYSE", "BUY");
        ExecutionFilter filter3 = new ExecutionFilter(124, "ACC124", "2023-01-01 10:00:00", "AAPL", "STK", "NYSE", "BUY");
        assertTrue(filter1.equals(filter2));
        assertFalse(filter1.equals(filter3));
    }
}
