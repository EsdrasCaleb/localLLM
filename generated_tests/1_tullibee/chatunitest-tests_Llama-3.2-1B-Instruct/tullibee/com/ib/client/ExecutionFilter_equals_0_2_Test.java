package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ExecutionFilter_equals_0_2_Test {

    @Test
    public void testEquals() {
        ExecutionFilter executionFilter1 = new ExecutionFilter(1, "ABC123", "2022-01-01", "symbol", "secType", "exchange", "buy");
        ExecutionFilter executionFilter2 = new ExecutionFilter(1, "ABC123", "2022-01-01", "symbol", "secType", "exchange", "buy");
        ExecutionFilter executionFilter3 = new ExecutionFilter(1, "ABC123", "2022-01-01", "symbol", "secType", "exchange", "sell");
        ExecutionFilter executionFilter4 = new ExecutionFilter(1, "ABC123", "2022-01-01", "symbol", "secType", "exchange", "buy");
        assertEquals(true, executionFilter1.equals(executionFilter2));
        assertEquals(false, executionFilter1.equals(executionFilter3));
        assertEquals(false, executionFilter1.equals(executionFilter4));
    }
}
