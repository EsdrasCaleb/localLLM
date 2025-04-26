package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ExecutionFilter_equals_0_0_Test {

    @Test
    public void testEquals_Null() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "Buy");
        ExecutionFilter filter2 = null;
        assertFalse(filter1.equals(filter2));
    }

    @Test
    public void testEquals_Self() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "Buy");
        ExecutionFilter filter2 = filter1;
        assertTrue(filter1.equals(filter2));
    }

    @Test
    public void testEquals_MatchAllFields() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "Buy");
        ExecutionFilter filter2 = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "Buy");
        assertTrue(filter1.equals(filter2));
    }

    @Test
    public void testEquals_DifferentFields() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "Buy");
        ExecutionFilter filter2 = new ExecutionFilter(2, "67890", "2022-01-01", "GOOG", "Stock", "NASDAQ", "Sell");
        assertFalse(filter1.equals(filter2));
    }

    @Test
    public void testEquals_MismatchCase() {
        ExecutionFilter filter1 = new ExecutionFilter(1, "12345", "2022-01-01", "aapl", "Stock", "NYSE", "Buy");
        ExecutionFilter filter2 = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "buy");
        assertTrue(filter1.equals(filter2));
    }
}
