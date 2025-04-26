package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ExecutionFilter_equals_0_0_Test {

    @Mock
    private Object objectToCompare;

    @Test
    public void testEquals_NullObject() {
        ExecutionFilter executionFilter = new ExecutionFilter();
        boolean result = executionFilter.equals(objectToCompare);
        assert result == false;
    }

    @Test
    public void testEquals_SameObject() {
        ExecutionFilter executionFilter = new ExecutionFilter();
        boolean result = executionFilter.equals(executionFilter);
        assert result == true;
    }

    @Test
    public void testEquals_DifferentObjects() {
        ExecutionFilter executionFilter1 = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "Buy");
        ExecutionFilter executionFilter2 = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "Buy");
        boolean result = executionFilter1.equals(executionFilter2);
        assert result == true;
    }

    @Test
    public void testEquals_DifferentProperties() {
        ExecutionFilter executionFilter1 = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "Buy");
        ExecutionFilter executionFilter2 = new ExecutionFilter(2, "67890", "2022-01-01", "GOOG", "Stock", "NASDAQ", "Sell");
        boolean result = executionFilter1.equals(executionFilter2);
        assert result == false;
    }

    @Test
    public void testEquals_NullStringComparison() {
        ExecutionFilter executionFilter = new ExecutionFilter(1, "12345", "2022-01-01", "AAPL", "Stock", "NYSE", "Buy");
        boolean result = executionFilter.equals(null);
        assert result == false;
    }
}
