package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ExecutionFilter_equals_0_0_Test {

    @Test
    void testEquals() {
        ExecutionFilter executionFilter1 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "side1");
        ExecutionFilter executionFilter2 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "secType1", "exchange1", "side1");
        ExecutionFilter executionFilter3 = new ExecutionFilter(2, "acct2", "time2", "symbol2", "secType2", "exchange2", "side2");
        assertTrue(executionFilter1.equals(executionFilter2));
        assertFalse(executionFilter1.equals(executionFilter3));
    }
}
