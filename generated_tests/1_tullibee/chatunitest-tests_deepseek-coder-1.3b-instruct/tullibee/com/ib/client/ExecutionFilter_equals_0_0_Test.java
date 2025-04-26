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
        ExecutionFilter executionFilter1 = new ExecutionFilter(1, "ACC1", "TIME1", "SYMBOL1", "SECTYPE1", "EXCHANGE1", "SIDE1");
        ExecutionFilter executionFilter2 = new ExecutionFilter(1, "ACC1", "TIME1", "SYMBOL1", "SECTYPE1", "EXCHANGE1", "SIDE1");
        ExecutionFilter executionFilter3 = new ExecutionFilter(2, "ACC2", "TIME2", "SYMBOL2", "SECTYPE2", "EXCHANGE2", "SIDE2");
        // Testing equal objects
        assertTrue(executionFilter1.equals(executionFilter2));
        // Testing different objects
        assertFalse(executionFilter1.equals(executionFilter3));
        // Testing null object
        assertFalse(executionFilter1.equals(null));
    }
}
