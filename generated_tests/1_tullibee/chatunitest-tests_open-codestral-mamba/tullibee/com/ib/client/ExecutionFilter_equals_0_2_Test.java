package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ExecutionFilter_equals_0_2_Test {

    private ExecutionFilter filter1;

    private ExecutionFilter filter2;

    @BeforeEach
    void setUp() {
        filter1 = new ExecutionFilter(1, "acctCode1", "time1", "symbol1", "secType1", "exchange1", "side1");
        filter2 = new ExecutionFilter(1, "acctCode1", "time1", "symbol1", "secType1", "exchange1", "side1");
    }

    @Test
    void testEquals() {
        // Testing if the object is equal to itself
        assertTrue(filter1.equals(filter1));
        // Testing if two equal objects are considered equal
        assertTrue(filter1.equals(filter2));
        // Changing a field to make the objects unequal
        filter2.m_clientId = 2;
        // Testing if two unequal objects are considered unequal
        assertFalse(filter1.equals(filter2));
        // Making the second object null
        filter2 = null;
        // Testing if the object is equal to null
        assertFalse(filter1.equals(filter2));
    }
}
