package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Execution_equals_0_1_Test {

    @Test
    void testEquals() {
        Execution execution1 = new Execution(1, 1, "exec1", "time1", "acct1", "exchange1", "side1", 10, 10.0, 1, 1, 1, 10.0);
        Execution execution2 = new Execution(2, 2, "exec2", "time2", "acct2", "exchange2", "side2", 20, 20.0, 2, 2, 2, 20.0);
        Execution execution3 = new Execution(3, 3, "exec3", "time3", "acct3", "exchange3", "side3", 30, 30.0, 3, 3, 3, 30.0);
        // Testing with null
        assertFalse(execution1.equals(null));
        // Testing with same object
        assertTrue(execution1.equals(execution1));
        // Testing with different object
        assertFalse(execution1.equals(execution2));
        assertFalse(execution2.equals(execution3));
        assertFalse(execution3.equals(execution1));
    }
}
