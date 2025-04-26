package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Execution_equals_0_0_Test {

    @Test
    void testEquals() {
        Execution execution1 = new Execution(1, 2, "exec1", "2023-10-26", "1234567890", "EURUSD", "BUY", 100, 100.5, 1, 0, 100, 100.5);
        Execution execution2 = new Execution(1, 2, "exec1", "2023-10-26", "1234567890", "EURUSD", "BUY", 100, 100.5, 1, 0, 100, 100.5);
        Execution execution3 = new Execution(1, 2, "exec2", "2023-10-26", "1234567890", "EURUSD", "BUY", 100, 100.5, 1, 0, 100, 100.5);
        assertTrue(execution1.equals(execution2));
        assertTrue(execution1.equals(execution3));
        assertFalse(execution1.equals(null));
        assertFalse(execution1.equals(new Object()));
    }
}
