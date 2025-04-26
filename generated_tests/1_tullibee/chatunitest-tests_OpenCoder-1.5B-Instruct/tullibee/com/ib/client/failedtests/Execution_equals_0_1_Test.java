package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Execution_equals_0_1_Test {

    @Test
    public void testEquals() {
        Execution execution = mock(Execution.class);
        Execution otherExecution = mock(Execution.class);
        when(execution.equals(otherExecution)).thenReturn(true);
        Assertions.assertTrue(execution.equals(otherExecution));
    }

    @Test
    public void testNotEquals() {
        Execution execution = mock(Execution.class);
        Execution otherExecution = mock(Execution.class);
        when(execution.equals(otherExecution)).thenReturn(false);
        Assertions.assertFalse(execution.equals(otherExecution));
    }

    @Test
    public void testNotEqualsNull() {
        Execution execution = mock(Execution.class);
        Assertions.assertFalse(execution.equals(null));
    }

    @Test
    public void testNotEqualsDifferentClass() {
        Execution execution = mock(Execution.class);
        Assertions.assertFalse(execution.equals(new Object()));
    }
}
