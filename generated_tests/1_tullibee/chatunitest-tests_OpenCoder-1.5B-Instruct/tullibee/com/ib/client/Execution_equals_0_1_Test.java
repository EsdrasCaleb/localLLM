package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Execution_equals_0_1_Test {

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
