// Test method
package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Execution_equals_0_0_Test {

    @Test
    public void testEqualsNull() {
        Execution execution1 = new Execution();
        // Test with null argument
        assertFalse(execution1.equals(null));
    }
}
