package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_IntMaxString_5_0_Test {

    @Test
    public void testIntMaxString() {
        // Given
        int value = Integer.MAX_VALUE;
        // When
        String result = Util.IntMaxString(value);
        // Then
        assertEquals("", result);
    }
}
