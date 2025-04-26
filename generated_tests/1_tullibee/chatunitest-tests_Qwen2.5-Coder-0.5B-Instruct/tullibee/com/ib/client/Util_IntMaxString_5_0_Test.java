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
    void testIntMaxString() {
        // Arrange
        int input = Integer.MAX_VALUE;
        // Act
        String result = Util.IntMaxString(input);
        // Assert
        // Expected: ""
        assertEquals("", result);
    }
}
