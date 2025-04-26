package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompare_2_0_Test {

    @Test
    public void testStringCompare() {
        // Arrange
        Util util = new Util();
        String lhs = "Hello World";
        String rhs = "hello world";
        // Act
        int result = util.StringCompare(lhs, rhs);
        // Assert
        assertEquals(0, result);
    }
}
