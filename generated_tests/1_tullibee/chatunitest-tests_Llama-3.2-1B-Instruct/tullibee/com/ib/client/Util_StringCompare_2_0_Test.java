package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_StringCompare_2_0_Test {

    @Mock
    private String lhs;

    @Mock
    private String rhs;

    @InjectMocks
    private Util util;

    @Test
    public void testStringCompare() {
        // Arrange
        when(lhs).thenReturn("TestString");
        // Act
        int result = util.StringCompare(lhs, rhs);
        // Assert
        assertEquals(0, result);
    }
}
