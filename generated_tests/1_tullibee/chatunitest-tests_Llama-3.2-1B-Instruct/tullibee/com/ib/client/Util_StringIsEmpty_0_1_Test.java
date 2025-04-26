package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_StringIsEmpty_0_1_Test {

    @Mock
    private String str;

    @InjectMocks
    private Util util;

    @Test
    public void testStringIsEmpty() {
        // Arrange
        when(str).thenReturn("");
        // Act
        boolean result = util.StringIsEmpty(str);
        // Assert
        assertTrue(result);
    }
}
