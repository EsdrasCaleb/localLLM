package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompare_2_1_Test {

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testStringCompare() {
        // Arrange
        String lhs = "Hello";
        String rhs = "World";
        // Act
        int result = Util.StringCompare(lhs, rhs);
        // Assert
        // Since 'H' comes before 'W', the comparison should be -1
        assertEquals(1, result);
    }
}
