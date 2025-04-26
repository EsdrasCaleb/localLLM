package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Util_StringIsEmpty_0_2_Test {

    @ExtendWith(MockitoExtension.class)
    public class Util_StringIsEmptyTest {

        @Test
        public void testStringIsEmpty() {
            // Arrange
            String str = "Hello";
            Mockito.when(Util.StringIsEmpty(str)).thenReturn(true);
            // Act
            boolean result = Util.StringIsEmpty(str);
            // Assert
            assertEquals(true, result);
        }
    }
}
