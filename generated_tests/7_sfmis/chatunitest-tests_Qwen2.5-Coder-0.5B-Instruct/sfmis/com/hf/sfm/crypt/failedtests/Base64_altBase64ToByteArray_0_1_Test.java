// Test method
package com.hf.sfm.crypt;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Base64_altBase64ToByteArray_0_1_Test {

    @Test
    public void testAltBase64ToByteArray() {
        // Arrange
        String input = "Hello, World!";
        String expectedOutput = "SGVsbG8gV29ybGQ=";
        // Act
        byte[] actualOutput = Base64.altBase64ToByteArray(input);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @ExtendWith(MockitoExtension.class)
    public class TestHelper {

        private Base64 base64;

        @BeforeEach
        public void setUp() {
            this.base64 = Mockito.mock(Base64.class);
        }

        @AfterEach
        public void tearDown() {
            Mockito.reset(this.base64);
        }

        @Test
        public void testAltBase64ToByteArray() {
            // Arrange
            String input = "Hello, World!";
            String expectedOutput = "SGVsbG8gV29ybGQ=";
            // Act
            byte[] actualOutput = base64.altBase64ToByteArray(input);
            // Assert
            assertEquals(expectedOutput, actualOutput);
        }
    }
}
