package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_1_3_Test {

    @Test
    public void testErrorMethod() {
        // Arrange
        String inputStr = "Test error message";
        // Act
        String actualResult = AnyWrapperMsgGenerator.error(inputStr);
        // Assert
        assertEquals(inputStr, actualResult);
    }

    @Test
    public void testErrorMethod_EmptyString() {
        // Arrange
        String inputStr = "";
        // Act
        String actualResult = AnyWrapperMsgGenerator.error(inputStr);
        // Assert
        assertEquals(inputStr, actualResult);
    }

    @Test
    public void testErrorMethod_NullInput() {
        // Arrange
        String inputStr = null;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> AnyWrapperMsgGenerator.error(inputStr));
    }
}
