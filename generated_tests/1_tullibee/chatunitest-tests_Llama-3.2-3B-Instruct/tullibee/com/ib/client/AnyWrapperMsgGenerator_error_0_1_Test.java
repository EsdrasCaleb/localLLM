package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_0_1_Test {

    @Test
    public void testErrorMethod() {
        // Arrange
        Exception exception = new Exception("Test Exception Message");
        // Act
        String errorMessage = AnyWrapperMsgGenerator.error(exception);
        // Assert
        assertEquals("Error - Test Exception Message", errorMessage);
    }

    @Test
    public void testErrorMethodNullInput() {
        // Arrange
        Exception exception = null;
        // Act and Assert
        String errorMessage = AnyWrapperMsgGenerator.error(exception);
        assertEquals("Error - null", errorMessage);
    }

    @Test
    public void testErrorMethodEmptyMessage() {
        // Arrange
        Exception exception = new Exception();
        // Act and Assert
        String errorMessage = AnyWrapperMsgGenerator.error(exception);
        assertEquals("Error - ", errorMessage);
    }
}
