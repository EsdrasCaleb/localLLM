package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_ioError_4_4_Test {

    @InjectMocks
    private AnyWrapperMsgGenerator anyWrapperMsgGenerator;

    @Test
    public void testIoError() {
        // Arrange
        Exception exception = new Exception("Test exception message");
        // Act
        String errorMessage = anyWrapperMsgGenerator.ioError(exception);
        // Assert
        assertEquals("Test exception message", errorMessage);
    }

    @Test
    public void testIoErrorNullException() {
        // Arrange
        Exception exception = null;
        // Act and Assert
        NullPointerException expectedException = null;
        try {
            anyWrapperMsgGenerator.ioError(exception);
            fail("Expected NullPointerException was not thrown");
        } catch (NullPointerException e) {
            assertEquals(expectedException, e);
        }
    }

    @Test
    public void testIoErrorEmptyException() {
        // Arrange
        Exception exception = new Exception();
        // Act and Assert
        assertEquals("", anyWrapperMsgGenerator.ioError(exception));
    }

    @Test
    public void testIoErrorCustomException() {
        // Arrange
        Exception exception = new CustomException("Test custom exception message");
        // Act
        String errorMessage = anyWrapperMsgGenerator.ioError(exception);
        // Assert
        assertEquals("Test custom exception message", errorMessage);
    }

    @Test
    public void testIoErrorError() {
        // Arrange
        Exception error = new Exception("Test error message");
        // Act
        String errorMessage = anyWrapperMsgGenerator.ioError(error);
        // Assert
        assertEquals("Test error message", errorMessage);
    }
}

class CustomException extends Exception {

    public CustomException(String message) {
        super(message);
    }
}
