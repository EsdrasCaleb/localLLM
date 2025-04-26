package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    public void testError_ValidInputs() {
        // Arrange
        int id = 1;
        int errorCode = 404;
        String errorMsg = "Not Found";
        // Act
        String result = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        // Assert
        assertEquals("1 | 404 | Not Found", result);
    }

    @Test
    public void testError_ZeroValues() {
        // Arrange
        int id = 0;
        int errorCode = 0;
        String errorMsg = "No Error";
        // Act
        String result = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        // Assert
        assertEquals("0 | 0 | No Error", result);
    }

    @Test
    public void testError_NegativeValues() {
        // Arrange
        int id = -1;
        int errorCode = -404;
        String errorMsg = "Not Found";
        // Act
        String result = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        // Assert
        assertEquals("-1 | -404 | Not Found", result);
    }

    @Test
    public void testError_EmptyMessage() {
        // Arrange
        int id = 1;
        int errorCode = 500;
        String errorMsg = "";
        // Act
        String result = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        // Assert
        assertEquals("1 | 500 | ", result);
    }

    @Test
    public void testError_NullMessage() {
        // Arrange
        int id = 1;
        int errorCode = 400;
        String errorMsg = null;
        // Act
        String result = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        // Assert
        assertEquals("1 | 400 | null", result);
    }
}
