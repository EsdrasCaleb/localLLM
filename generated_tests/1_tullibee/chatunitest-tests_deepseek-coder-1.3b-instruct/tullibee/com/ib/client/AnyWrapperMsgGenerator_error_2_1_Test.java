package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_2_1_Test {

    @Test
    public void testError() {
        // Arrange
        int id = 123;
        int errorCode = 404;
        String errorMsg = "Not Found";
        // Act
        String result = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        // Assert
        assertEquals("123 | 404 | Not Found", result);
    }
}
