package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_ioError_4_0_Test {

    @Test
    void ioErrorTest() {
        // Arrange
        Exception ex = new Exception("Test Exception");
        // Act
        String result = AnyWrapperMsgGenerator.ioError(ex);
        // Assert
        // 1.
        assertNotNull(result);
        // 2.
        assertFalse(result.isEmpty());
        // 3.
        assertTrue(result.matches(".*\\S.*"));
    }
}
