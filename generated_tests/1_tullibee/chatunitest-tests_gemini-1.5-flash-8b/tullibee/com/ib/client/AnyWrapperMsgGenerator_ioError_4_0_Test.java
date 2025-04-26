package com.ib.client;

import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_ioError_4_0_Test {

    @Test
    void ioError_withIOException_returnsExpectedMessage() {
        IOException ioException = Mockito.mock(IOException.class);
        Mockito.when(ioException.getMessage()).thenReturn("Simulated I/O error");
        String expectedMessage = "Simulated I/O error";
        String actualMessage = AnyWrapperMsgGenerator.ioError(ioException);
        Assertions.assertEquals(expectedMessage, actualMessage);
    }

    @Test
    void ioError_withNullException_returnsExpectedMessage() {
        Exception ex = null;
        String expectedMessage = "Exception is null";
        String actualMessage = AnyWrapperMsgGenerator.ioError(ex);
        Assertions.assertEquals(expectedMessage, actualMessage);
    }

    @Test
    void ioError_withDifferentException_returnsExpectedMessage() {
        Exception ex = new RuntimeException("Different error");
        String expectedMessage = "Different error";
        String actualMessage = AnyWrapperMsgGenerator.ioError(ex);
        Assertions.assertEquals(expectedMessage, actualMessage);
    }

    // Helper method (crucial for testing the error() method).
    private String error(Exception ex) {
        if (ex == null) {
            return "Exception is null";
        }
        return ex.getMessage();
    }
}
