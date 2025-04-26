package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_0_0_Test {

    @Test
    public void testError_withNullException() {
        Exception ex = null;
        String result = AnyWrapperMsgGenerator.error(ex);
        assertEquals("Error - null", result);
    }

    @Test
    public void testError_withRuntimeException() {
        Exception ex = new RuntimeException("Runtime exception occurred");
        String result = AnyWrapperMsgGenerator.error(ex);
        assertEquals("Error - java.lang.RuntimeException: Runtime exception occurred", result);
    }

    @Test
    public void testError_withCheckedException() {
        Exception ex = new Exception("Checked exception occurred");
        String result = AnyWrapperMsgGenerator.error(ex);
        assertEquals("Error - java.lang.Exception: Checked exception occurred", result);
    }

    @Test
    public void testError_withCustomException() {
        class CustomException extends Exception {

            public CustomException(String message) {
                super(message);
            }
        }
        Exception ex = new CustomException("Custom exception occurred");
        String result = AnyWrapperMsgGenerator.error(ex);
        assertEquals("Error - AnyWrapperMsgGeneratorTest$1: Custom exception occurred", result);
    }
}
