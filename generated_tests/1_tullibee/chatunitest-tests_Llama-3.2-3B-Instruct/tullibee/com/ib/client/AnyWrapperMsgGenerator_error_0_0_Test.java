package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_0_0_Test {

    @Test
    public void testError_NullPointerException() {
        Exception ex = new NullPointerException();
        String result = AnyWrapperMsgGenerator.error(ex);
        assertEquals("Error - null pointer exception", result);
    }

    @Test
    public void testError_InvalidException() {
        Exception ex = new Exception("Test Exception");
        String result = AnyWrapperMsgGenerator.error(ex);
        assertEquals("Error - Test Exception", result);
    }

    @Test
    public void testError_EmptyException() {
        Exception ex = new Exception();
        String result = AnyWrapperMsgGenerator.error(ex);
        assertEquals("Error - ", result);
    }

    @Test
    public void testError_NullException() {
        Exception ex = null;
        String result = AnyWrapperMsgGenerator.error(ex);
        assertEquals("Error - null", result);
    }
}
