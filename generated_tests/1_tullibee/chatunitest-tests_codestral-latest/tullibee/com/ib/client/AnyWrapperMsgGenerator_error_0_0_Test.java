package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_0_0_Test {

    @Test
    public void testErrorWithNullException() {
        Exception ex = null;
        assertThrows(NullPointerException.class, () -> AnyWrapperMsgGenerator.error(ex));
    }

    @Test
    public void testErrorWithValidException() {
        Exception ex = new Exception("Test Exception");
        String expectedMessage = "Error - java.lang.Exception: Test Exception";
        String actualMessage = AnyWrapperMsgGenerator.error(ex);
        assertEquals(expectedMessage, actualMessage);
    }
}
