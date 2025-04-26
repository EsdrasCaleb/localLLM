package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_ioError_4_1_Test {

    @Test
    public void testIoError() {
        Exception ex = mock(Exception.class);
        when(ex.getMessage()).thenReturn("Test Exception");
        String errorMessage = AnyWrapperMsgGenerator.ioError(ex);
        assertEquals("Error: Test Exception", errorMessage);
    }
}
