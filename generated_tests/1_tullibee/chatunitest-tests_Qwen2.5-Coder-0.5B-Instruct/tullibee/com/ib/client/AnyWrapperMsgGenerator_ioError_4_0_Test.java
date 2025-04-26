package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_ioError_4_0_Test {

    @Test
    public void testIoError() throws Exception {
        // Mocking the exception
        Exception mockException = Mockito.mock(Exception.class);
        // Stubbing the error method
        Mockito.when(mockException.getMessage()).thenReturn("An error occurred");
        // Calling the method under test
        String result = AnyWrapperMsgGenerator.ioError(mockException);
        // Asserting the result
        assertEquals("An error occurred", result);
    }
}
